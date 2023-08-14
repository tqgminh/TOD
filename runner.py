"""
   MTTOD: runner.py

   implements train and predict function for MTTOD model.

   Copyright 2021 ETRI LIRS, Yohan Lee

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import re
import copy
import math
import time
import json
import glob
import shutil
from abc import *
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput
from tensorboardX import SummaryWriter

from model import T5WithSpan, T5WithTokenSpan
from reader import MultiWOZIterator, MultiWOZReader
from evaluator import MultiWozEvaluator

from utils import definitions
from utils.io_utils import get_or_create_logger, load_json, save_json


logger = get_or_create_logger(__name__)

def load_dbs(db_path=None, num_examples=None):
    """load all dbs in all gpus"""
    dbs = json.loads(open(db_path, 'r', encoding='utf-8').read().lower())
    if num_examples is not None:
        dbs = dbs[:num_examples]
    return dbs

def entity_to_text_wo_dk(entity):
    text = "<database>"
    for key, val in entity.items():
        if val != "dontknow":
            text += f" {key} {val} <sep_attributes>"
    return text

def entity_to_text_w_dk_mask(entity, dk_mask=False):
    text = "<database>"
    mask = []
    for key, val in entity.items():
        text += f" {key} {val} <sep_attributes>"
        if val == "dontknow":
            mask.append(0)
        else:
            mask.append(1)
    if dk_mask is True:
        return text, mask
    else:
        return text, None

def padSeqs(sequences, maxlen=None, truncated='max_len', pad_method='post', trunc_method='pre', dtype='int32',
            value=0.):
    assert truncated in ['max_len', 'batch_max_len']
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    # if maxlen is not None and truncated:
    #     maxlen = min(seq_maxlen, maxlen)
    # else:
    #     maxlen = seq_maxlen
    if truncated == 'max_len':
        maxlen = maxlen
    else:
        maxlen = min(maxlen, seq_maxlen)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x

class DBDataset(torch.utils.data.Dataset):
    """use for database text"""

    def __init__(self, dbs, db_type="entrance", use_dk=False, dk_mask=False):
        self.dbs = dbs
        self.db_type = db_type
        self.use_dk = use_dk
        self.dk_mask = dk_mask

    def __len__(self):
        return len(self.dbs)

    def __getitem__(self, index):
        example = self.dbs[index]
        if self.db_type == "entrance":
            if self.use_dk is False:
                text = entity_to_text_wo_dk(example)
                return_dict = {"db_index": index, "db_text": text}
            else:
                text, mask = entity_to_text_w_dk_mask(example, dk_mask=self.dk_mask)
                if self.dk_mask is True:
                    return_dict = {"db_index": index, "db_text": text, "db_mask": mask}
                else:
                    return_dict = {"db_index": index, "db_text": text}
        else:
            raise ValueError
        return return_dict

class DBCollator(object):
    """use for database text"""

    def __init__(self, tokenizer, maxlength, type='generator'):
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.pad_id = 0
        if len(self.tokenizer.encode("<sep_attributes>")) == 1:
            self.sep_id = self.tokenizer.encode("<sep_attributes>")[0]
            self.db_id = self.tokenizer.encode("<database>")[0]
        else:
            self.sep_id = self.tokenizer.encode("<sep_attributes>")[1]
            self.db_id = self.tokenizer.encode("<database>")[1]
        self.type = type

    def __call__(self, batch):
        index = [x["db_index"] for x in batch]
        text = [x["db_text"] for x in batch]
        text = self.tokenizer.batch_encode_plus(text)
        text_ids = torch.tensor(padSeqs(text['input_ids'],
                                         maxlen=self.maxlength, truncated='max_len',
                                         pad_method='post', trunc_method='pre', dtype='int32',
                                         value=self.pad_id))
        text_mask = torch.tensor(padSeqs(text['attention_mask'],
                                         maxlen=self.maxlength, truncated='max_len',
                                         pad_method='post', trunc_method='pre', dtype='int32',
                                         value=self.pad_id)).bool()
        if "db_mask" in batch[0] and self.type == "generator":  # dk_mask only for generator
            attr_mask = [x["db_mask"] for x in batch]
            attr_mask = torch.tensor(attr_mask).bool()
            sep_index = text_ids.eq(self.sep_id).nonzero()
            db_index = text_ids.eq(self.db_id).nonzero()
            attr_num = attr_mask.size(1)
            assert len(sep_index) % attr_num == 0
            attr_true_mask = torch.ones_like(text_mask).bool()
            for i, idx in enumerate(sep_index):
                if i % attr_num == 0:
                    start_idx = db_index[sep_index[i][0]][1]
                else:
                    start_idx = sep_index[i - 1][1] + 1
                end_idx = sep_index[i][1] + 1
                attr_true_mask[sep_index[i][0], start_idx: end_idx] = attr_mask[sep_index[i][0], i % attr_num]
            text_mask = text_mask * attr_true_mask
        else:
            attr_mask = None
        if "token_type_ids" in text:
            text_token_type = torch.tensor(padSeqs(text['token_type_ids'],
                                                   maxlen=self.maxlength, truncated='max_len',
                                                   pad_method='post', trunc_method='pre', dtype='int32',
                                                   value=1))
        else:
            text_token_type = None
        return index, text_ids, text_mask, text_token_type, attr_mask

class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.span_loss = 0.0
        self.resp_loss = 0.0

        self.belief_correct = 0.0
        self.span_correct = 0.0
        self.resp_correct = 0.0

        self.belief_count = 0.0
        self.span_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.belief_loss += step_outputs["belief"]["loss"]
        self.belief_correct += step_outputs["belief"]["correct"]
        self.belief_count += step_outputs["belief"]["count"]

        if "span" in step_outputs:
            self.span_loss += step_outputs["span"]["loss"]
            self.span_correct += step_outputs["span"]["correct"]
            self.span_count += step_outputs["span"]["count"]

            do_span_stats = True
        else:
            do_span_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]

            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_span_stats, do_resp_stats)

    def info_stats(self, data_type, global_step, do_span_stats=False, do_resp_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        belief_ppl = math.exp(self.belief_loss / self.belief_count)
        belief_acc = (self.belief_correct / self.belief_count) * 100

        self.summary_writer.add_scalar(
            "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.belief_loss, belief_ppl, belief_acc)

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.resp_loss, resp_ppl, resp_acc)
        else:
            resp_info = ""

        if do_span_stats:
            if self.span_count == 0:
                span_acc = 0.0
            else:
                span_acc = (self.span_correct / self.span_count) * 100

            self.summary_writer.add_scalar(
                "{}/span_loss".format(data_type), self.span_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/span_acc".format(data_type), span_acc, global_step=global_step)

            span_info = "[span] loss {0:.2f}; acc {1:.2f};".format(
                self.span_loss, span_acc)

        else:
            span_info = ""

        logger.info(
            " ".join([common_info, belief_info, resp_info, span_info]))

        self.init_stats()
