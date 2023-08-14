    def predict(self):
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context, _ = self.iterator.flatten_dial_history(
                        dial_history[t], [], len(turn["user"]), self.cfg.context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(
                    batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                # belief tracking
                with torch.no_grad():
                    encoder_outputs = self.model(input_ids=batch_encoder_input_ids,
                                                 attention_mask=attention_mask,
                                                 return_dict=False,
                                                 encoder_only=True,
                                                 add_auxiliary_task=self.cfg.add_auxiliary_task)

                    span_outputs, encoder_hidden_states = encoder_outputs

                    if isinstance(encoder_hidden_states, tuple):
                        last_hidden_state = encoder_hidden_states[0]
                    else:
                        last_hidden_state = encoder_hidden_states

                    # wrap up encoder outputs
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state)

                    belief_outputs = self.model.generate(encoder_outputs=encoder_outputs,
                                                         attention_mask=attention_mask,
                                                         eos_token_id=self.reader.eos_token_id,
                                                         max_length=200,
                                                         do_sample=self.cfg.do_sample,
                                                         num_beams=self.cfg.beam_size,
                                                         early_stopping=early_stopping,
                                                         temperature=self.cfg.temperature,
                                                         top_k=self.cfg.top_k,
                                                         top_p=self.cfg.top_p,
                                                         decoder_type="belief")

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                if self.cfg.add_auxiliary_task:
                    pred_spans = span_outputs[1].cpu().numpy().tolist()

                    input_ids = batch_encoder_input_ids.cpu().numpy().tolist()
                else:
                    pred_spans = None
                    input_ids = None

                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs, domain_history, constraint_dicts, pred_spans, input_ids)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])
                    '''
                    print(self.reader.tokenizer.decode(input_ids[t]))
                    print(self.reader.tokenizer.decode(turn["bspn_gen"]))
                    print(turn["span"])
                    print(self.reader.tokenizer.decode(turn["bspn_gen_with_span"]))
                    input()
                    '''

                if self.cfg.task == "e2e":
                    dbpn = []

                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for turn in turn_batch:
                            if self.cfg.add_auxiliary_task:
                                bspn_gen = turn["bspn_gen_with_span"]
                            else:
                                bspn_gen = turn["bspn_gen"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)

                            db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)

                            turn["dbpn_gen"] = dbpn_gen

                            dbpn.append(dbpn_gen)

                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db

                    #print(dbpn)

                    # aspn has different length
                    if self.cfg.use_true_curr_aspn:
                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = self.iterator.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state[t].unsqueeze(0))

                            with torch.no_grad():
                                resp_outputs = self.model.generate(
                                    encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=300,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    decoder_type="resp")

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])

                    else:
                        resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                        # response generation
                        with torch.no_grad():
                            resp_outputs = self.model.generate(
                                encoder_outputs=encoder_outputs,
                                attention_mask=attention_mask,
                                decoder_input_ids=resp_decoder_input_ids,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=300,
                                do_sample=self.cfg.do_sample,
                                num_beams=self.cfg.beam_size,
                                early_stopping=early_stopping,
                                temperature=self.cfg.temperature,
                                top_k=self.cfg.top_k,
                                top_p=self.cfg.top_p,
                                decoder_type="resp")

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)

                        for t, turn in enumerate(turn_batch):
                            turn.update(**decoded_resp_outputs[t])

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        if self.cfg.add_auxiliary_task:
                            pv_bspn = turn["bspn_gen_with_span"]
                        else:
                            pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if self.cfg.use_true_prev_resp:
                        if self.cfg.task == "e2e":
                            pv_resp = turn["redx"]
                        else:
                            pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)

        if self.cfg.task == "e2e":
            bleu, success, match = evaluator.e2e_eval(
                results, eval_dial_list=eval_dial_list, add_auxiliary_task=self.cfg.add_auxiliary_task)

            score = 0.5 * (success + match) + bleu

            logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                match, success, bleu, score))
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))

if __name__ == "__main__":
    reader = MultiWOZReader("t5-small", "2.1")
    print(reader.tokenizer)
