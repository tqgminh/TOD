class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        reader = MultiWOZReader(cfg.backbone, cfg.version)

        self.iterator = MultiWOZIterator(reader)

        super(MultiWOZRunner, self).__init__(cfg, reader)

    def step_fn(self, inputs, entities, span_labels, belief_labels, resp_labels):
        inputs = inputs.to(self.cfg.device)
        entities = entities.to(self.cfg.device)
        span_labels = span_labels.to(self.cfg.device)
        belief_labels = belief_labels.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)

        attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)
        entities_attention_mask = torch.where(entities == self.reader.pad_token_id, 0, 1)

        belief_outputs = self.model(input_ids=inputs,
                                    attention_mask=attention_mask,
                                    span_labels=span_labels,
                                    lm_labels=belief_labels,
                                    return_dict=False,
                                    add_auxiliary_task=self.cfg.add_auxiliary_task,
                                    decoder_type="belief")

        encoder = self.model.encoder
        encoder.eval()
        with torch.no_grad():
            entity_embeddings = encoder(input_ids=entities,
                                        attention_mask=entities_attention_mask,
                                        return_dict=False)[0]
        encoder.train()
        entity_embeddings = torch.mean(entity_embeddings, dim=1)

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        span_loss = belief_outputs[2]
        span_pred = belief_outputs[3]

        context_embeddings = torch.mean(belief_outputs[5], dim=1)
        # print(encoder_hidden_states.shape)
        # print(entities_embedding)

        retriever_entity_scores = torch.einsum("bd,nd->bn", context_embeddings, entity_embeddings)
        retriever_top_k_entity_indexes = retriever_entity_scores.sort(-1, True)[1][:, :1].unsqueeze(2) # top 1

        bsz = retriever_top_k_entity_indexes.size(0)
        num_entities = entities.size(-1)
        top_k_entity_ids = torch.gather(entities.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                        retriever_top_k_entity_indexes.long().repeat(1, 1, num_entities))
        top_k_entity_mask = torch.gather(entities_attention_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                         retriever_top_k_entity_indexes.long().repeat(1, 1, num_entities))
        context_top_k_entity_input_ids = self.iterator.concat_context_and_entity_input(inputs, top_k_entity_ids)
        context_top_k_entity_mask = self.iterator.concat_context_and_entity_input(attention_mask, top_k_entity_mask)

        context_top_k_entity_input_ids = context_top_k_entity_input_ids.view(context_top_k_entity_input_ids.size(0), -1)
        context_top_k_entity_mask = context_top_k_entity_mask.view(context_top_k_entity_mask.size(0), -1)

        resp_outputs = self.model(input_ids=context_top_k_entity_input_ids,
                                  attention_mask=context_top_k_entity_mask,
                                  lm_labels=resp_labels,
                                  return_dict=False,
                                  decoder_type="resp")
        
        resp_loss = resp_outputs[0]
        resp_pred = resp_outputs[1]

        num_resp_correct, num_resp_count = self.count_tokens(
            resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        # if self.cfg.task == "e2e":
        #     last_hidden_state = belief_outputs[5]

        #     encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)

        #     resp_outputs = self.model(attention_mask=attention_mask,
        #                               encoder_outputs=encoder_outputs,
        #                               lm_labels=resp_labels,
        #                               return_dict=False,
        #                               decoder_type="resp")

        #     resp_loss = resp_outputs[0]
        #     resp_pred = resp_outputs[1]

        #     num_resp_correct, num_resp_count = self.count_tokens(
        #         resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        if self.cfg.add_auxiliary_task:
            num_span_correct, num_span_count = self.count_tokens(
                span_pred, span_labels, pad_id=0)

        loss = belief_loss

        if self.cfg.add_auxiliary_task and self.cfg.aux_loss_coeff > 0:
            loss += (self.cfg.aux_loss_coeff * span_loss)

        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)

        '''
        if self.cfg.num_gpus > 1:
            loss = loss.sum()
            belief_loss = belief_loss.sum()
            num_belief_correct = num_belief_correct.sum()
            num_belief_count = num_belief_count.sum()

            if self.cfg.add_auxiliary_task:
                span_loss = span_loss.sum()
                num_span_correct = num_span_correct.sum()
                num_span_count = num_span_count.sum()

            if self.cfg.task == "e2e":
                resp_loss = resp_loss.sum()
                num_resp_correct = num_resp_correct.sum()
                num_resp_count = num_resp_count.sum()
        '''

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        if self.cfg.add_auxiliary_task:
            step_outputs["span"] = {"loss": span_loss.item(),
                                    "correct": num_span_correct.item(),
                                    "count": num_span_count.item()}

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}

        return loss, step_outputs

    def train_epoch(self, train_iterator, optimizer, scheduler, reporter=None, entity_tensor=None):
        self.model.train()
        self.model.zero_grad()

        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            inputs, labels = batch

            _, belief_labels, _ = labels

            loss, step_outputs = self.step_fn(inputs, entity_tensor, *labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

    def train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            "train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
            num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)
        
        entities = self.iterator.get_entities(excluded_domains=self.cfg.excluded_domains)
        entities_tensor = self.iterator.get_entities_tensor(entities)

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = self.iterator.get_data_iterator(
                train_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, reporter, entities_tensor)

            logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            if not self.cfg.no_validation:
                self.validation(reporter.global_step)

    def validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

        dev_iterator = self.iterator.get_data_iterator(
            dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size)

        entities = self.iterator.get_entities(excluded_domains=self.cfg.excluded_domains)
        entity_tensor = self.iterator.get_entities_tensor(entities)

        reporter = Reporter(1000000, self.cfg.model_dir)

        torch.set_grad_enabled(False)
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validation"):
            start_time = time.time()

            inputs, labels = batch

            _, step_outputs = self.step_fn(inputs, entity_tensor, *labels)

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_span_stats = True if "span" in step_outputs else False
        do_resp_stats = True if "resp" in step_outputs else False

        reporter.info_stats("dev", global_step, do_span_stats, do_resp_stats)

        torch.set_grad_enabled(True)
