class Trainer():
    def __init__(
            self,
            model=None,
            args: TrainingArguments = None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            optimizers=(None, None),
            callbacks=None,
            preprocess_logits_for_metrics=None,
    ):
        self.args = args

        with CodeBlock("初始化随机种子"):
            pass  # todo: 初始化随机种子

        with CodeBlock("实例化accelerator"):
            self.create_accelerator_and_postprocess()
            args._setup_devices  # 设置device、n_gpu

        with CodeBlock("model、tokenizer预处理"):
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()

            ############### 判断模型是否被分布加载到多个gpu ###############
            self.is_model_parallel = False
            if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
                self.is_model_parallel = True
            if getattr(model, "hf_device_map", None) is not None:
                devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
                self.is_model_parallel = False
                if len(devices) > 1:
                    self.is_model_parallel = True
                elif len(devices) == 1:
                    self.is_model_parallel = self.args.device != torch.device(devices[0])

            ############### 将模型移动到device ###############
            self.place_model_on_device = args.place_model_on_device
            if self.is_model_parallel or self.is_deepspeed_enabled or self.is_fsdp_enabled:
                self.place_model_on_device = False
            if self.place_model_on_device:
                self._move_model_to_device(model, args.device)

            self.model = model
            self.tokenizer = tokenizer

        with CodeBlock("dataset、collator预处理"):
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            default_collator = (
                DataCollatorWithPadding(tokenizer)
                if tokenizer is not None and isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
                else default_data_collator
            )
            self.data_collator = data_collator if data_collator is not None else default_collator

        with CodeBlock('optimzier, scheduler处理'):
            self.optimizer, self.lr_scheduler = optimizers

        with CodeBlock("其他参数初始化"):
            ############### 判断model.forward()的输入参数 ###############
            self._signature_columns = None
            default_label_names = find_labels(self.model.__class__)
            self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

            ############### 评测相关的参数 ###############
            self.compute_metrics = compute_metrics
            self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

            ############### label smoothing ###############
            if self.args.label_smoothing_factor != 0:
                self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
            else:
                self.label_smoother = None

            ############### neftune ###############
            self.neftune_noise_alpha = args.neftune_noise_alpha

            self.is_in_train = False

        with CodeBlock("参数异常校验"):
            if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and (self.optimizer is not None or self.lr_scheduler is not None):
                raise RuntimeError('使用deepspeed或fsdp时要求动态创建optimizer、scheduler，可以继承Trainer并修改create_optimizer、create_scheduler方法')
            if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
                raise RuntimeError('使用model_init()时要求动态创建optimizer、scheduler，可以继承Trainer并修改create_optimizer、create_scheduler方法')

        with CodeBlock("state、control、callbacks初始化"):
            ############### TrainerState, TrainerControl初始化 ###############
            self.state = TrainerState(
                is_local_process_zero=self.is_local_process_zero(),
                is_world_process_zero=self.is_world_process_zero(),
            )
            self.control = TrainerControl()

            ############### callbacks初始化 ###############
            # DefaultFlowCallback:
            # 1. 根据设置的logging_strategy, evaluation_strategy, save_strategy等触发should_log, should_evaluate, should_save
            # 2. 根据训练steps触发control.should_training_stop
            default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
            callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
            self.callback_handler = CallbackHandler(callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler)
            self.callback_handler.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)

            ############### 触发on_init_end回调 ###############
            self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def __训练流程__(self):
        pass

    def train(self, ignore_keys_for_eval=None, resume_from_checkpoint=None, **kwargs):
        self.is_in_train = True
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        with CodeBlock("训练循环"):
            train_output = self._inner_training_loop(
                batch_size=self.args.train_batch_size,
                args=self.args,
                ignore_keys_for_eval=ignore_keys_for_eval,
                resume_from_checkpoint=kwargs.pop('model_path', resume_from_checkpoint),
            )

        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)
        self.is_in_train = False

        return train_output

    def _inner_training_loop(self, batch_size=None, args=None,
                             ignore_keys_for_eval=None, resume_from_checkpoint=None):
        self.accelerator.free_memory()  # 清除变量引用和显存cache

        with CodeBlock("初始化dataloader"):
            self._train_batch_size = batch_size
            train_dataloader = self.get_train_dataloader()

        with CodeBlock("确定训练steps、epochs"):
            assert has_length(train_dataloader) or args.max_steps > 0, "train_dataloader长度未知情况下，必须设置max_steps，以正确终止训练"
            assert args.num_train_epochs > 0 or args.max_steps > 0, "max_steps或num_train_epochs必须至少设置一项，以正确终止训练"

            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

            if has_length(train_dataloader):
                ############### 数据集长度已知 ###############
                num_examples = self.num_examples(train_dataloader)
                num_update_steps_per_epoch = max(len(train_dataloader) // args.gradient_accumulation_steps, 1)
                if args.max_steps > 0:
                    ############### 用户设定了max_steps ###############
                    max_steps = args.max_steps
                    num_train_epochs = max_steps // num_update_steps_per_epoch + int(max_steps % num_update_steps_per_epoch > 0)
                    num_train_samples = args.max_steps * total_train_batch_size
                else:
                    ############### 用户设定num_train_epochs ###############
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
            else:
                ############### 数据集长度未知 ###############
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps = args.max_steps
                num_examples = num_train_samples = total_train_batch_size * args.max_steps

        with CodeBlock("初始化optimizer、scheduler"):
            ############### deepspeed: 初始化DummyOptim或调用create_optimzier() ###############
            if self.is_deepspeed_enabled:
                # 注意！deepspeed_init与trainer的耦合性很强
                # 会调用trainer.model, trainer.args
                # 会调用trainer.create_optimizer(), trainer.create_scheduler(), 并设置trainer.optimizer, trainer.scheduler
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            ############### fsdp: 需要在model包装之后创建optimizer、scheduler ###############
            delay_optimizer_creation = self.is_fsdp_enabled
            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        with CodeBlock("使用accelerator包装model、optimizer、scheduler"):
            self.model_wrapped = self.model
            ############### DP、DDP处理 ###############
            # 在_wrap_model内，如果n_gpu>1，用torch.nn.DataParallel (后续也不需要使用accelerator)
            # 否则只进行accelerate的DistributedDataParallelKwargs配置，由后面accelerator.prepare()完成模型包装
            model = self._wrap_model(self.model_wrapped)
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                ############### 先处理model，再创建optimizer、scheduler ###############
                if use_accelerator_prepare:
                    self._fsdp_qlora_plugin_updates()
                    self.model = self.accelerator.prepare_model(self.model)
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            if use_accelerator_prepare:
                ############### 对model、optimizer、scheduler应用accelerator.prepare ###############
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                else:
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)

            ############### 处理model、self.model、self.model_wrapped ###############
            if self.is_fsdp_enabled:
                self.model = self.model_wrapped = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        with CodeBlock("加载ckpt"):
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
            if resume_from_checkpoint is not None:
                ############### 加载模型参数 ###############
                if self.is_deepspeed_enabled or self.is_fsdp_enabled:
                    self._load_from_checkpoint(resume_from_checkpoint, model=self.model_wrapped)
                else:
                    self._load_from_checkpoint(resume_from_checkpoint)

                ############### 加载optimizer、scheduler参数 ###############
                self._load_optimizer_and_scheduler(resume_from_checkpoint)

        with CodeBlock("恢复训练进度"):
            if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                assert self.state.train_batch_size == self._train_batch_size  # 确保batch_size没有改变
                # TODO: compare_trainer_and_checkpoint_args

                ############### 恢复epoch ###############
                epochs_trained = self.state.global_step // num_update_steps_per_epoch

                ############### 恢复step ###############
                if args.ignore_data_skip:
                    steps_trained_in_current_epoch = 0
                else:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    ############### 模拟前epochs_trained次数据采样，以恢复sampler的rng_state ###############
                    logger.info(f"  Will skip the first {epochs_trained} epochs")
                    for epoch in range(epochs_trained):
                        sampler = get_dataloader_sampler(train_dataloader)
                        sampler_kinds = [RandomSampler]
                        if version.parse(accelerate_version) > version.parse("0.23.0"):
                            sampler_kinds.append(SeedableRandomSampler)
                        is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                        if not is_random_sampler:
                            for _ in train_dataloader:  # TODO: 没明白非随机sampler为什么需要模拟数据采样
                                break
                        else:
                            sampler = sampler if sampler is not None else []
                            _ = list(sampler)

        with CodeBlock('更新state, control, callbacks'):
            def maybe_abs_or_ratio(abs_or_ratio, base_value, default):
                if abs_or_ratio is None: return default
                return base_value * abs_or_ratio if abs_or_ratio < 1 else abs_or_ratio

            self.state.logging_steps = maybe_abs_or_ratio(args.logging_steps, max_steps, default=self.state.logging_steps)
            self.state.eval_steps = maybe_abs_or_ratio(args.eval_steps, max_steps, default=self.state.eval_steps)
            self.state.save_steps = maybe_abs_or_ratio(args.save_steps, max_steps, default=self.state.save_steps)
            self.state.epoch = 0
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.train_batch_size = self._train_batch_size
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader

            # 训练开始
            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        with CodeBlock('训练循环'):
            with CodeBlock('变量初始化'):
                tr_loss = torch.tensor(0.0).to(args.device)
                self._total_loss_scalar = 0.0
                self.current_flos = 0
                # epochs_trained = 0
                total_batched_samples = 0
                self._globalstep_last_logged = self.state.global_step

            ############### epoch循环 ###############
            for epoch in range(epochs_trained, num_train_epochs):
                # 开始epoch训练
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                epoch_iterator = train_dataloader
                if hasattr(epoch_iterator, "set_epoch"):
                    epoch_iterator.set_epoch(epoch)

                steps_in_epoch = (
                    len(epoch_iterator)
                    if has_length(epoch_iterator)
                    else args.max_steps * args.gradient_accumulation_steps
                )
                ############### 恢复训练进度: 跳过steps_trained_in_current_epoch ###############
                steps_skipped = 0
                if epoch == epochs_trained and resume_from_checkpoint is not None:
                    if steps_trained_in_current_epoch > 0:
                        epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                        steps_skipped = steps_trained_in_current_epoch

                    self._load_rng_state(resume_from_checkpoint)

                ############### step循环 ###############
                for step, inputs in enumerate(epoch_iterator):
                    ############### 单次forward、backward ###############
                    total_batched_samples += 1

                    ############### 记录tokens数 ###############
                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name in inputs:
                            input_device = inputs[main_input_name].device
                            self.state.num_input_tokens_seen += torch.sum(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            ).item()
                    if step % args.gradient_accumulation_steps == 0:
                        # 开始step训练
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    model.zero_grad()
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)
                    tr_loss += tr_loss_step
                    self.current_flos += float(self.floating_point_ops(inputs))

                    ############### 模型参数更新 ###############
                    is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )
                    if (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                            or
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            is_last_step_and_steps_less_than_grad_acc
                    ):
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        with CodeBlock("梯度裁剪"):
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:  # 梯度裁剪
                                if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                                    grad_norm = model.get_global_grad_norm()
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        with CodeBlock("参数更新"):
                            self.optimizer.step()
                            if not self.accelerator.optimizer_step_was_skipped:
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                        ############### state, control, callbacks ###############
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        # 1个step训练结束
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        ############### evaluate ###############
                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, ignore_keys_for_eval)
                    else:
                        # 梯度累计的1个substep训练结束
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # 跳出step循环
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                ############### state, control, callbacks ###############
                if step < 0:
                    self.control.should_training_stop = True
                # 1个epoch训练结束
                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

                ############### evaluate ###############
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, ignore_keys_for_eval)

                # 跳出epoch循环
                if self.control.should_training_stop:
                    break

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        train_loss = tr_loss.item() / max(self.state.global_step, 0.001)

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        return TrainOutput(self.state.global_step, train_loss, None)