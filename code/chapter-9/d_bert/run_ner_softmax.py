import argparse
import glob
import logging
import os
import json
import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM

from tools.common import seed_everything
from tools.common import init_logger, logger

from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer
from models.bert_for_ner import BertSoftmaxForNer
from processors.utils_ner import get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSoftmaxForNer, BertTokenizer),
}


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # --------------------------- step1 : 构建dataloader --------------------------- #
    # 构建dataloader，注意，这里用了collate_fn，把dataset还做了进一步处理
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # --------------------------- step2 : 构建optimizer和lr_scheduler --------------------------- #
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    if args.do_adv:
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps == -1 and args.logging_steps == -1:
        args.logging_steps = len(train_dataloader)
        args.save_steps = len(train_dataloader)

    # --------------------------- step3 : 正式for循环训练 --------------------------- #
    step_count = 0
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            step_count += 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # 从这里看到了给模型的是3个输入，input_ids，attention_mask，labels
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.do_adv:
                fgm.attack()
                loss_adv = model(**inputs)[0]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        eval_result = evaluate(args, model, tokenizer)
                        logger.info(f"step:{step_count}\t"
                                    f"train_loss: {loss.item():.4f}\t"
                                    f"eval_loss: {eval_result['loss']:.4f}\t")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(preds[i][j])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:-1]  # [CLS]XXXX[SEP]
        tags = [args.id2label[x] for x in preds]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags)
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_submit_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split(os.path.sep))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def main():
    # --------------------------- step0 : 参数解析，日志初始化 --------------------------- #
    args = get_argparse().parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # debug模式配置
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # GPU 设置
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    # --------------------------- step1 : 任务数据初始化 --------------------------- #
    # 获取NER数据标签，数据处理类 processor
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    # --------------------------- step2 : 加载模型与tokenizer初始化 --------------------------- #
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.loss_type = args.loss_type
    # 加载tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, )
    # 加载模型
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # --------------------------- step3 : 模型训练 --------------------------- #
    if args.do_train:
        # Dataset构建
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        # 执行训练
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # --------------------------- step3 : 模型验证 --------------------------- #
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    # --------------------------- step3 : 模型预测 --------------------------- #
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()
