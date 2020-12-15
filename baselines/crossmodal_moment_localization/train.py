import os
import time
import json
import pprint
import random
import numpy as np
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baselines.crossmodal_moment_localization.config import BaseOptions
from baselines.crossmodal_moment_localization.model_xml import XML
from baselines.crossmodal_moment_localization.start_end_dataset import \
    StartEndDataset, start_end_collate, StartEndEvalDataset, prepare_batch_inputs
from baselines.crossmodal_moment_localization.inference import eval_epoch, start_inference
from baselines.crossmodal_moment_localization.optimization import BertAdam
from utils.basic_utils import AverageMeter
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True):
    logger.info("use train_epoch func for training: {}".format(training))
    model.train(mode=training)
    if opt.hard_negtiave_start_epoch != -1 and epoch_i >= opt.hard_negtiave_start_epoch:
        model.set_hard_negative(True, opt.hard_pool_size)
    if opt.train_span_start_epoch != -1 and epoch_i >= opt.train_span_start_epoch:
        model.set_train_st_ed(opt.lw_st_ed)

    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meters = OrderedDict(loss_st_ed=AverageMeter(),
                              loss_neg_ctx=AverageMeter(),
                              loss_neg_q=AverageMeter(),
                              loss_overall=AverageMeter())

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        global_step = epoch_i * num_training_examples + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)

        # continue
        timer_start = time.time()
        model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)
        # logger.info("model_inputs {}"
        #             .format({k: (type(k), v.shape if isinstance(v, torch.Tensor) else v)
        #                      for k, v in model_inputs.items()}))
        # logger.info("model_inputs \n{}".format({k: (type(v), v.shape, v.dtype) for k, v in model_inputs.items()}))
        timer_start = time.time()
        loss, loss_dict = model(**model_inputs)
        model_forward_time.update(time.time() - timer_start)
        timer_start = time.time()
        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)

            opt.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            for k, v in loss_dict.items():
                opt.writer.add_scalar("Train/{}".format(k), v, global_step)

        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    if training:
        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} "
              "min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n"
              "".format(dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
                        model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        for k, v in loss_meters.items():
            opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)


def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def train(model, train_dataset, train_eval_dataset, val_dataset, opt):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    train_loader = DataLoader(train_dataset,
                              collate_fn=start_end_collate,
                              batch_size=opt.bsz,
                              num_workers=opt.num_workers,
                              shuffle=True,
                              pin_memory=opt.pin_memory)

    train_eval_loader = DataLoader(train_eval_dataset,
                                   collate_fn=start_end_collate,
                                   batch_size=opt.bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=opt.pin_memory)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    num_train_optimization_steps = len(train_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         weight_decay=opt.wd,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")

    prev_best_score = 0.
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0
    eval_tasks_at_training = opt.eval_tasks_at_training  # VR is computed along with VCMR
    save_submission_filename = \
        "latest_{}_{}_predictions_{}.json".format(opt.dset_name, opt.eval_split_name, "_".join(eval_tasks_at_training))
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True)
        # TODO: continue from here.
        global_step = (epoch_i + 1) * len(train_loader)
        if opt.eval_path is not None:
            with torch.no_grad():
                train_epoch(model, train_eval_loader, optimizer, opt, epoch_i, training=False)

                metrics_no_nms, metrics_nms, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename,
                               tasks=eval_tasks_at_training, max_after_nms=100)
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                eval_metrics_str=json.dumps(metrics_no_nms))
            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
            logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms, indent=4)))

            # metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
            metrics = metrics_no_nms
            # early stop/ log / save model
            for task_type in ["SVMR", "VCMR"]:
                if task_type in metrics:
                    task_metrics = metrics[task_type]
                    for iou_thd in [0.5, 0.7]:
                        opt.writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd),
                                               {k: v for k, v in task_metrics.items() if str(iou_thd) in k},
                                               global_step)

            task_type = "VR"
            if task_type in metrics:
                task_metrics = metrics[task_type]
                opt.writer.add_scalars("Eval/{}".format(task_type),
                                       {k: v for k, v in task_metrics.items()},
                                       global_step)

            # use the most strict metric available
            stop_metric_names = ["r1"] if opt.stop_task == "VR" else ["0.5-r1", "0.7-r1"]
            stop_score = sum([metrics[opt.stop_task][e] for e in stop_metric_names])

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "model_cfg": model.config,
                    "epoch": epoch_i}
                torch.save(checkpoint, opt.ckpt_filepath)

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write("Early Stop at epoch {}".format(epoch_i))
                    logger.info("Early stop at {} with {} {}"
                                .format(epoch_i, " ".join([opt.stop_task] + stop_metric_names), prev_best_score))
                    break
        else:
            checkpoint = {
                "model": model.state_dict(),
                "model_cfg": model.config,
                "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

        if opt.debug:
            break

    opt.writer.close()


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    train_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        vid_feat_path_or_handler=opt.vid_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
    )

    if opt.eval_path is not None:
        # val dataset, used to get eval loss
        train_eval_dataset = StartEndDataset(
            dset_name=opt.dset_name,
            data_path=opt.eval_path,
            desc_bert_path_or_handler=train_dataset.desc_bert_h5,
            sub_bert_path_or_handler=train_dataset.sub_bert_h5 if "sub" in opt.ctx_mode else None,
            max_desc_len=opt.max_desc_l,
            max_ctx_len=opt.max_ctx_l,
            vid_feat_path_or_handler=train_dataset.vid_feat_h5 if "video" in opt.ctx_mode else None,
            clip_length=opt.clip_length,
            ctx_mode=opt.ctx_mode,
            h5driver=opt.h5driver,
            data_ratio=opt.data_ratio,
            normalize_vfeat=not opt.no_norm_vfeat,
            normalize_tfeat=not opt.no_norm_tfeat
        )

        eval_dataset = StartEndEvalDataset(
            dset_name=opt.dset_name,
            eval_split_name=opt.eval_split_name,  # should only be val set
            data_path=opt.eval_path,
            desc_bert_path_or_handler=train_dataset.desc_bert_h5,
            sub_bert_path_or_handler=train_dataset.sub_bert_h5 if "sub" in opt.ctx_mode else None,
            max_desc_len=opt.max_desc_l,
            max_ctx_len=opt.max_ctx_l,
            video_duration_idx_path=opt.video_duration_idx_path,
            vid_feat_path_or_handler=train_dataset.vid_feat_h5 if "video" in opt.ctx_mode else None,
            clip_length=opt.clip_length,
            ctx_mode=opt.ctx_mode,
            data_mode="query",
            h5driver=opt.h5driver,
            data_ratio=opt.data_ratio,
            normalize_vfeat=not opt.no_norm_vfeat,
            normalize_tfeat=not opt.no_norm_tfeat
        )
    else:
        eval_dataset = None

    model_config = EDict(
        merge_two_stream=not opt.no_merge_two_stream,  # merge video and subtitles
        cross_att=not opt.no_cross_att,  # use cross-attention when encoding video and subtitles
        span_predictor_type=opt.span_predictor_type,  # span_predictor_type
        encoder_type=opt.encoder_type,  # gru, lstm, transformer
        add_pe_rnn=opt.add_pe_rnn,  # add pe for RNNs
        pe_type=opt.pe_type,  #
        visual_input_size=opt.vid_feat_size,
        sub_input_size=opt.sub_feat_size,  # for both desc and subtitles
        query_input_size=opt.q_feat_size,  # for both desc and subtitles
        hidden_size=opt.hidden_size,  #
        stack_conv_predictor_conv_kernel_sizes=opt.stack_conv_predictor_conv_kernel_sizes,  #
        conv_kernel_size=opt.conv_kernel_size,
        conv_stride=opt.conv_stride,
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        input_drop=opt.input_drop,
        cross_att_drop=opt.cross_att_drop,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        ctx_mode=opt.ctx_mode,  # video, sub or video_sub
        margin=opt.margin,  # margin for ranking loss
        ranking_loss_type=opt.ranking_loss_type,  # loss type, 'hinge' or 'lse'
        lw_neg_q=opt.lw_neg_q,  # loss weight for neg. query and pos. context
        lw_neg_ctx=opt.lw_neg_ctx,  # loss weight for pos. query and neg. context
        lw_st_ed=0,  # will be assigned dynamically at training time
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size,
        use_self_attention=not opt.no_self_att,  # whether to use self attention
        no_modular=opt.no_modular
    )
    logger.info("model_config {}".format(model_config))
    model = XML(model_config)
    count_parameters(model)
    logger.info("Start Training...")
    train(model, train_dataset, train_eval_dataset, eval_dataset, opt)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    model_dir, eval_split_name, eval_path, debug = start_training()
    if not debug:
        model_dir = model_dir.split(os.sep)[-1]
        tasks = ["SVMR", "VCMR", "VR"]
        input_args = ["--model_dir", model_dir,
                      "--nms_thd", "0.5",
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path,
                      "--tasks"] + tasks

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model in {}".format(model_dir))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference()
