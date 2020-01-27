import os
import time
import json
import pprint
import random
import numpy as np
from collections import OrderedDict
from easydict import EasyDict as EDict
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baselines.mixture_embedding_experts.config import BaseOptions
from baselines.mixture_embedding_experts.model import MEE
from baselines.mixture_embedding_experts.retrieval_dataset import \
    RetrievalDataset, retrieval_collate, RetrievalEvalDataset, prepare_batch_inputs
from baselines.mixture_embedding_experts.inference import eval_epoch, start_inference
from utils.basic_utils import save_jsonl, save_json, AverageMeter
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


def train_epoch(model, train_loader, optimizer, opt, epoch_i):
    model.train()

    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meter = AverageMeter()

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        dataloading_time.update(time.time() - timer_dataloading)

        # continue
        timer_start = time.time()
        model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)
        timer_start = time.time()
        loss = model(**model_inputs)
        model_forward_time.update(time.time() - timer_start)
        timer_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        if opt.grad_clip != -1:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        model_backward_time.update(time.time() - timer_start)

        global_step = epoch_i * num_training_examples + batch_idx
        opt.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
        opt.writer.add_scalar("Train/Loss", float(loss), global_step)
        loss_meter.update(float(loss))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break
    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i,
        loss_str=str(loss_meter.avg))
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


def train(model, train_dataset, val_dataset, opt):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr)
    # reduce the lr by 0.1 every 30 epochs
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )

    train_loader = DataLoader(train_dataset,
                              collate_fn=retrieval_collate,
                              batch_size=opt.bsz,
                              num_workers=opt.num_workers,
                              shuffle=True,
                              pin_memory=opt.pin_memory)

    prev_best_score = 0.
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0
    eval_tasks_at_training = ["VR"]
    save_submission_filename = \
        "latest_{}_{}_predictions_{}.json".format(opt.dset_name, opt.eval_split_name, "_".join(eval_tasks_at_training))
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            with torch.autograd.detect_anomaly():
                train_epoch(model, train_loader, optimizer, opt, epoch_i)
        global_step = (epoch_i + 1) * len(train_loader)
        scheduler.step()
        if opt.eval_path is not None:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, tasks=eval_tasks_at_training)
            logger.info("metrics_no_nms {}".format(
                pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
            logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))

            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                eval_metrics_str=json.dumps(metrics_no_nms))
            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)

            # metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
            metrics = metrics_no_nms
            # early stop/ log / save model
            for task_type, task_metrics in metrics.items():
                for iou_thd in [0.5, 0.7]:
                    opt.writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd),
                                           {k: v for k, v in task_metrics.items() if str(iou_thd) in k},
                                           global_step)

            # use the most strict metric available
            if metrics["VR"]["r1"] > prev_best_score:
                es_cnt = 0
                prev_best_score = metrics["VR"]["r1"]

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
                if es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write("Early Stop at epoch {}".format(epoch_i))
                    logger.info("Early stop at {} with VR r1 {}".format(epoch_i, prev_best_score))
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


def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


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

    train_dataset = RetrievalDataset(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        vid_feat_path_or_handler=opt.vid_feat_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        ctx_mode=opt.ctx_mode,
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
    )

    if opt.eval_path is not None:
        eval_dataset = RetrievalEvalDataset(
            dset_name=opt.dset_name,
            eval_split_name=opt.eval_split_name,  # should only be val set
            data_path=opt.eval_path,
            desc_bert_path_or_handler=train_dataset.desc_bert_h5,
            sub_bert_path_or_handler=train_dataset.sub_bert_h5 if "sub" in opt.ctx_mode else None,
            max_desc_len=opt.max_desc_l,
            max_ctx_len=opt.max_ctx_l,
            video_duration_idx_path=opt.video_duration_idx_path,
            vid_feat_path_or_handler=train_dataset.vid_feat_h5 if "video" in opt.ctx_mode else None,
            ctx_mode=opt.ctx_mode,
            data_mode="query",
            h5driver=opt.h5driver,
            data_ratio=opt.data_ratio,
            normalize_vfeat=not opt.no_norm_vfeat,
            normalize_tfeat=not opt.no_norm_tfeat,
        )
    else:
        eval_dataset = None

    model_config = EDict(
        ctx_mode=opt.ctx_mode,
        text_input_size=opt.sub_feat_size,
        vid_input_size=opt.vid_feat_size,  #
        output_size=opt.output_size,
        margin=opt.margin,  # margin for ranking loss
    )
    logger.info("model_config {}".format(model_config))
    model = MEE(model_config)
    count_parameters(model)
    logger.info("Start Training...")
    train(model, train_dataset, eval_dataset, opt)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    model_dir, eval_split_name, eval_path, debug = start_training()
    if not debug:
        model_dir = model_dir.split(os.sep)[-1]
        tasks = ["VR"]
        input_args = ["--model_dir", model_dir,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path,
                      "--tasks"] + tasks

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model in {}".format(model_dir))
        start_inference()
