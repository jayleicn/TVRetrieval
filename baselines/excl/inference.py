import os
import copy
import math
import pprint
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from baselines.excl.config import TestOptions
from baselines.excl.model import EXCL
from baselines.excl.start_end_dataset import \
    start_end_collate, ExCLDataset, prepare_batch_inputs
from baselines.clip_alignment_with_language.inference import \
    get_submission_top_n, post_processing_vcmr_nms, post_processing_svmr_nms
from utils.basic_utils import save_json
from utils.tensor_utils import pad_sequences_1d, find_max_triples, find_max_triples_from_upper_triangle_product
from standalone_eval.eval import eval_retrieval

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def compute_query2ctx_info_svmr_only(model, eval_dataset, opt,
                                     max_before_nms=1000, max_n_videos=200, tasks=("SVMR",)):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 500 (hsz) * 4 / (1024**2) = 38.15 MB
    max_n_videos: int, use max_n_videos videos for computing VCMR results
    """
    model.eval()
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn=start_end_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=opt.pin_memory)
    video2idx = eval_dataset.video2idx
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz
    ctx_len = eval_dataset.max_ctx_len  # all pad to this length

    svmr_gt_st_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)
    svmr_gt_ed_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)

    query_metas = []
    for idx, batch in tqdm(
            enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        _query_metas = batch[0]
        query_metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        _, _, _st_probs, _ed_probs = model(**model_inputs)
        # normalize to get true probabilities!!!
        # the probabilities here are already (pad) masked, so only need to do softmax
        _st_probs = F.softmax(_st_probs, dim=-1)  # (_N_q, L)
        _ed_probs = F.softmax(_ed_probs, dim=-1)

        svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[1]] = _st_probs.cpu().numpy()
        svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[1]] = _ed_probs.cpu().numpy()

        if opt.debug:
            break
    svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs,
                                             query_metas, video2idx,
                                             clip_length=opt.clip_length,
                                             min_pred_l=opt.min_pred_l,
                                             max_pred_l=opt.max_pred_l,
                                             max_before_nms=max_before_nms)
    return dict(SVMR=svmr_res)


def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle with upper-right corner masked,
    below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]

    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span

    Returns:

    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32)  # (1, ..., 1, L, L)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1


def get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
                                  clip_length, min_pred_l, max_pred_l, max_before_nms):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query

    Returns:

    """
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in query_metas]

    # masking very long ones! Since most are relatively short.
    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)  # (N, L, L)
    # extra_length_mask_array = np.ones(st_ed_prob_product.shape, dtype=np.bool)  # (N, L, L)
    # mask_triu = np.triu(extra_length_mask_array, k=min_pred_l)
    # mask_triu_reversed = np.logical_not(np.triu(extra_length_mask_array, k=max_pred_l))
    # final_prob_mask = np.logical_and(mask_triu, mask_triu_reversed)  # with valid bit to be 1
    valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask  # invalid location will become zero!

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(
        st_ed_prob_product, top_n=max_before_nms, prob_thd=None)
    for i, q_vid_name in tqdm(enumerate(query_vid_names),
                              desc="[SVMR] Loop over queries to generate predictions",
                              total=len(query_vid_names)):  # i is query_id
        q_m = query_metas[i]
        video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        _sorted_triples[:, 1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [[video_idx, ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(
            desc_id=q_m["desc_id"],
            desc=q_m["desc"],
            predictions=cur_ranked_predictions
        )
        svmr_res.append(cur_query_pred)
    return svmr_res


def get_eval_res(model, eval_dataset, opt, tasks, max_after_nms):
    """compute and save query and video proposal embeddings"""
    eval_res = compute_query2ctx_info_svmr_only(model, eval_dataset, opt,
                                                max_before_nms=opt.max_before_nms,
                                                max_n_videos=max_after_nms,
                                                tasks=tasks)
    eval_res["video2idx"] = eval_dataset.video2idx
    return eval_res


POST_PROCESSING_MMS_FUNC = {
    "SVMR": post_processing_svmr_nms,
    "VCMR": post_processing_vcmr_nms
}


def eval_epoch(model, eval_dataset, opt, save_submission_filename,
               tasks=("SVMR",), max_after_nms=100):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")
    eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks, max_after_nms=max_after_nms)

    IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    metrics = eval_retrieval(eval_submission, eval_dataset.data,
                             iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug)
    save_metrics_path = submission_path.replace(".json", "_metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    latest_file_paths = [submission_path, save_metrics_path]

    if opt.nms_thd != -1:
        logger.info("Performing nms with nms_thd {}".format(opt.nms_thd))
        eval_submission_after_nms = dict(video2idx=eval_submission_raw["video2idx"])
        for k, nms_func in POST_PROCESSING_MMS_FUNC.items():
            if k in eval_submission_raw:
                eval_submission_after_nms[k] = nms_func(eval_submission_raw[k],
                                                        nms_thd=opt.nms_thd,
                                                        max_before_nms=opt.max_before_nms,
                                                        max_after_nms=max_after_nms)

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".json", "_nms_thd_{}.json".format(opt.nms_thd))
        save_json(eval_submission_after_nms, submission_nms_path)
        metrics_nms = eval_retrieval(eval_submission_after_nms, eval_dataset.data,
                                     iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug)
        save_metrics_nms_path = submission_nms_path.replace(".json", "_metrics.json")
        save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
        latest_file_paths += [submission_nms_path, save_metrics_nms_path]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    checkpoint = torch.load(opt.ckpt_filepath)
    model = EXCL(checkpoint["model_cfg"])
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}"
                .format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    assert opt.eval_path is not None
    eval_dataset = ExCLDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
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
        video_duration_idx_path=opt.video_duration_idx_path,
        eval_split_name=opt.eval_split_name
    )

    model = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_predictions_{}.json".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id, "_".join(opt.tasks))
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename,
                       tasks=opt.tasks, max_after_nms=100)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
