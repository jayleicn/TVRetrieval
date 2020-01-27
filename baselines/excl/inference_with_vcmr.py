import os
import copy
import math
import pprint
from tqdm import tqdm, trange
import numpy as np

import time
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from baselines.excl.config import TestOptions
from baselines.excl.model import EXCL
from baselines.excl.start_end_dataset import \
    start_end_collate, ExCLEvalDataset, prepare_batch_inputs
from baselines.clip_alignment_with_language.inference import \
    get_submission_top_n, post_processing_vcmr_nms, post_processing_svmr_nms
from utils.basic_utils import save_json, load_json, flat_list_of_lists
from utils.tensor_utils import pad_sequences_1d, find_max_triples, find_max_triples_from_upper_triangle_product
from standalone_eval.eval import eval_retrieval

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def load_external_vr_res_with_scores(external_vr_res_path, top_n_vr_videos=5):
    """return a mapping from desc_id to top retrieved (vid_name, score)"""
    external_vr_res = load_json(external_vr_res_path)
    external_vr_res = get_submission_top_n(external_vr_res, top_n=top_n_vr_videos)["VR"]
    query2video = {e["desc_id"]: [[sub_e[0], sub_e[3]] for sub_e in e["predictions"]] for e in external_vr_res}
    return query2video


def compute_query2ctx_info(model, eval_dataset, opt,
                           max_before_nms=1000, max_n_videos=200, tasks=("SVMR",)):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 500 (hsz) * 4 / (1024**2) = 38.15 MB
    max_n_videos: int, use max_n_videos videos for computing VCMR results
    """
    model.eval()
    eval_dataset.set_data_mode("query")

    logger.info("Using external VR results from {}".format(opt.external_inference_vr_res_path))
    external_query2video = load_external_vr_res_with_scores(
        opt.external_inference_vr_res_path, top_n_vr_videos=100)  # {desc_id: [(vid_name1, score1), ...]}
    video2idx = eval_dataset.video2idx
    idx2video = {v: k for k, v in video2idx.items()}
    vcmr_res = []
    for idx, single_query_data in tqdm(enumerate(eval_dataset), desc="query2ctx", total=len(eval_dataset)):
        single_query_meta = single_query_data["meta"]
        desc_id = single_query_meta["desc_id"]
        vid_names = [idx2video[e[0]] for e in external_query2video[desc_id]]
        bsz = len(vid_names)
        model_inputs = eval_dataset.get_batched_context(vid_names)[1]
        model_inputs["st_ed_indices"] = torch.zeros(bsz, 2).long()
        model_inputs["query_feat"] = (single_query_data["model_inputs"]["query_feat"].unsqueeze(0).repeat(bsz, 1, 1),
                                      torch.ones(bsz, len(single_query_data["model_inputs"]["query_feat"])))
        model_inputs = prepare_batch_inputs(model_inputs, device=opt.device, non_blocking=opt.pin_memory)
        _, _, _st_probs, _ed_probs = model(**model_inputs)

        # normalize to get true probabilities!!!
        # the probabilities here are already (pad) masked, so only need to do softmax
        _st_probs = F.softmax(_st_probs, dim=-1)  # (_N_q, L)
        _ed_probs = F.softmax(_ed_probs, dim=-1)

        vr_scores = _st_probs.new([e[1] for e in external_query2video[desc_id]]).unsqueeze(1)  # (N, 1)

        _st_probs = _st_probs * torch.exp(opt.q2c_alpha * vr_scores)

        st_ed_prob_product = torch.einsum("bm,bn->bmn", _st_probs, _ed_probs)  # (Nq, L, L)
        valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape,
                                                       min_l=opt.min_pred_l,
                                                       max_l=opt.max_pred_l)
        st_ed_prob_product *= st_ed_prob_product.new(valid_prob_mask)  # invalid location will become zero!

        st_ed_prob_product = st_ed_prob_product.cpu().numpy()
        batched_sorted_triples = find_max_triples_from_upper_triangle_product(
            st_ed_prob_product, top_n=50, prob_thd=None)
        # print("batched_sorted_triples", batched_sorted_triples[0][:4])
        # print("[12, ] + batched_sorted_triples[0][0]", [12, ] + batched_sorted_triples[0][0].tolist())
        # print("", batched_sorted_triples[0][0].tolist(), type(batched_sorted_triples[0][0].tolist()))
        batched_spans_with_names = []
        for vid_name, b in zip(vid_names, batched_sorted_triples):
            cur_video_idx = video2idx[vid_name]
            batched_spans_with_names += [[cur_video_idx] + e.tolist() for e in b]

        # print("batched_spans_with_names", len(batched_spans_with_names), batched_spans_with_names[0])
        cur_vcmr_redictions = sorted(batched_spans_with_names, key=lambda x: x[3], reverse=True)[:max_before_nms]
        cur_query_pred = dict(
            desc_id=single_query_meta["desc_id"],
            desc=single_query_meta["desc"],
            predictions=cur_vcmr_redictions)
        vcmr_res.append(cur_query_pred)

        if opt.debug and idx == 10:
            break
    return dict(VCMR=vcmr_res)


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


def get_eval_res(model, eval_dataset, opt, tasks, max_after_nms):
    """compute and save query and video proposal embeddings"""
    eval_res = compute_query2ctx_info(model, eval_dataset, opt,
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
    # logger.info("Start timing")
    # times = []
    # for _ in range(3):
    #     st_time = time.time()
    eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks, max_after_nms=max_after_nms)
        # times += [time.time() - st_time]
    # times = torch.FloatTensor(times)

    IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    metrics = eval_retrieval(eval_submission, eval_dataset.query_data,
                             iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug)
    # metrics["time_avg"] = float(times.mean())
    # metrics["time_std"] = float(times.std())
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
        metrics_nms = eval_retrieval(eval_submission_after_nms, eval_dataset.query_data,
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
    assert opt.external_inference_vr_res_path is not None

    assert opt.eval_path is not None
    eval_dataset = ExCLEvalDataset(
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
