import os
import time
import math
import pprint
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict, OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from baselines.clip_alignment_with_language.config import TestOptions
from baselines.clip_alignment_with_language.model import CALWithSub
from baselines.clip_alignment_with_language.proposal_retrieval_dataset import \
    proposal_retrieval_collate, ProposalRetrievalEvalDataset, prepare_batch_inputs
from utils.basic_utils import save_jsonl, save_json, load_json
from utils.temporal_nms import temporal_non_maximum_suppression
from utils.tensor_utils import pad_sequences_1d
from standalone_eval.eval import eval_retrieval

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def combine_single_video_proposal_embeddings(proposals_embedding_list, proposals_mask_list):
    """
    Args:
        proposals_embedding_list: list(torch.Tensor), bsz * (N_prop, N_clips, D_o)
        proposals_mask_list: list(torch.Tensor), bsz * (N_prop, N_clips)
    """
    if len(proposals_embedding_list) == 1:
        return proposals_embedding_list[0], proposals_mask_list[0]
    else:  # > 1
        max_n_clips = max([e.shape[1] for e in proposals_embedding_list])
        n_proposals = sum([len(e) for e in proposals_embedding_list])
        d = proposals_embedding_list[0].shape[2]
        proposals_embedding = proposals_embedding_list[0].new_zeros((n_proposals, max_n_clips, d))
        proposals_mask = proposals_mask_list[0].new_zeros((n_proposals, max_n_clips))
        mask_lengths = [0, ] + [len(m) for m in proposals_mask_list]
        mask_cumsum_lengths = np.cumsum(mask_lengths)
        for idx, (e, m) in enumerate(zip(proposals_embedding_list, proposals_mask_list)):
            proposals_embedding[mask_cumsum_lengths[idx]:mask_cumsum_lengths[idx + 1], :e.shape[1]] = e
            proposals_mask[mask_cumsum_lengths[idx]:mask_cumsum_lengths[idx + 1], :m.shape[1]] = m
        return proposals_embedding, proposals_mask


def compute_query_embeddings(model, eval_dataset, opt, load_gt_vid_name):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 100 (hsz) * 4 / (1024**2) = 7.63 MB
    """
    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(load_gt_vid_name)
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn=proposal_retrieval_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=opt.pin_memory)
    global_meta_list = []  # list(dicts)
    # n_query = min(len(eval_dataset), opt.eval_query_bsz) if opt.debug else len(eval_dataset)
    n_query = len(eval_dataset)
    global_query_embedding = torch.empty((n_query,
                                          model.config.output_size),
                                         dtype=torch.float32, device=opt.device)  # (N_q, D_o)
    for idx, batch in tqdm(enumerate(query_eval_loader),
                           desc="Computing q embedding",
                           total=len(query_eval_loader)):
        global_meta_list.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        global_query_embedding[idx * opt.eval_query_bsz: (idx + 1) * opt.eval_query_bsz] = \
            model.query_encoder(**model_inputs)

        if opt.debug:
            break
    return global_meta_list, global_query_embedding


def compute_proposal_embeddings(model, eval_dataset, opt):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated 1000 (videos) * 300 (proposals) * 20 (clips) * 100 (hsz) * 4 / (1024 ** 3) = 2.24 GB
    """
    model.eval()
    eval_dataset.set_data_mode("context")
    global_meta_list = []  # list(dicts)
    global_proposal_video_embedding_list = []  # list(torch.tensor), N_videos * [N_prop, N_clips, D_o]
    global_proposal_sub_embedding_list = []  # list(torch.tensor), N_videos * [N_prop, N_clips, D_o]
    global_proposal_video_mask_list = []  # list(torch.tensor), N_videos * [N_prop, N_clips]
    global_proposal_sub_mask_list = []  # list(torch.tensor), N_videos * [N_prop, N_clips]
    for idx, single_video_info in tqdm(enumerate(eval_dataset),
                                       desc="Computing prop embedding for videos",
                                       total=len(eval_dataset)):
        global_meta_list.append(single_video_info["meta"])
        if model.use_video or model.tef_only:
            proposals_features_list = single_video_info["model_inputs"]["video_moment_features_list"]
            proposals_mask_list = single_video_info["model_inputs"]["video_moment_mask_list"]
            proposals_mask_list = [e.to(opt.device, non_blocking=opt.pin_memory) for e in proposals_mask_list]
            proposals_embedding_list = []  # (N_prop, D_o)
            for feat in proposals_features_list:
                proposals_embedding_list.append(
                    model.moment_encoder(feat.to(opt.device, non_blocking=opt.pin_memory), module_name="video"))
            p, m = combine_single_video_proposal_embeddings(proposals_embedding_list, proposals_mask_list)
            global_proposal_video_embedding_list.append(p)
            global_proposal_video_mask_list.append(m)
        else:
            global_proposal_video_embedding_list.append(None)

        if model.use_sub:
            proposals_features_list = single_video_info["model_inputs"]["sub_moment_features_list"]
            proposals_mask_list = single_video_info["model_inputs"]["sub_moment_mask_list"]
            proposals_mask_list = [e.to(opt.device, non_blocking=opt.pin_memory) for e in proposals_mask_list]
            proposals_embedding_list = []  # (N_prop, D_o)
            for feat in proposals_features_list:
                proposals_embedding_list.append(
                    model.moment_encoder(feat.to(opt.device, non_blocking=opt.pin_memory), module_name="sub"))
            p, m = combine_single_video_proposal_embeddings(proposals_embedding_list, proposals_mask_list)
            global_proposal_sub_embedding_list.append(p)
            global_proposal_sub_mask_list.append(m)
        else:
            global_proposal_sub_embedding_list.append(None)

        if opt.debug and idx == 100:
            break
    global_proposal_mask_list = global_proposal_sub_mask_list if model.use_sub else global_proposal_video_mask_list
    return global_meta_list, global_proposal_video_embedding_list, \
           global_proposal_sub_embedding_list, global_proposal_mask_list


def compute_query_proposal_distance(model, eval_dataset, opt, tasks=("SVMR",)):
    """compute and save query and video proposal embeddings,
    tasks: SVMR (single video moment retrieval), VCMR (video corpus moment retrieval)
    """
    is_svmr = "SVMR" in tasks
    is_vcmr = "VCMR" in tasks
    query_meta_list, query_embed = compute_query_embeddings(model, eval_dataset, opt,
                                                            load_gt_vid_name=is_svmr)
    video_meta_list, video_prop_embed_list, sub_prop_embed_list, prop_mask_list = \
        compute_proposal_embeddings(model, eval_dataset, opt)

    eval_res = dict(
        query_meta=query_meta_list,  # N_q * dict()
        video_meta=video_meta_list,  # N_videos * dict()
        video2idx=eval_dataset.video2idx,  # dict {vid_name: index}
        query_prop_dist_vcmr=[],  # N_videos * (N_q, N_prop), note N_prop is changing for each video.
        query_prop_dist_svmr=[],  # N_q * (N_prop, ), each query has a GT video, no need to calc. for all.
    )
    if is_vcmr:
        for v_prop_embed, s_prop_embed, prop_mask in tqdm(
                zip(video_prop_embed_list, sub_prop_embed_list, prop_mask_list),
                desc="Computing VCMR q to prop dist for videos",
                total=len(video_prop_embed_list)):
            query_prop_dist = model.compute_cdist_inference(
                query_embed, v_prop_embed, s_prop_embed, prop_mask)  # (N_q, N_prop)
            eval_res["query_prop_dist_vcmr"].append(query_prop_dist.cpu())
            if opt.debug:
                break

    if is_svmr:
        if opt.debug:
            debug_query_meta = []
        # this is different from video2idx
        svmr_video2meta_idx = {e["vid_name"]: idx for idx, e in enumerate(video_meta_list)}
        # logger.info("svmr_video2idx {}".format(list(svmr_video2idx.keys())[:3]))
        for single_q_embed, single_q_meta in tqdm(zip(query_embed, query_meta_list),
                                                  desc="Computing SVMR q to prop dist for videos",
                                                  total=len(query_embed)):
            # logger.info("single_q_meta[vid_name] {}".format(single_q_meta["vid_name"]))
            if opt.debug:
                if single_q_meta["vid_name"] not in svmr_video2meta_idx:
                    continue
                debug_query_meta.append(single_q_meta)
            q_gt_vid_meta_idx = svmr_video2meta_idx[single_q_meta["vid_name"]]
            v_prop_embed = video_prop_embed_list[q_gt_vid_meta_idx]  # [N_prop, N_clips, D_o]
            s_prop_embed = sub_prop_embed_list[q_gt_vid_meta_idx]  # [N_prop, N_clips, D_o]
            prop_mask = prop_mask_list[q_gt_vid_meta_idx]  # [N_prop, N_clips]
            query_prop_dist = model.compute_cdist_inference(
                single_q_embed.unsqueeze(0), v_prop_embed, s_prop_embed, prop_mask)  # (1, N_prop)
            eval_res["query_prop_dist_svmr"].append(query_prop_dist.squeeze(0).cpu().numpy())
        if opt.debug:
            eval_res["query_meta"] = debug_query_meta
    return eval_res


def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6,
                       max_before_nms=1000, max_after_nms=100, score_col_idx=3):
    """ Apply non-maximum suppression for all the predictions for each video.
    1) group predictions by video index
    2) apply nms individually for each video index group
    3) combine and sort the predictions
    Args:
        all_video_predictions: list(sublist),
            Each sublist is [video_idx (int), st (float), ed(float), score (float)]
            Note the scores are negative distances.
        nms_threshold: float
        max_before_nms: int
        max_after_nms: int
        score_col_idx: int
    Returns:

    """
    predictions_neg_by_video_group = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_neg_by_video_group[pred[0]].append(pred[1:])  # [st (float), ed(float), score (float)]

    predictions_by_video_group_neg_after_nms = dict()
    for video_idx, grouped_preds in predictions_neg_by_video_group.items():
        predictions_by_video_group_neg_after_nms[video_idx] = \
            temporal_non_maximum_suppression(grouped_preds, nms_threshold=nms_threshold)

    predictions_after_nms = []
    for video_idx, grouped_preds in predictions_by_video_group_neg_after_nms.items():
        for pred in grouped_preds:
            pred = [video_idx] + pred  # [video_idx (int), st (float), ed(float), score (float)]
            predictions_after_nms.append(pred)

    # ranking happens across videos
    predictions_after_nms = sorted(predictions_after_nms,
                                   key=lambda x: x[score_col_idx],
                                   reverse=True)[:max_after_nms]  # descending order
    return predictions_after_nms


def post_processing_vcmr_nms(vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    vcmr_res: list(dict), each dict is{
        "desc": str,
        "desc_id": int,
        "predictions": list(sublist)  # each sublist is
            [video_idx (int), st (float), ed(float), score (float)], video_idx could be different
    }
    """
    processed_vcmr_res = []
    for e in vcmr_res:
        e["predictions"] = filter_vcmr_by_nms(e["predictions"],
                                              nms_threshold=nms_thd,
                                              max_before_nms=max_before_nms,
                                              max_after_nms=max_after_nms)
        processed_vcmr_res.append(e)
    return processed_vcmr_res


def post_processing_svmr_nms(svmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    svmr_res: list(dict), each dict is
        {"desc": str,
         "desc_id": int,
         "predictions": list(sublist)  # each sublist is
            [video_idx (int), st (float), ed(float), score (float)], video_idx is the same.
         }
    """
    processed_svmr_res = []
    for e in svmr_res:
        # the predictions are sorted inside the nms func.
        e["predictions"] = temporal_non_maximum_suppression(
            e["predictions"][:max_before_nms], nms_threshold=nms_thd)[:max_after_nms]
        processed_svmr_res.append(e)
    return processed_svmr_res


def generate_vcmr_predictions_from_res_with_external(eval_res, max_prop_per_query=300, query_bsz_in_sort=1000):
    """ This function is for Video Corpus Moment Retrieval (VCMR).
    Generate prediction file which could be evaluated using standalone_eval.eval.
    Args:
        eval_res: dict(
            query_meta=query_meta_list,  # N_q * dict(), each dict is {"desc_id": int, "desc": str}
            video_meta=video_meta_list,  # N_videos * dict(), {"vid_name": str, "duration": float, "proposals": ndarray}
            video2idx=eval_dataset.video2idx,  # dict {vid_name: index}
            video_bsz_in_sort=[],  # N_videos * (N_q, N_prop)
        )
        max_prop_per_query: int or None. If None, generate ranking for all possible moments, else generate top {}.
        query_bsz_in_sort: int, only sort a subset of queries at a time, it will be too large to sort all queries.
    return:
        list(dicts): each dict is dict(desc=str, desc_id=int, predictions=list(sublist)),
            each sublist is [vid_name (str), st (float), ed (float), score (float)], score is negative distance.
    """
    # video2idx
    video2idx = eval_res["video2idx"]
    video_meta = eval_res["video_meta"]
    query_meta = eval_res["query_meta"]
    video_idx2meta_idx = {video2idx[m["vid_name"]]: i for i, m in enumerate(video_meta)}
    external_query2video = eval_res["external_query2video"] if "external_query2video" in eval_res else None
    # 「query idx： [video meta idx]」
    external_query2video_meta_idx = {k: [video_idx2meta_idx[e] for e in v] for k, v in external_query2video.items()}

    external_ordered_video_meta_indices = torch.LongTensor(
        [external_query2video_meta_idx[e["desc_id"]] for e in query_meta])  # (Nq, 5)
    top_n_retrieved = external_ordered_video_meta_indices.shape[1]

    # (N_videos, N_prop, N_q), (N_videos, N_prop)
    padded_dist, padded_mask = pad_sequences_1d([e.transpose(0, 1) for e in eval_res["query_prop_dist_vcmr"]],
                                                dtype=eval_res["query_prop_dist_vcmr"][0].dtype,
                                                device=eval_res["query_prop_dist_vcmr"][0].device)
    # putting 'NaN' into the invalid bits, torch.sort considers 'NaN' as larger than any number!!!
    padded_dist += (padded_mask.unsqueeze(2) == 0).float() * 1e10
    n_videos, n_prop, n_q = padded_dist.shape
    padded_dist = padded_dist.permute(2, 0, 1)  # (N_q, N_videos, N_prop)

    # get only top retrieved, N_videos now decreased to top_n_retrieved
    row_indices = torch.arange(n_q, device=padded_dist.device)
    padded_dist = torch.stack([
        padded_dist[row_indices, external_ordered_video_meta_indices[:, col_idx]]
        for col_idx in range(top_n_retrieved)], dim=1)  # (N_q, 5, N_prop)
    n_videos = top_n_retrieved

    padded_dist = padded_dist.view(n_q, -1).contiguous()  # (N_q, N_video*N_prop)
    print("n_videos, n_prop, n_q {}".format((n_videos, n_prop, n_q)))
    print("padded_dist, {}".format(padded_dist.shape))

    sorted_distances, sorted_indices = torch.topk(padded_dist.to(torch.device("cuda:0"), non_blocking=True),
                                                  k=min(max_prop_per_query, n_videos * n_prop),
                                                  dim=1, largest=False, sorted=True)  # (N_q, max_prop_per_query) * 2
    print("orted_distances {}, sorted_indices {}".format(sorted_distances.shape, sorted_indices.shape))
    sorted_distances = - sorted_distances.cpu().numpy()

    # (N_q, max_prop_per_query) * 2, prop_indices: inside video indices.
    video_meta_indices_retrieved = torch.floor(sorted_indices.float() / n_prop).long().cpu().numpy()
    # map back to original video idx (not video meta idx, but real video idx)
    video_indices = np.array([[external_query2video[query_meta[i]["desc_id"]][j] for j in r]
                              for i, r in enumerate(video_meta_indices_retrieved)])  # (N_q, max_prop_per_query)
    prop_indices = torch.remainder(sorted_indices, n_prop).cpu().numpy()  # (N_q, max_prop_per_query)
    print("video_indices {}, prop_indices {}".format(video_indices.shape, prop_indices.shape))

    vr_res = []
    for i in trange(n_q, desc="[VR] Loop over queries to generate predictions"):
        row = video_indices[i]
        score_row = - sorted_distances[i]
        cur_vr_redictions = []
        for j, video_idx in enumerate(row):
            cur_vr_redictions.append([int(video_idx), 0, 0, float(score_row[j])])
        cur_query_pred = dict(
            desc_id=query_meta[i]["desc_id"],
            desc=query_meta[i]["desc"],
            predictions=cur_vr_redictions
        )
        vr_res.append(cur_query_pred)

    vcmr_res = []
    logger.debug("sorted_indices {}".format(sorted_indices.shape))
    logger.debug("sorted_distances {}".format(sorted_distances.shape))
    out_bounds_cnt = 0
    for idx, (v_row_indices, p_row_indices) in tqdm(enumerate(zip(video_indices, prop_indices)),
                                                    desc="[VCMR] Loop over queries to generate predictions",
                                                    total=n_q):  # query
        sorted_distances_row = - sorted_distances[idx]  # converted to negative distance
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = []
        for col_idx, (v_col_idx, p_col_idx) in enumerate(zip(v_row_indices, p_row_indices)):
            cur_proposals = eval_res["video_meta"][video_idx2meta_idx[v_col_idx]]["proposals"]
            cur_pred = []
            cur_pred += [int(v_col_idx), ]
            # what is wrong with the indexing below??? (out of bounds), but results seems fine???
            # Not a bug. Since there might be less than max_before_nms proposals from the top retrieved videos
            if p_col_idx >= len(cur_proposals):
                out_bounds_cnt += 1
                p_col_idx = len(cur_proposals)-1
            cur_pred += cur_proposals[p_col_idx].tolist()
            cur_pred += [float(sorted_distances_row[col_idx])]
            cur_ranked_predictions.append(cur_pred)
        cur_query_pred = dict(
            desc_id=eval_res["query_meta"][idx]["desc_id"],
            desc=eval_res["query_meta"][idx]["desc"],
            predictions=cur_ranked_predictions
        )
        vcmr_res.append(cur_query_pred)
    logger.info("[DEBUG] out_bounds_cnt {}".format(out_bounds_cnt))
    return vcmr_res, vr_res


def generate_vcmr_predictions_from_res(eval_res, max_prop_per_query=300, query_bsz_in_sort=1000):
    """ This function is for Video Corpus Moment Retrieval (VCMR).
    Generate prediction file which could be evaluated using standalone_eval.eval.
    Args:
        eval_res: dict(
            query_meta=query_meta_list,  # N_q * dict(), each dict is {"desc_id": int, "desc": str}
            video_meta=video_meta_list,  # N_videos * dict(), {"vid_name": str, "duration": float, "proposals": ndarray}
            video2idx=eval_dataset.video2idx,  # dict {vid_name: index}
            video_bsz_in_sort=[],  # N_videos * (N_q, N_prop)
        )
        max_prop_per_query: int or None. If None, generate ranking for all possible moments, else generate top {}.
        query_bsz_in_sort: int, only sort a subset of queries at a time, it will be too large to sort all queries.
    return:
        list(dicts): each dict is dict(desc=str, desc_id=int, predictions=list(sublist)),
            each sublist is [vid_name (str), st (float), ed (float), score (float)], score is negative distance.
    """
    # video2idx
    video2idx = eval_res["video2idx"]

    # (N_videos, N_prop, N_q), (N_videos, N_prop)
    padded_dist, padded_mask = pad_sequences_1d([e.transpose(0, 1) for e in eval_res["query_prop_dist_vcmr"]],
                                                dtype=eval_res["query_prop_dist_vcmr"][0].dtype,
                                                device=eval_res["query_prop_dist_vcmr"][0].device)
    # putting 'NaN' into the invalid bits, torch.sort considers 'NaN' as larger than any number!!!
    padded_dist += (padded_mask.unsqueeze(2) == 0).float() * 1e10
    n_videos, n_prop, n_q = padded_dist.shape
    print("n_videos, n_prop, n_q {}".format((n_videos, n_prop, n_q)))
    padded_dist = padded_dist.view(n_videos * n_prop, n_q).transpose(0, 1).contiguous()  # (N_q, N_video*N_prop)
    print("padded_dist, {}".format(padded_dist.shape))

    sorted_distances, sorted_indices = torch.topk(padded_dist.to(torch.device("cuda:0"), non_blocking=True),
                                                  k=min(max_prop_per_query, n_videos * n_prop),
                                                  dim=1, largest=False, sorted=True)  # (N_q, max_prop_per_query) * 2
    sorted_distances = - sorted_distances.cpu().numpy()

    # (N_q, max_prop_per_query) * 2, prop_indices: inside video indices.
    video_meta_indices = torch.floor(sorted_indices.float() / n_prop).long().cpu().numpy()
    prop_indices = torch.remainder(sorted_indices, n_prop).cpu().numpy()

    vr_res = []
    query_meta = eval_res["query_meta"]
    for i in trange(n_q, desc="[VR] Loop over queries to generate predictions"):
        row = video_meta_indices[i]
        score_row = - sorted_distances[i]
        cur_vr_redictions = []
        for j, meta_idx in enumerate(row):
            video_idx = video2idx[eval_res["video_meta"][meta_idx]["vid_name"]]
            cur_vr_redictions.append([video_idx, 0, 0, float(score_row[j])])
        cur_query_pred = dict(
            desc_id=query_meta[i]["desc_id"],
            desc=query_meta[i]["desc"],
            predictions=cur_vr_redictions
        )
        vr_res.append(cur_query_pred)

    vcmr_res = []
    logger.debug("sorted_indices {}".format(sorted_indices.shape))
    logger.debug("sorted_distances {}".format(sorted_distances.shape))
    for idx, (vm_row_indices, p_row_indices) in tqdm(enumerate(zip(video_meta_indices, prop_indices)),
                                                     desc="[VCMR] Loop over queries to generate predictions",
                                                     total=n_q):  # query
        sorted_distances_row = - sorted_distances[idx]  # converted to negative distance
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = []
        for col_idx, (v_col_idx, p_col_idx) in enumerate(zip(vm_row_indices, p_row_indices)):
            cur_pred = []
            cur_pred += [video2idx[eval_res["video_meta"][v_col_idx]["vid_name"]], ]
            cur_pred += eval_res["video_meta"][v_col_idx]["proposals"][p_col_idx].tolist()
            cur_pred += [float(sorted_distances_row[col_idx])]
            cur_ranked_predictions.append(cur_pred)
        cur_query_pred = dict(
            desc_id=eval_res["query_meta"][idx]["desc_id"],
            desc=eval_res["query_meta"][idx]["desc"],
            predictions=cur_ranked_predictions
        )
        vcmr_res.append(cur_query_pred)
    return vcmr_res, vr_res


def generate_svmr_predictions_from_res(eval_res, max_prop_per_query=None):
    """ This function is for Video Corpus Moment Retrieval (VCMR).
    Generate prediction file which could be evaluated using standalone_eval.eval.
    Args:
        eval_res: dict(
            query_meta=query_meta_list,  # N_q * dict(), each dict is {"desc_id": int, "desc": str}
            video_meta=video_meta_list,  # N_videos * dict(), {"vid_name": str, "duration": float, "proposals": ndarray}
            video2idx=eval_dataset.video2idx,  # dict {vid_name: index}
            query_prop_dist_svmr=[],  # N_q * (N_prop, )
        )
        max_prop_per_query: not used
    return:
        list(dicts): each dict is dict(desc=str, desc_id=int, predictions=list(sublist)),
            each sublist is [vid_name (str), st (float), ed (float), score (float)], score is negative distance.
    """
    video2idx = eval_res["video2idx"]

    svmr_res = []
    svmr_video2meta_idx = {e["vid_name"]: idx for idx, e in enumerate(eval_res["video_meta"])}
    for idx, (q_p_dist, q_m) in tqdm(enumerate(zip(eval_res["query_prop_dist_svmr"], eval_res["query_meta"])),
                                     desc="Loop over queries to generate predictions",
                                     total=len(eval_res["query_prop_dist_svmr"])):  # query
        sorted_indices = np.argsort(q_p_dist)  # (N_prop, )  # ascending order, distance
        if max_prop_per_query is not None:
            sorted_indices = sorted_indices[:max_prop_per_query]
        v_eval_idx = video2idx[q_m["vid_name"]]
        v_meta_idx = svmr_video2meta_idx[q_m["vid_name"]]
        proposals = eval_res["video_meta"][v_meta_idx]["proposals"]  # (N_p, 2)
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [
            [v_eval_idx, ] + proposals[sort_idx].tolist() + [- round(float(q_p_dist[sort_idx]), 4), ]
            for sort_idx in sorted_indices]
        cur_query_pred = dict(
            desc_id=q_m["desc_id"],
            desc=q_m["desc"],
            predictions=cur_ranked_predictions
        )
        svmr_res.append(cur_query_pred)
    return svmr_res


POST_PROCESSING_MMS_FUNC = {
    "SVMR": post_processing_svmr_nms,
    "VCMR": post_processing_vcmr_nms
}


def get_submission_top_n(submission, top_n=100):
    def get_prediction_top_n(list_dict_predictions, top_n):
        top_n_res = []
        for e in list_dict_predictions:
            e["predictions"] = e["predictions"][:top_n]
            top_n_res.append(e)
        return top_n_res

    top_n_submission = dict(video2idx=submission["video2idx"], )
    for k in submission:
        if k != "video2idx":
            top_n_submission[k] = get_prediction_top_n(submission[k], top_n)
    return top_n_submission


def load_external_vr_res(external_vr_res_path, top_n_vr_videos=5):
    """return a mapping from desc_id to top retrieved video id"""
    external_vr_res = load_json(external_vr_res_path)
    external_vr_res = get_submission_top_n(external_vr_res, top_n=top_n_vr_videos)["VR"]
    query2video = {e["desc_id"]: [sub_e[0] for sub_e in e["predictions"]] for e in external_vr_res}
    return query2video


def eval_epoch(model, eval_dataset, opt, save_submission_filename,
               tasks=("SVMR",), max_before_nms=1000, max_after_nms=100):
    model.eval()
    logger.info("Computing scores")
    logger.info("Start timing")
    # times = []  # do not use
    # for _ in range(3):
    #     st_time = time.time()
    if opt.use_intermediate:
        intermediate_cache_path = os.path.join(opt.results_dir, "{}_eval_res.pt".format(opt.eval_split_name))
        if not os.path.exists(intermediate_cache_path):
            logger.info("Saving intermediate results {}.".format(intermediate_cache_path))
            eval_res = compute_query_proposal_distance(model, eval_dataset, opt, tasks=tasks)
            torch.save(eval_res, intermediate_cache_path)
        else:
            logger.info("Loading intermediate results {}.".format(intermediate_cache_path))
            eval_res = torch.load(intermediate_cache_path)
    else:
        logger.info("Running without saving intermediate results, you might want to turn on --use_intermediate.")
        eval_res = compute_query_proposal_distance(model, eval_dataset, opt, tasks=tasks)
    # del model  # We dont need model anymore

    # eval_res = compute_query_proposal_distance(model, eval_dataset, opt, tasks=tasks)

    logger.info("Generating predictions from scores")
    eval_submission_raw = dict(video2idx=eval_res["video2idx"])
    if "SVMR" in tasks:
        eval_submission_raw["SVMR"] = generate_svmr_predictions_from_res(
            eval_res, max_prop_per_query=max_before_nms)
    # vcmr_loading_time = 0
    if "VCMR" in tasks:
        if opt.external_inference_vr_res_path is not None:
            logger.info("Using external VR results from {}".format(opt.external_inference_vr_res_path))
            # vcmr_loading_time = time.time()
            eval_res["external_query2video"] = load_external_vr_res(
                opt.external_inference_vr_res_path, top_n_vr_videos=5)
            # vcmr_loading_time = time.time() - vcmr_loading_time
            vcmr_res, vr_res = generate_vcmr_predictions_from_res_with_external(
                eval_res, max_prop_per_query=max_before_nms)
        else:
            vcmr_res, vr_res = generate_vcmr_predictions_from_res(
                eval_res, max_prop_per_query=max_before_nms)
        eval_submission_raw["VCMR"] = vcmr_res
        eval_submission_raw["VR"] = vr_res
        # times += [time.time() - st_time - vcmr_loading_time]
    # times = torch.FloatTensor(times)
    IOU_THDS = (0.5, 0.7)

    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    if max_after_nms < 1000:
        save_json(eval_submission, submission_path)
    else:
        torch.save(eval_submission, submission_path.replace(".json", ".pt"))

    metrics = eval_retrieval(eval_submission, eval_dataset.query_data,
                             iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug,
                             use_desc_type=opt.dset_name == "tvr")
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
                                                        max_before_nms=max_before_nms,
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
    model = CALWithSub(checkpoint["model_cfg"])
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
    eval_dataset = ProposalRetrievalEvalDataset(
        dset_name=opt.dset_name,
        model_type=opt.model_type,
        eval_split_name=opt.eval_split_name,  # should only be val set
        data_path=opt.eval_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        video_duration_idx_path=opt.video_duration_idx_path,
        vid_feat_path_or_handler=opt.vid_feat_path,
        clip_length=opt.clip_length,
        eval_proposal_bsz=opt.eval_proposal_bsz,
        ctx_mode=opt.ctx_mode,
        data_mode="query",
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
    )

    model = setup_model(opt)
    save_submission_filename = \
        "inference_{}_{}_{}_predictions_{}.json".format(
            opt.dset_name, opt.eval_split_name, opt.eval_id, "_".join(opt.tasks))
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=opt.tasks,
                       max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
