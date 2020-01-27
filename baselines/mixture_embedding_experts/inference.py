import os
import pprint
import time
from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from baselines.mixture_embedding_experts.config import TestOptions
from baselines.mixture_embedding_experts.model import MEE
from baselines.mixture_embedding_experts.retrieval_dataset import \
    retrieval_collate, RetrievalEvalDataset, prepare_batch_inputs
from utils.basic_utils import save_json
from standalone_eval.eval import eval_retrieval

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def compute_context_embeddings(model, eval_dataset, opt):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated 1000 (videos) * 300 (proposals) * 20 (clips) * 100 (hsz) * 4 / (1024 ** 3) = 2.24 GB
    """
    model.eval()
    eval_dataset.set_data_mode("context")
    context_eval_loader = DataLoader(eval_dataset,
                                     collate_fn=retrieval_collate,
                                     batch_size=opt.eval_ctx_bsz,
                                     num_workers=opt.num_workers,
                                     shuffle=False,
                                     pin_memory=opt.pin_memory)
    n_videos = len(eval_dataset)
    eval_ctx_bsz = opt.eval_ctx_bsz
    global_meta_list = []  # list(dicts)
    global_video_embedding, global_sub_embedding = None, None
    if model.use_video:
        global_video_embedding = torch.empty((n_videos, model.config.output_size),
                                             dtype=torch.float32, device=opt.device)  # (N_q, D_o)
    if model.use_sub:
        global_sub_embedding = torch.empty((n_videos, model.config.output_size),
                                           dtype=torch.float32, device=opt.device)  # (N_q, D_o)
    for idx, batch in tqdm(enumerate(context_eval_loader),
                           desc="Computing context embedding for videos",
                           total=len(context_eval_loader)):
        global_meta_list.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        encoded_video, encoded_sub = model.encode_context(model_inputs["video_feat"], model_inputs["sub_feat"])
        if model.use_video:
            global_video_embedding[idx * eval_ctx_bsz: (idx + 1) * eval_ctx_bsz] = encoded_video
        if model.use_sub:
            global_sub_embedding[idx * eval_ctx_bsz: (idx + 1) * eval_ctx_bsz] = encoded_sub

        if opt.debug and idx == 100:
            break
    return dict(video_meta=global_meta_list,
                encoded_video=global_video_embedding,
                encoded_sub=global_sub_embedding)


def compute_query2ctx_scores(model, eval_dataset, opt, max_n_videos=100):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 100 (hsz) * 4 / (1024**2) = 7.63 MB
    """
    ctx_info = compute_context_embeddings(model, eval_dataset, opt)

    model.eval()
    eval_dataset.set_data_mode("query")
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn=retrieval_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=opt.pin_memory)
    global_meta_list = []  # list(dicts)
    eval_query_bsz = opt.eval_query_bsz
    n_query = eval_query_bsz if opt.debug else len(eval_dataset)
    all_scores = torch.empty((n_query, max_n_videos), dtype=torch.float32)  # (N_q, max_n_videos)
    all_indices = torch.empty((n_query, max_n_videos), dtype=torch.long)  # (N_q, max_n_videos)
    for idx, batch in tqdm(enumerate(query_eval_loader),
                           desc="Computing q embedding",
                           total=len(query_eval_loader)):
        global_meta_list.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        pooled_query = model.query_pooling(model_inputs["query_feat"])  # (Nq, Dt)
        conf_matrix = model.get_score_from_pooled_query_with_encoded_ctx(
            pooled_query, ctx_info["encoded_video"], ctx_info["encoded_sub"])  # (Nq, Nc)
        sorted_values, sorted_indices = \
            torch.topk(conf_matrix, max_n_videos, dim=1, largest=True)  # (Nq, max_n_videos)
        all_scores[idx * eval_query_bsz: (idx + 1) * eval_query_bsz] = sorted_values.cpu()
        all_indices[idx * eval_query_bsz: (idx + 1) * eval_query_bsz] = sorted_indices.cpu()
        if opt.debug:
            break
    return dict(
        video_meta=ctx_info["video_meta"],
        query_meta=global_meta_list,
        q2ctx_scores=all_scores,
        q2ctx_indices=all_indices,
        video2idx=eval_dataset.video2idx
    )


def generate_vr_predictions_from_res(eval_res):
    video_meta = eval_res["video_meta"]  # list, (Nc, )
    query_meta = eval_res["query_meta"]  # list, (Nq, )
    video2idx = eval_res["video2idx"]
    q2ctx_scores = eval_res["q2ctx_scores"]  # (Nq, max_n_videos)
    q2ctx_indices = eval_res["q2ctx_indices"]  # (Nq, max_n_videos)

    vr_res = []
    for i, (scores_row, indices_row) in tqdm(enumerate(zip(q2ctx_scores, q2ctx_indices)),
                                             desc="[VR] Loop over queries to generate predictions",
                                             total=len(query_meta)):
        cur_vr_redictions = []
        for j, (v_score, v_meta_idx) in enumerate(zip(scores_row, indices_row)):
            video_idx = video2idx[video_meta[v_meta_idx]["vid_name"]]
            cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
        cur_query_pred = dict(
            desc_id=query_meta[i]["desc_id"],
            desc=query_meta[i]["desc"],
            predictions=cur_vr_redictions
        )
        vr_res.append(cur_query_pred)
    return vr_res


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


def eval_epoch(model, eval_dataset, opt, save_submission_filename,
               tasks=("SVMR",), max_before_nms=1000, max_after_nms=100):
    model.eval()
    logger.info("Computing scores")
    logger.info("Start timing")
    # times = []
    # for _ in range(3):
    #     st_time = time.time()
    eval_res = compute_query2ctx_scores(model, eval_dataset, opt)
    logger.info("Generating predictions from scores")
    eval_submission_raw = dict(video2idx=eval_res["video2idx"])
    eval_submission_raw["VR"] = generate_vr_predictions_from_res(eval_res)
        # times += [time.time() - st_time]
    # times = torch.FloatTensor(times)
    IOU_THDS = (0.5, 0.7)

    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=100)
    save_json(eval_submission, submission_path)

    metrics = eval_retrieval(eval_submission, eval_dataset.query_data,
                             iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug)
    # metrics["time_avg"] = float(times.mean())
    # metrics["time_std"] = float(times.std())
    save_metrics_path = submission_path.replace(".json", "_metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    latest_file_paths = [submission_path, save_metrics_path]

    metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    checkpoint = torch.load(opt.ckpt_filepath)
    model = MEE(checkpoint["model_cfg"])
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
    eval_dataset = RetrievalEvalDataset(
        dset_name=opt.dset_name,
        eval_split_name=opt.eval_split_name,  # should only be val set
        data_path=opt.eval_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_duration_idx_path=opt.video_duration_idx_path,
        vid_feat_path_or_handler=opt.vid_feat_path,
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
            eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=opt.tasks)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
