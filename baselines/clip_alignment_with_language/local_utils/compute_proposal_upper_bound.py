"""
Compute oracle upper bound for a given proposal method, which acts like
a reversed recall, where we recall the GT timestamp pairs in the set of
generated proposals.
"""
import pprint
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils.basic_utils import load_jsonl, save_json
from standalone_eval.eval import compute_temporal_iou_batch
from baselines.clip_alignment_with_language.local_utils.proposal import get_proposal_interface, ProposalConfigs


def get_didemo_agreed_ts(times_list):
    """
    input example: [[1, 1], [1, 1], [1, 1], [0, 0]],
    return: [1, 1]"""
    times_str_list = [tuple(e) for e in times_list]
    times_str_list_counter = Counter(times_str_list)
    most_frequent_times = times_str_list_counter.most_common(1)[0][0]
    return most_frequent_times


def get_proposals_for_single_desc_video_pair(single_data, proposal_fn, dset_name):
    proposal_info = dict(
        vid_name=single_data["vid_name"],
        desc_id=single_data["desc_id"],
        gt_ts=single_data["ts"] if dset_name != "didemo" else get_didemo_agreed_ts(single_data["ts"]),
        proposals=proposal_fn(video_id="", metadata={"duration": single_data["duration"]}),
    )
    proposal_info["proposal_ious"] = compute_temporal_iou_batch(
        proposal_info["proposals"], proposal_info["gt_ts"])
    return proposal_info


def get_proposals_for_videos(datalist, dset_name):
    """datalist list(dict): each dict is
    {"desc_id": str/int, "duration": float, "ts": [st (float), ed (float)], ...}
    Note for Didemo dataset, "ts" entry is a list of [st (float), ed (float)] from different annotators,
    here we use the most frequent ts, we break ties by randomly sample one
    """
    proposal_interface = get_proposal_interface(dset_name)
    video_proposals_list = []
    for e in tqdm(datalist, desc="Computing video proposals"):
        video_proposals_list.append(
            get_proposals_for_single_desc_video_pair(e, proposal_interface, dset_name))
    return video_proposals_list


def is_recalled_single_moment(proposal_ious, iou_thds=(0.5, 0.7)):
    """
    Args:
        proposal_ious: np.ndarray, shape (N_proposal, )
        iou_thds: set, temporal IoU thresholds

    Returns:
        list(bool), len == len(iou_thds), indicates whether recall under a iou_thd is found.
    """
    recalled = [False, ] * len(iou_thds)
    for idx, iou_thd in enumerate(iou_thds):
        recalled[idx] = np.sum(proposal_ious >= iou_thd) >= 1  # at least one
    return recalled


def compute_proposal_recall_upper_bound(video_proposals_list, iou_thds=(0.5, 0.7)):
    """video_proposals_list from get_proposals_for_videos()"""
    iou_corrects = np.empty((len(video_proposals_list), 2), dtype=np.float32)
    for idx, d in tqdm(enumerate(video_proposals_list),
                       desc="Computing recall for videos",
                       total=len(video_proposals_list)):
        iou_corrects[idx] = is_recalled_single_moment(d["proposal_ious"],
                                                      iou_thds=iou_thds)
    recall_by_iou = {iou_thd: float(np.mean(iou_corrects[:, idx]))
                     for idx, iou_thd in enumerate(iou_thds)}
    return recall_by_iou


def main_compute_upper_bound():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dset_name", type=str, choices=["tvr"])
    parser.add_argument("-eval_file_path", type=str, help="path to the file containing data to be evaluated")
    parser.add_argument("-save_path", type=str, help="path to save the results")
    parser.add_argument("-verbose", action="store_true")
    args = parser.parse_args()

    eval_datalist = load_jsonl(args.eval_file_path)
    video_proposals_list = get_proposals_for_videos(eval_datalist, args.dset_name)
    recall_metrics = compute_proposal_recall_upper_bound(video_proposals_list, iou_thds=(0.5, 0.7))

    video_proposals_list_by_video = {}
    for p in video_proposals_list:
        if p["vid_name"] in video_proposals_list_by_video:
            continue
        else:
            video_proposals_list_by_video[p["vid_name"]] = p
    video_proposals_list_by_video = list(video_proposals_list_by_video.values())
    total_n_clips_in_proposals = \
        np.sum([np.sum(e["proposals"][:, 1] - e["proposals"][:, 0]) for e in video_proposals_list_by_video])

    results = dict(
        avg_num_proposals=float(np.mean([len(e["proposals"]) for e in video_proposals_list_by_video])),
        total_num_proposals=int(np.sum([len(e["proposals"]) for e in video_proposals_list_by_video])),
        recall_metrics=recall_metrics,
        dset_name=args.dset_name,
        filename=args.eval_file_path,
        proposal_config=ProposalConfigs[args.dset_name]
    )
    results["avg_clip_per_proposal"] = total_n_clips_in_proposals / results["total_num_proposals"]
    save_json(results, args.save_path, save_pretty=True)
    if args.verbose:
        pprint.pprint(results)


if __name__ == '__main__':
    main_compute_upper_bound()
