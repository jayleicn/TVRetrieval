import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils.basic_utils import flat_list_of_lists, load_jsonl, save_json, load_json


def process_single_vid_sub(sub_listdicts, clip_length):
    """
    Args:
        sub_listdicts: list(dicts), each dict is, e.g.,
            {'text': " Chase : That's all this is?", 'start': 0.862, 'end': 1.862}
        clip_length: float
    Returns:
        clip_idx2sentence_indices: dict, {clip_idx: [sen_idx1, sen_idx2, ...]}, which sentences are
            associated with which clips. The indices are in ascending order, i.e., sen_idx1 < sen_idx2 < ...
    """
    timestamps = np.array([[e["start"], e["end"]] for e in sub_listdicts], dtype=np.float32)  # (n_sub_sen, 2)
    timestamps = timestamps / clip_length
    # r-th row of clip_indices is [st_idx, ed_idx), where [st_idx, st_idx+1, ..., ed_idx-1]
    # should be with r-th clip, which is [r*clip_length, (r+1)*clip_length]
    sentence2clip_st_ed = np.empty_like(timestamps, dtype=np.int)
    sentence2clip_st_ed[:, 0] = np.floor(timestamps[:, 0])
    sentence2clip_st_ed[:, 1] = np.ceil(timestamps[:, 1])
    sentence_idx2clip_indices = {sen_idx: set(range(clip_st_idx, clip_ed_idx))
                                 for sen_idx, (clip_st_idx, clip_ed_idx) in enumerate(sentence2clip_st_ed)}
    all_clip_indices = set(flat_list_of_lists(list(sentence_idx2clip_indices.values())))
    clip_idx2sentence_indices = \
        {str(clip_idx): sorted([k for k, v in sentence_idx2clip_indices.items() if clip_idx in v])
         for clip_idx in all_clip_indices}
    return clip_idx2sentence_indices


def load_process_sub_meta(sub_meta_path, clip_length):
    """ which subtitle sentences should be assigned to which clips
    Args:
        sub_meta_path: contains a jsonl file, each line is a dict {"vid_name": str, "sub": list(dicts)},
            each dict under "sub" is, e.g., {'text': " Chase : That's all this is?", 'start': 0.862, 'end': 1.862}.
            The dicts under "sub" are ordered the same as the original .srt files.
        clip_length: float, assign each subtitle sentence to a clip segment
    Returns:
    """
    video2sub = {e["vid_name"]: e for e in load_jsonl(sub_meta_path)}
    for vid_name, sub_info in tqdm(video2sub.items(), desc="processing subtitles"):
        sub_info["clip2sen"] = process_single_vid_sub(sub_info["sub"], clip_length)
        video2sub[vid_name] = sub_info
    return video2sub


def convert_h5(sub_words_h5, vid_clip_h5, sub_clip_h5, video2sub_info, pool_type="max", debug=False):
    assert pool_type in ["max", "avg"]
    np_pool_func = np.max if pool_type == "max" else np.mean
    debug_cnt = 0
    not_equal_cnt = []
    skip_cnt = 0
    for k in tqdm(sub_words_h5.keys(), desc="Converting to clip features"):
        if "-lengths" in k:
            continue
        sub_words_features = sub_words_h5[k]
        sub_sen_lengths = sub_words_h5[k + "-lengths"]
        num_sens = len(sub_sen_lengths)
        clip2sen = video2sub_info[k]["clip2sen"]

        if len(sub_sen_lengths) != len(video2sub_info[k]["sub"]):
            not_equal_cnt.append(len(video2sub_info[k]["sub"]) - len(sub_sen_lengths))

        length_indices = [0, ]
        for i in range(len(sub_sen_lengths)):
            length_indices.append(length_indices[i] + sub_sen_lengths[i])

        n_clips = len(vid_clip_h5[k])
        clip_features = np.zeros((n_clips, sub_words_features.shape[-1]), dtype=np.float32)
        clip_mask = np.zeros(n_clips, dtype=np.float32)
        for clip_idx in range(n_clips):
            if str(clip_idx) in clip2sen:
                # the sen_indices tells which sentences belong to this clip,
                # e.g., [1, 2, 3] mean we should get [1, 4) to include all the indicated sentences
                sen_indices = [min(e, num_sens-1) for e in clip2sen[str(clip_idx)]]
                word_st_idx = length_indices[sen_indices[0]]
                word_ed_idx = length_indices[sen_indices[-1] + 1]
                if word_st_idx == word_ed_idx:
                    skip_cnt += 1
                    continue
                clip_features[clip_idx] = np_pool_func(sub_words_features[word_st_idx:word_ed_idx], axis=0)
                clip_mask[clip_idx] = 1
        sub_clip_h5.create_dataset(k, data=clip_features, dtype=np.float32)
        sub_clip_h5.create_dataset(k + "-mask", data=clip_mask, dtype=np.float32)
        debug_cnt += 1
        if debug and debug_cnt == 5:
            break
    print("skip_cnt {}".format(skip_cnt))
    print("Counter not_equal_cnt {}".format(Counter(not_equal_cnt).most_common()))
    # Counter not_equal_cnt [(1, 150), (2, 7), (4, 1)] for clip_length==1.5


def main_convert():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_h5_file", type=str, help="subtitle words level feature .h5 file")
    parser.add_argument("--vid_clip_h5_file", type=str, help="video clip level feature .h5 file")
    parser.add_argument("--sub_meta_path", type=str, help="processed subtitle .jsonl path")
    parser.add_argument("--tgt_h5_file", type=str, help=".h5 path to stores the converted data")
    parser.add_argument("--pool_type", type=str, default="max",
                        choices=["max", "avg"], help="how to aggreate frame features")
    parser.add_argument("--clip_length", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    sub_info_cache_path = args.tgt_h5_file.replace(".h5", "_sub_info.json")
    if not os.path.exists(sub_info_cache_path):
        video2sub_info = load_process_sub_meta(args.sub_meta_path, clip_length=args.clip_length)
        save_json(video2sub_info, sub_info_cache_path)
    else:
        video2sub_info = load_json(sub_info_cache_path)
    with h5py.File(args.src_h5_file, "r") as src_h5:
        with h5py.File(args.vid_clip_h5_file, "r") as vid_clip_h5:
            with h5py.File(args.tgt_h5_file, "w") as tgt_h5:
                convert_h5(src_h5, vid_clip_h5, tgt_h5, video2sub_info,
                           pool_type=args.pool_type, debug=args.debug)


if __name__ == '__main__':
    main_convert()
