"""
Merge i3d features from all shows. Meanwhile, align it with the imagenet feature
so that they have the same number of feature vectors.
"""
import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import Counter


def convert_for_single_h5(src_h5, tgt_h5, align_h5_key2len, debug=False):
    """
    Args:
        src_h5: h5py.File object, containing the frame level features
        tgt_h5: h5py.File object, containing the clip level features
        align_h5_key2len: dict, {key: len}, each value indicates the length (L) of the array (L, D)
        debug:
    Returns:

    """
    for k, feat in tqdm(src_h5.items()):
        if k in align_h5_key2len:
            if len(feat) != align_h5_key2len[k]:
                align_len = align_h5_key2len[k]
                aligned_feat = np.zeros((align_h5_key2len[k], feat.shape[1]), dtype=np.float32)
                aligned_feat[:len(feat)] = feat[:align_len]
                feat = aligned_feat
            tgt_h5.create_dataset(k, data=feat, dtype=np.float32)
        else:
            print("Skipping {}".format(k))
        if debug:
            break


def get_clip2frm_idx_mapping(clip_length=1.5, max_video_length=300):
    """ This function depends on how the features are extracted.
    original features are extract from frames (video fps=30):
    [3, 13, 23] frame in a second.
    Args:
        clip_length: float,
        max_video_length: int,

    Returns:
        {clip_idx1 (int): [frm_idx0, frm_idx1, ...],
         ...
        }
    """
    # frame 0 in the feature is actually the frame 3 in the original video, so its
    # corresponding time is 3 / 30 = 0.1s. More generally ==> [0.1, 0.43, 0.77] + n.
    frm2seconds = np.concatenate([
        np.array([3, 13, 23]) / 30. + offset for offset in np.arange(0, max_video_length)], axis=0)

    clip_boundaries = np.arange(0, max_video_length, clip_length)
    # no need to worry about search boundary.
    # indexed as clip_boundaries_in_frm_idx[idx]:clip_boundaries_in_frm_idx[idx+1]
    clip_boundaries_in_frm_idx = np.searchsorted(frm2seconds, clip_boundaries)
    return clip_boundaries_in_frm_idx


def main_convert():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_h5_files", type=str, nargs='+', help="frm .h5 file paths")
    parser.add_argument("--tgt_h5_file", type=str, help=".h5 path to stores the converted data")
    parser.add_argument("--align_h5_file", type=str, help=".h5 path to the file to align at length dim")
    parser.add_argument("--check_alignment_only", action="store_true", help="Check alignment only")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.align_h5_file, "r") as align_h5:
        align_h5_key2len = {k: len(v) for k, v in tqdm(align_h5.items(), desc="[Get Length] Loop over align h5")}

    src_h5_key2len = {}
    for src_f in args.src_h5_files:
        with h5py.File(src_f, "r") as src_h5:
            for k, v in tqdm(src_h5.items(), desc="[Get length] Loop over one of the src h5"):
                src_h5_key2len[k] = len(v)

    not_found_keys = list(set(align_h5_key2len.keys()) - set(src_h5_key2len.keys()))
    diff_key2len = {k: align_h5_key2len[k] - src_h5_key2len[k] for k in align_h5_key2len if k in src_h5_key2len}
    diff_counter = Counter(list(diff_key2len.values()))
    print("Not found keys total {}, examples: {}".format(len(not_found_keys), not_found_keys[:3]))
    print("diff_counter {}".format(diff_counter.most_common()))

    if not args.check_alignment_only:
        assert not os.path.exists(args.tgt_h5_file)
        with h5py.File(args.tgt_h5_file, "a") as tgt_h5:
            for src_f in args.src_h5_files:
                with h5py.File(src_f, "r") as src_h5:
                    convert_for_single_h5(src_h5, tgt_h5, align_h5_key2len, debug=args.debug)


if __name__ == '__main__':
    main_convert()
