"""
Convert frame level (FPS1) features for videos to clip level (FPS2) features, by pooling across multiple frames.

FeaturePerSecond (FPS): FPS1 > FPS2.
"""
import os
import h5py
import numpy as np
from tqdm import tqdm


def convert_for_single_h5(frm_h5, clip_h5, clip_boundaries_in_frm_idx, pool_type="max", debug=False):
    """
    Args:
        frm_h5: h5py.File object, containing the frame level features
        clip_h5: h5py.File object, containing the clip level features
        clip_boundaries_in_frm_idx: list, features belong to clip `clip_idx` should be indexed as
            features[clip_boundaries_in_frm_idx[clip_idx]:clip_boundaries_in_frm_idx[clip_idx+1]]
        pool_type: max or avg
        debug:
    Returns:

    """
    assert pool_type in ["max", "avg"]
    np_pool_func = np.max if pool_type == "max" else np.mean
    for k in tqdm(frm_h5.keys()):
        frm_features = frm_h5[k]
        clip_features = []
        for idx in range(len(clip_boundaries_in_frm_idx)):
            cur_clip_feat = frm_features[clip_boundaries_in_frm_idx[idx]:clip_boundaries_in_frm_idx[idx+1]]
            if len(cur_clip_feat) == 0:
                break
            cur_clip_feat = np_pool_func(cur_clip_feat, axis=0, keepdims=True)
            clip_features.append(cur_clip_feat)
        clip_h5.create_dataset(k, data=np.concatenate(clip_features, axis=0), dtype=np.float32)
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
    parser.add_argument("--pool_type", type=str, default="max",
                        choices=["max", "avg"], help="how to aggreate frame features")
    parser.add_argument("--clip_length", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    clip_boundaries_in_frm_idx = get_clip2frm_idx_mapping(clip_length=args.clip_length)
    assert not os.path.exists(args.tgt_h5_file)
    with h5py.File(args.tgt_h5_file, "a") as tgt_h5:
        for src_f in args.src_h5_files:
            with h5py.File(src_f, "r") as src_h5:
                convert_for_single_h5(src_h5, tgt_h5, clip_boundaries_in_frm_idx,
                                      pool_type=args.pool_type, debug=args.debug)


if __name__ == '__main__':
    main_convert()
