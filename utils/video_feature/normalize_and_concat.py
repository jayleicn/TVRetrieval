"""
L2 Normalize then concat I3D and ResNet features
"""
import os
import h5py
import numpy as np
from tqdm import tqdm
from utils.basic_utils import l2_normalize_np_array


def main_norm_cat():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_h5_file", type=str, help="ResNet .h5 file paths")
    parser.add_argument("--i3d_h5_file", type=str, help="I3D .h5 file paths")
    parser.add_argument("--tgt_h5_file", type=str, help=".h5 path to stores the converted data")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    assert not os.path.exists(args.tgt_h5_file)
    with h5py.File(args.resnet_h5_file, "r") as resnet_h5:
        with h5py.File(args.i3d_h5_file, "r") as i3d_h5:
            with h5py.File(args.tgt_h5_file, "w") as tgt_h5:
                for k in tqdm(resnet_h5.keys()):
                    resnet_feat = l2_normalize_np_array(resnet_h5[k][:])
                    i3d_feat = l2_normalize_np_array(i3d_h5[k][:])
                    tgt_h5.create_dataset(k,
                                          data=np.concatenate([resnet_feat, i3d_feat], axis=-1),
                                          dtype=np.float32)


if __name__ == '__main__':
    main_norm_cat()
