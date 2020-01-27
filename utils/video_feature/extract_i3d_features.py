"""Extract feature ActivityNet i3d RGB/Flow feature
Modified from [1] and [2]
[1] https://github.com/deepmind/kinetics-i3d/blob/master/evaluate_sample.py
[2] https://github.com/tensorflow/hub/blob/master/examples/colab/action_recognition_with_tf_hub.ipynb

Model Notes:
    For model performance on Kinetics-400, please see the repository. In a nutshell,
    1) imagenet_pretrained models are better than scratch models
    2) RGB models are better than Flow models

Dataset Notes:
    1) Kinetics-400 has 400 classes, each with at least 400 video clips,
    2) Kinetics-600 has 600 classes, each with at least 600 video clips.

Please find any missing files/resources/info from https://github.com/deepmind/kinetics-i3d.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import json
import math
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import utils.video_feature.i3d as i3d
import h5py
from multiprocessing import Pool

from utils.basic_utils import save_lines, read_lines, load_pickle, save_pickle

_IMAGE_SIZE = 224
MIN_N_FRAMES = 9
CLIP2N_FRAMES = {  # fps ==15
    1: 15,
    1.5: 23  # evenly separated every 3 seconds,
}


_KINETICS_HOME = "/net/bvisionserver4/playpen10/jielei/tools/VideoFeatureExtraction/kinetics-i3d"
_CHECKPOINT_PATHS = {
    "rgb": os.path.join(_KINETICS_HOME, "data/checkpoints/rgb_scratch/model.ckpt"),
    "rgb600": os.path.join(_KINETICS_HOME, "data/checkpoints/rgb_scratch_kin600/model.ckpt"),
    "flow": os.path.join(_KINETICS_HOME, "data/checkpoints/flow_scratch/model.ckpt"),
    "rgb_imagenet": os.path.join(_KINETICS_HOME, "data/checkpoints/rgb_imagenet/model.ckpt"),
    "flow_imagenet": os.path.join(_KINETICS_HOME, "data/checkpoints/flow_imagenet/model.ckpt"),
}

_LABEL_MAP_PATH = os.path.join(_KINETICS_HOME, "data/label_map.txt")
_LABEL_MAP_PATH_600 = os.path.join(_KINETICS_HOME, "data/label_map_600.txt")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def process_single_image(image_path, resize=(224, 224)):
    img = cv2.imread(image_path)  # BGR image
    img = crop_center_square(img)
    img = cv2.resize(img, resize)
    return int(image_path.split("/")[-1].split(".")[0][-5:]), img[:, :, [2, 1, 0]]


def process_images(multi_pool, image_paths):
    pairs = multi_pool.imap_unordered(process_single_image, image_paths)
    pairs = sorted(pairs, key=lambda x: x[0])
    imgs = [e[1] for e in pairs]
    return np.array(imgs) / 255.0


def mk_divisible(array, divisor):
    """array: (N x _IMAGE_SIZE x _IMAGE_SIZE x 3)
    append N to make it divisible by
    """
    raw_length = len(array)
    residual = raw_length % divisor
    if residual != 0:
        if raw_length < divisor - residual:
            array = np.concatenate([array] + [array] * (int((divisor - residual) / raw_length) + 1))[-divisor:]
        else:
            array = np.concatenate([array, array[-int(divisor-residual):]], axis=0)
    return array


def mk_batch(images_array, batch_size, clip_length=1.5):
    """images_array: N x _IMAGE_SIZE x _IMAGE_SIZE x 3
    return [B x _N_FRAMES x _IMAGE_SIZE x _IMAGE_SIZE x 3, ] (B <= batch_size)
    """
    assert clip_length in CLIP2N_FRAMES
    n_frm = CLIP2N_FRAMES[clip_length]

    if clip_length == 1:
        n_frm = 15
        images_array = mk_divisible(images_array, n_frm)
    elif clip_length == 1.5:
        n_frm = 23  # math.ceil(45 / 2)
        n_frm_3_secs = 45
        clipwise_image_array = []
        for idx in range(math.ceil(len(images_array)/n_frm_3_secs)):
            clipwise_image_array.append(images_array[idx * n_frm_3_secs: idx * n_frm_3_secs + n_frm])
            clipwise_image_array.append(images_array[(idx+1) * n_frm_3_secs - n_frm: (idx+1) * n_frm_3_secs])
        images_array = np.concatenate(
            [mk_divisible(e, n_frm) for e in clipwise_image_array if len(e) > 0], axis=0)

    images_array = images_array.reshape(-1, n_frm, _IMAGE_SIZE, _IMAGE_SIZE, 3)
    n_clips = len(images_array)
    if n_clips > batch_size:
        batches = [images_array[idx * batch_size:(idx + 1) * batch_size] for idx in
                   range(int(n_clips / batch_size) + 1)]
        if len(batches[-1]) == 0:  # when n_clips / batch_size is an integer
            del batches[-1]
        return batches
    else:
        return [images_array]


def get_image_paths(dir_path, image_filename_pattern="img_{:05d}.jpg"):
    """each dir contains the same number of flow_x_{:05d}.jpg, flow_y_{:05d}.jpg, img_{:05d}.jpg.
    Index starts at 1, not 0, thus there is no img_00000.jpg, etc.
    """
    num_rgb_images = int(len(os.listdir(dir_path)) / 3)  # must be divisible by 3
    # original frames are extracted for the following frames, (video fps=30): [1-5], [11-15], [21-25] + 30*n
    selected_img_indices = np.arange(num_rgb_images) + 1  # index starting from 1
    return [image_filename_pattern.format(e) for e in selected_img_indices]


def get_img_info_by_dir(base_dir, cache_file):
    """frm_info_list: list(sublist),
        each sublist[0] is vid_name, sublist[1] is an ordered list of image full paths, """
    if os.path.exists(cache_file):
        tf.logging.info("Found cache file, loading at {}".format(cache_file))
        return load_pickle(cache_file)
    tf.logging.info("Cache file not found, building from scratch")
    frm_info_list = []
    sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for k in tqdm(sub_dirs, desc="Get image info from directory"):
        frm_info_list.append([k, get_image_paths(os.path.join(base_dir, k))])
    save_pickle(frm_info_list, cache_file)
    return frm_info_list


def get_args():
    parser = argparse.ArgumentParser("i3d feature extractor")
    parser.add_argument("--eval_type", type=str, default="rgb600", choices=["rgb", "rgb600"])
    parser.add_argument("--imagenet_pretrained", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=100, help="batch_size * clips")
    parser.add_argument("--base_dir", type=str, help="frame_dir/*/*jpg")
    parser.add_argument("--feature_file", type=str, help="path to save the features")
    parser.add_argument("--cache_file", type=str, help="path to store all the videos")
    parser.add_argument("--clip_length", type=float, default=1.5,
                        help="clip length in seconds, each clip will have its own feature")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    tf.logging.info("Args: %s", json.dumps(vars(args), indent=4, sort_keys=True))
    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    args = get_args()
    eval_type = args.eval_type
    imagenet_pretrained = args.imagenet_pretrained

    NUM_CLASSES = 600 if eval_type == "rgb600" else 400

    if eval_type not in ["rgb", "rgb600", "flow", "joint"]:
        raise ValueError("Bad `eval_type`, must be one of rgb, rgb600, flow, joint")

    frame_infos = get_img_info_by_dir(args.base_dir, cache_file=args.cache_file)

    n_frm = CLIP2N_FRAMES[args.clip_length]
    assert n_frm >= MIN_N_FRAMES, "Number of input frames must be larger than or equal to 9"

    # RGB input has 3 channels.
    rgb_input = tf.placeholder(tf.float32, shape=(None, n_frm, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    with tf.variable_scope("RGB"):
        rgb_model = i3d.InceptionI3d(NUM_CLASSES, spatial_squeeze=True, final_endpoint="Logits")
        rgb_logits, end_points = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if eval_type == "rgb600":
            rgb_variable_map[variable.name.replace(":0", "")[len("RGB/inception_i3d/"):]] = variable
        else:
            rgb_variable_map[variable.name.replace(":0", "")] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.Session() as sess:
        feed_dict = {}
        if imagenet_pretrained:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS["rgb_imagenet"])
        else:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
        tf.logging.info("RGB checkpoint restored")

        feed_dict[rgb_input] = np.random.randn(args.batch_size, n_frm, _IMAGE_SIZE, _IMAGE_SIZE, 3)
        avg_pool3d_feature = sess.run([end_points["avg_pool3d"]], feed_dict=feed_dict)[0]
        avg_pool3d_feature = np.squeeze(avg_pool3d_feature, axis=(1, 2, 3))
        tf.logging.info("Test input size {}, output feature size {}"
                        .format(feed_dict[rgb_input].shape, avg_pool3d_feature.shape))

        pool = Pool(24)
        feat_h5 = h5py.File(args.feature_file, "a")
        exist_keys = list(feat_h5.keys())
        debug_loop_cnt = 10
        frame_infos = [e for e in frame_infos if e[0] not in exist_keys]
        for videoname, frame_paths in tqdm(frame_infos, desc="Extracting"):
            frame_paths = [os.path.join(args.base_dir, videoname, e) for e in frame_paths]
            debug_loop_cnt -= 1
            if args.debug and debug_loop_cnt == 0:
                break
            try:
                images = process_images(pool, frame_paths)
                if len(images) == 0:
                    continue

                batches = mk_batch(images, args.batch_size, clip_length=args.clip_length)
                features = []
                for batch in batches:
                    feed_dict[rgb_input] = batch
                    avg_pool3d_feature = sess.run([end_points["avg_pool3d"]], feed_dict=feed_dict)[0]
                    avg_pool3d_feature = np.squeeze(avg_pool3d_feature, axis=(1, 2, 3))
                    features.append(avg_pool3d_feature)

                # write to file
                feat_h5.create_dataset(videoname, data=np.concatenate(features, axis=0), dtype=np.float32)
            except Exception as e:
                print("Exception ", e)
                continue

        feat_h5.close()
        pool.close()


if __name__ == "__main__":
    tf.app.run(main)



