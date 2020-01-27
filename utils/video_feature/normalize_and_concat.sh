#!/usr/bin/env bash
# Usage:
# bash utils/video_feature/normalize_and_concat.sh [clip_length] ANY_OTHER_PYTHON_ARGS
clip_length=${1}
feature_root=/net/bvisionserver14/playpen-ssd/jielei/data/tvr/video_feature
resnet_h5_file=${feature_root}/tvr_resnet152_rgb_max_cl-${clip_length}.h5
i3d_h5_file=${feature_root}/tvr_i3d_rgb600_avg_cl-${clip_length}.h5
tgt_h5_file=${feature_root}/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-${clip_length}.h5

python utils/video_feature/normalize_and_concat.py \
--resnet_h5_file ${resnet_h5_file} \
--i3d_h5_file ${i3d_h5_file} \
--tgt_h5_file ${tgt_h5_file} \
${@:2}
