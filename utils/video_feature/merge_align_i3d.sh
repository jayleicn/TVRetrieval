#!/usr/bin/env bash
# Usage:
# bash utils/video_feature/merge_align_i3d.sh [clip_length] ANY_OTHER_PYTHON_ARGS
clip_length=${1}
feature_root=/net/bvisionserver14/playpen-ssd/jielei/data/tvr/video_feature
src_h5_files=()
for show_name in bbt friends grey house met castle
do
    cur_src_h5_file=${feature_root}/i3d_featrues_by_show/tvr_${show_name}_i3d_rgb600_avg_cl-${clip_length}.h5
    src_h5_files+=(${cur_src_h5_file})
done
echo "Running with src_h5_files ${src_h5_files}"

pool_type=max
tgt_h5_file=${feature_root}/tvr_i3d_rgb600_avg_cl-${clip_length}.h5
align_h5_file=${feature_root}/tvr_resnet152_rgb_max_cl-${clip_length}.h5

python utils/video_feature/merge_align_i3d.py \
--src_h5_files ${src_h5_files[@]} \
--tgt_h5_file ${tgt_h5_file} \
--align_h5_file ${align_h5_file} \
${@:2}
