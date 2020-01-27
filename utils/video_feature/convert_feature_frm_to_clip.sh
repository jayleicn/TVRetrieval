#!/usr/bin/env bash
# Usage:
# bash utils/video_feature/convert_feature_frm_to_clip.sh [clip_length] ANY_OTHER_PYTHON_ARGS
clip_length=${1}
feature_root=/net/bvisionserver14/playpen-ssd/jielei/data/tvr/video_feature
src_h5_files=()
for show_name in bbt friends grey house met castle
do
    cur_src_h5_file=${feature_root}/frm_level_resnet152/tvr_${show_name}_resnet152_3fps.h5
    src_h5_files+=(${cur_src_h5_file})
done
echo "Running with src_h5_files ${src_h5_files}"

pool_type=max
tgt_h5_file=${feature_root}/tvr_resnet152_rgb_${pool_type}_cl-${clip_length}.h5

python utils/video_feature/convert_feature_frm_to_clip.py \
--src_h5_files ${src_h5_files[@]} \
--tgt_h5_file ${tgt_h5_file} \
--pool_type ${pool_type} \
--clip_length ${clip_length} \
${@:2}
