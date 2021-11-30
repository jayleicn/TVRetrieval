#!/usr/bin/env bash
# Usage:
# bash utils/text_feature/convert_sub_feature_word_to_clip.sh POOL_TYPE CLIP_LENGTH [--debug]

pool_type=$1  # [max, avg]
clip_length=$2
sub_token_h5_file=$3
sub_clip_h5_file=$4
vid_clip_h5_file=$5  # .h5 file stores the clip-level video features, to make sure subtitle clip-level features have the same length as the video features.
sub_meta_path=data/tvqa_preprocessed_subtitles.jsonl

python utils/text_feature/convert_sub_feature_word_to_clip.py \
--pool_type ${pool_type} \
--clip_length ${clip_length} \
--src_h5_file ${sub_token_h5_file} \
--tgt_h5_file ${sub_clip_h5_file} \
--sub_meta_path ${sub_meta_path} \
--vid_clip_h5_file ${vid_clip_h5_file} \
${@:6}
