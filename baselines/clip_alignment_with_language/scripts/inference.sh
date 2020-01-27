#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/clip_alignment_with_language/scripts/inference.sh ANY_OTHER_PYTHON_ARGS
model_dir=$1
eval_split_name=$2
eval_path=data/tvr_${eval_split_name}_release.jsonl
tasks=(VR)
tasks+=(SVMR)
tasks+=(VCMR)
echo "tasks ${tasks[@]}"
python baselines/clip_alignment_with_language/inference.py \
--model_dir ${model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
