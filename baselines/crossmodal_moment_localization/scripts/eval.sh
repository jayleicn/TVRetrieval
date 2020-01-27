#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/crossmodal_moment_localization/scripts/eval.sh ANY_OTHER_PYTHON_ARGS
eval_split_name=$1
submission_path=$2
save_path=$3
gt_path=data/tvr_${eval_split_name}_release.jsonl

python standalone_eval/eval.py \
--gt_path ${gt_path} \
--submission_path ${submission_path} \
--save_path ${save_path} \
${@:4}
