#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/excl/scripts/inference.sh ANY_OTHER_PYTHON_ARGS
model_dir=$1
eval_split_name=$2
eval_path=data/tvr_${eval_split_name}_release.jsonl
tasks=()
tasks+=(VCMR)

project_root=./baselines
external_model_dir=tvr-video_sub-res-2019_11_06_00_33_39
external_inference_vr_res_path=${project_root}/mixture_embedding_experts/results/${external_model_dir}/inference_tvr_${eval_split_name}_None_predictions_VR.json


echo "tasks ${tasks[@]}"
python baselines/excl/inference_with_vcmr.py \
--model_dir ${model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--external_inference_vr_res_path ${external_inference_vr_res_path} \
--eval_id ${external_model_dir} \
--eval_path ${eval_path} \
${@:3}
