#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/crossmodal_moment_localization/scripts/inference_with_external.sh
#model_dir=$1
# DO not use NMS, since it gives worse results
eval_model=$1  # [xml, xml_tef]
eval_split_name=$2
external_model=mee  # [mee, mcn, cal]
eval_path=data/tvr_${eval_split_name}_release.jsonl
project_root=./baselines

# setup eval model
if [[ ${eval_model} == xml ]]; then
    eval_model_dir=tvr-video_sub-resnet_i3d_no_norm_v-2019_11_03_12_22_19
elif [[ ${eval_model} == xml_tef ]]; then
    eval_model_dir=tvr-video_sub_tef-resnet_i3d_no_norm_v-2019_11_03_12_53_01
fi

# setup external
if [[ ${external_model} == mee ]]; then
    external_model_dir=tvr-video_sub-res-2019_11_06_00_33_39
    external_inference_vr_res_path=${project_root}/mixture_embedding_experts/results/${external_model_dir}/inference_tvr_${eval_split_name}_None_predictions_VR.json
fi

tasks=(VR)
tasks+=(SVMR)
tasks+=(VCMR)
echo "tasks ${tasks[@]}"
python baselines/crossmodal_moment_localization/inference.py \
--model_dir ${eval_model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--external_inference_vr_res_path ${external_inference_vr_res_path} \
--eval_id ${external_model_dir} \
${@:3}

#--use_intermediate \  # temporary removed

