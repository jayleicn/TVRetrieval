#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/clip_alignment_with_language/scripts/inference_with_external.sh
#model_dir=$1
# DO not use NMS, since it gives worse results
eval_model=$1  # [mcn, mcn_tef, cal, cal_tef, mcn_retrain, cal_retrain], retrain models should only be paired with mee
external_model=$2  # [mee, mcn, cal]
eval_split_name=$3
eval_path=data/tvr_${eval_split_name}_release.jsonl
project_root=/net/bvisionserver14/playpen-ssd/jielei/projects/video_retrieval/baselines

# setup eval model
if [[ ${eval_model} == mcn ]]; then
    eval_model_dir=tvr-mcn-video_sub-res-2019_11_05_14_16_40
elif [[ ${eval_model} == mcn_tef ]]; then
    eval_model_dir=tvr-mcn-video_sub_tef-res-2019_11_05_14_14_57
elif [[ ${eval_model} == cal ]]; then
    eval_model_dir=tvr-cal-video_sub-res-2019_11_05_14_32_59
elif [[ ${eval_model} == cal_tef ]]; then
    eval_model_dir=tvr-cal-video_sub_tef-res-2019_11_05_14_25_49
elif [[ ${eval_model} == mcn_tef_retrain ]]; then
    eval_model_dir=tvr-mcn-video_sub_tef-+ex_vr_mee_tvr-video_sub-res-2019_11_06_00_33_39_tvr-mcn-video_sub_tef-res-2019_11_05_14_14_57+-2019_11_06_02_26_49
elif [[ ${eval_model} == cal_tef_retrain ]]; then
    eval_model_dir=tvr-cal-video_sub_tef-+ex_vr_mee_tvr-video_sub-res-2019_11_06_00_33_39_tvr-cal-video_sub_tef-res-2019_11_05_14_25_49+-2019_11_06_03_12_15
fi

# setup external
if [[ ${external_model} == mee ]]; then
    external_model_dir=tvr-video_sub-res-2019_11_06_00_33_39
    external_inference_vr_res_path=${project_root}/mixture_embedding_experts/results/${external_model_dir}/inference_tvr_${eval_split_name}_None_predictions_VR.json
elif [[ ${external_model} == mcn ]]; then
    external_model_dir=tvr-mcn-video_sub-res-2019_11_05_14_16_40
    external_inference_vr_res_path=${project_root}/clip_alignment_with_language/results/${external_model_dir}/inference_tvr_${eval_split_name}_None_predictions_VR_SVMR_VCMR.json
elif [[ ${external_model} == cal ]]; then
    external_model_dir=tvr-cal-video_sub-res-2019_11_05_14_32_59
    external_inference_vr_res_path=${project_root}/clip_alignment_with_language/results/${external_model_dir}/inference_tvr_${eval_split_name}_None_predictions_VR_SVMR_VCMR.json
fi

tasks=(VR)
tasks+=(SVMR)
tasks+=(VCMR)
echo "tasks ${tasks[@]}"
python baselines/clip_alignment_with_language/inference.py \
--model_dir ${eval_model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--external_inference_vr_res_path ${external_inference_vr_res_path} \
--eval_id ${external_model_dir} \
${@:4}

#--use_intermediate \  # temporary removed

