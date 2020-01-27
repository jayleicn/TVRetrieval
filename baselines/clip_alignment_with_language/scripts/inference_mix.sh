#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/clip_alignment_with_language/scripts/inference_mix.sh
eval_model=$1  # [mcn, cal], retrain models should only be paired with mee
project_root=/net/bvisionserver14/playpen-ssd/jielei/projects/video_retrieval/baselines/clip_alignment_with_language/results

# setup eval model
if [[ ${eval_model} == mcn ]]; then
    pred_dir=tvr-mcn-video_sub-res-2019_11_05_14_16_40
    tef_pred_dir=tvr-mcn-video_sub_tef-res-2019_11_05_14_14_57
elif [[ ${eval_model} == cal ]]; then
    pred_dir=tvr-cal-video_sub-res-2019_11_05_14_32_59
    tef_pred_dir=tvr-cal-video_sub_tef-res-2019_11_05_14_25_49
fi

pred_path=${project_root}/${pred_dir}/inference_tvr_test_public_max200_predictions_VR_SVMR_VCMR.json
save_path=${project_root}/${pred_dir}/inference_tvr_test_public_max200_predictions_VR_SVMR_VCMR_rerank_${tef_pred_dir}.json
tef_pred_path=${project_root}/${tef_pred_dir}/inference_tvr_test_public_max10000_predictions_VCMR.pt
gt_path=data/tvr_test_public_archive.jsonl


python baselines/clip_alignment_with_language/mix_model_prediction.py \
--pred_path=${pred_path} \
--tef_pred_path=${tef_pred_path} \
--gt_path=${gt_path} \
--save_path=${save_path}
