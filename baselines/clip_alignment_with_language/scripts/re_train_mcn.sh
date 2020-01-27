#!/usr/bin/env bash

lr=0.00005
n_epoch=20
project_root=/net/bvisionserver14/playpen-ssd/jielei/projects/video_retrieval
ckpt_filename="model.ckpt"
init_ckpt_path=${project_root}/baselines/clip_alignment_with_language/results/tvr-mcn-video_sub_tef-res-2019_11_05_14_14_57/${ckpt_filename}
exp_id=+ex_vr_mee_tvr-video_sub-res-2019_11_06_00_33_39_tvr-mcn-video_sub_tef-res-2019_11_05_14_14_57+
external_train_vr_res_path=${project_root}/baselines/mixture_embedding_experts/results/tvr-video_sub-res-2019_11_06_00_33_39/inference_tvr_train_None_predictions_VR.json
model_type=mcn

bash baselines/clip_alignment_with_language/scripts/train.sh tvr video_sub_tef resnet_i3d \
--no_norm_vfeat \
--model_type ${model_type} \
--exp_id ${exp_id} \
--init_ckpt_path ${init_ckpt_path} \
--external_train_vr_res_path ${external_train_vr_res_path} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--max_es_cnt 5 \
${@:1}
