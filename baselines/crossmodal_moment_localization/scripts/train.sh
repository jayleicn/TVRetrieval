#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/crossmodal_moment_localization/scripts/train.sh tvr all ANY_OTHER_PYTHON_ARGS
# use --eval_tasks_at_training ["VR", "SVMR", "VCMR"] --stop_task ["VR", "SVMR", "VCMR"] for
# use --lw_neg_q 0 --lw_neg_ctx 0 for training SVMR/SVMR only
# use --lw_st_ed 0 for training with VR only
dset_name=$1  # see case below
ctx_mode=$2  # [video, sub, tef, video_sub, video_tef, sub_tef, video_sub_tef]
vid_feat_type=$3  # [resnet, i3d, resnet_i3d]
feature_root=data/tvr_feature_release
results_root=baselines/crossmodal_moment_localization/results
vid_feat_size=2048
extra_args=()

if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
    if [[ ${dset_name} != "tvr" ]]; then
        echo "The use of subtitles is only supported in tvr."
        exit 1
    fi
fi


case ${dset_name} in
    tvr)
        train_path=data/tvr_train_release.jsonl
        video_duration_idx_path=data/tvr_video2dur_idx.json
        desc_bert_path=${feature_root}/bert_feature/query_only/tvr_query_pretrained_w_query.h5
        if [[ ${vid_feat_type} == "i3d" ]]; then
            echo "Using I3D feature with shape 1024"
            vid_feat_path=${feature_root}/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5
            vid_feat_size=1024
        elif [[ ${vid_feat_type} == "resnet" ]]; then
            echo "Using ResNet feature with shape 2048"
            vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5
            vid_feat_size=2048
        elif [[ ${vid_feat_type} == "resnet_i3d" ]]; then
            echo "Using concatenated ResNet and I3D feature with shape 2048+1024"
            vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5
            vid_feat_size=3072
            extra_args+=(--no_norm_vfeat)  # since they are already normalized.
        fi
        eval_split_name=val
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=(data/tvr_val_release.jsonl)
        clip_length=1.5
        extra_args+=(--max_ctx_l)
        extra_args+=(100)  # max_ctx_l = 100 for clip_length = 1.5, only ~109/21825 has more than 100.
        extra_args+=(--max_pred_l)
        extra_args+=(16)
        if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
            echo "Running with sub."
            desc_bert_path=${feature_root}/bert_feature/sub_query/tvr_query_pretrained_w_sub_query.h5  # overwrite
            sub_bert_path=${feature_root}/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5
            sub_feat_size=768
            extra_args+=(--sub_feat_size)
            extra_args+=(${sub_feat_size})
            extra_args+=(--sub_bert_path)
            extra_args+=(${sub_bert_path})
        fi
        ;;
    *)
        echo -n "Unknown argument"
        ;;
esac

echo "Start training with dataset [${dset_name}] in Context Mode [${ctx_mode}]"
echo "Extra args ${extra_args[@]}"
python baselines/crossmodal_moment_localization/train.py \
--dset_name=${dset_name} \
--eval_split_name=${eval_split_name} \
--nms_thd=${nms_thd} \
--results_root=${results_root} \
--train_path=${train_path} \
--desc_bert_path=${desc_bert_path} \
--video_duration_idx_path=${video_duration_idx_path} \
--vid_feat_path=${vid_feat_path} \
--clip_length=${clip_length} \
--vid_feat_size=${vid_feat_size} \
--ctx_mode=${ctx_mode} \
${extra_args[@]} \
${@:4}
