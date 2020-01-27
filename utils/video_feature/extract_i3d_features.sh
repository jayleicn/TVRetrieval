#!/usr/bin/env bash
show_name=$1
clip_length=$2
eval_type=rgb600
feature_root=/net/bvisionserver14/playpen-ssd/jielei/data/tvr/video_feature
image_root=/net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/dense_flow_frames_step1_new
feature_file=${feature_root}/tvr_${show_name}_i3d_${eval_type}_avg_cl-${clip_length}.h5  # !!!!! TODO
cache_file=cache/tvr_${show_name}_vid_all_frm_pairs.pkl


echo "Running with show ${show_name}"
case ${show_name} in
    bbt)
        base_dir=${image_root}/new_bbt
        ;;
    friends | grey | house | met | castle)
        base_dir=${image_root}/${show_name}
        ;;
    *)
        echo -n "Unknown argument"
        ;;
esac


python utils/video_feature/extract_i3d_features.py \
--eval_type=${eval_type} \
--batch_size=60 \
--base_dir=${base_dir} \
--feature_file=${feature_file} \
--cache_file=${cache_file} \
--clip_length=${clip_length} \
${@:3}
