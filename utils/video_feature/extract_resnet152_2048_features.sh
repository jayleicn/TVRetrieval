#!/usr/bin/env bash
show_name=$1
feature_root=/net/bvisionserver14/playpen-ssd/jielei/data/tvr/video_feature
image_root=/net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/dense_flow_frames_step1_new
feature_file=${feature_root}/tvr_${show_name}_resnet152_3fps.h5
cache_dir=cache


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


python utils/video_feature/extract_image_features.py \
--feature_file=${feature_file} \
--base_dir=${base_dir} \
--feature_type=2048 \
--batch_size=300 \
--cache_dir=${cache_dir} \
${@:2}
