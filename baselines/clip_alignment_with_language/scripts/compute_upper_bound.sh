#!/usr/bin/env bash
# run at project root dir
dset_name=$1  # see case below
split_name=$2  # train/val/test, some datasets may not support all the 3 splits
result_dir="baselines/clip_alignment_with_language/results"

echo "Running with dataset ${dset_name} with split ${split_name}"
case ${dset_name} in
    tvr)  # only supports train/val
        eval_file_path=data/tvr_${split_name}_release.jsonl
        save_path=${result_dir}/tvr_${split_name}_proposal_upper_bound.json
        ;;
    *)
        echo -n "Unknown argument"
        ;;
esac

echo "Running evaluation"
python baselines/clip_alignment_with_language/local_utils/compute_proposal_upper_bound.py \
-dset_name=${dset_name} \
-eval_file_path=${eval_file_path} \
-save_path=${save_path} \
-verbose
