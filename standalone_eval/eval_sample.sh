#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
submission_path=standalone_eval/sample_val_predictions.json
gt_path=data/tvr_val_release.jsonl
save_path=standalone_eval/sample_val_predictions_metrics.json

python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
