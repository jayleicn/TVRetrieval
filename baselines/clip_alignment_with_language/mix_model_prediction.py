"""
Implement the CAL + CAL (TEF) model mentioned in
```
@article{Escorcia2019TemporalLO,
  title={Temporal Localization of Moments in Video Collections with Natural Language},
  author={Victor Escorcia and Mattia Soldan and Josef Sivic and Bernard Ghanem and Bryan Russell},
  journal={ArXiv},
  year={2019},
  volume={abs/1907.12763}
}
```

Methods:
    1, Give top200 predictions for each query in CAL then using CAL (TEF) to re-rank.
    2, This is approximated by re-ranking the top200 CAL using top1000 CAL(TEF) -- we assume they will be all covered.
"""

import torch
import subprocess
import numpy as np
from tqdm import tqdm
from utils.basic_utils import load_json, save_json


def load_saved_res(pred_path):
    if pred_path.endswith(".json"):
        pred = load_json(pred_path)
    else:
        pred = torch.load(pred_path)
    vcmr_res = {e["desc_id"]: e for e in pred["VCMR"]}
    video2idx = pred["video2idx"]
    return vcmr_res, video2idx


def main_mix_results(pred_path, tef_pred_path, save_path, max_after_nms=100):
    """
    Args:
        pred_path: contains top-200 VCMR predictions
        tef_pred_path: contains top-1000 VCMR predictions
        save_path:
        max_after_nms: int,
    Returns:
        save
    """
    vcmr_res, video2idx = load_saved_res(pred_path)
    tef_vcmr_res, video2idx = load_saved_res(tef_pred_path)

    reranked_vcmr_res = {}
    num_valid = []
    for desc_id, preds in tqdm(vcmr_res.items(), desc="Loop over the predictions"):
        tef_preds = tef_vcmr_res[desc_id]["predictions"]
        pred_moments = set([tuple(e[:3]) for e in preds["predictions"]])
        reranked_moments = [e for e in tef_preds if tuple(e[:3]) in pred_moments][:max_after_nms]
        num_valid += [len(reranked_moments)]
        if len(reranked_moments) != 100:
            reranked_moments += reranked_moments[:100 - len(reranked_moments)]
        reranked_vcmr_res[desc_id] = dict(
            predictions=reranked_moments,
            desc_id=desc_id,
            desc=preds["desc"]
        )

    print("There are {} moments founded on average".format(np.mean(num_valid)))
    reranked_predictions = dict(
        VCMR=list(reranked_vcmr_res.values()),
        video2idx=video2idx
    )

    save_json(reranked_predictions, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, help="path to prediction res")
    parser.add_argument("--tef_pred_path", type=str, help="path to TEF prediction res")
    parser.add_argument("--save_path", type=str, help="path to save the re-ranked predictions, same dir as --pred_path")
    parser.add_argument("--gt_path", type=str, help="path to ground truth file")
    args = parser.parse_args()

    main_mix_results(args.pred_path, args.tef_pred_path, args.save_path)

    metrics_path = args.save_path.replace(".json", "_metrics.json")
    eval_cmd = "python standalone_eval/eval.py --submission_path " + args.save_path + " --gt_path " + args.gt_path + \
        " --save_path " + metrics_path
    results = subprocess.run(eval_cmd, shell=True)
