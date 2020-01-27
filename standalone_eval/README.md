TVR Evalation
================================================================

### Task Definition
Given a natural language query and a large pool of videos (with subtitles),
the TVR (VCMR) task requires a system to retrieve a relevant moment from the videos.
The table below shows a comparison of the TVR task and the subtasks: 

| Task | Description |
| --- | --- | 
| VCMR | or VSCMR, *Video (-Subtitle) Corpus Moment Retrieval*. Localize a moment from a large video corpus. |
| SVMR | or SVSMR, *Single Video (-Subtitle) Moment Retrieval*. Localize a moment from a given video. |
| VR | or VSR, *Video (-Subtitle) Retrieval*. Retrieve a video from a large video corpus. |

VCMR and VR only requires a query and a video corpus, SVMR additionally requires knowing the ground-truth video. 
Thus it is not possible to perform SVMR on our `test-public` set, where the ground-truth video is hidden. 


### How to construct a prediction file?

An example of such file is [sample_val_predictions.json](sample_val_predictions.json), it is formatted as:
```
{
    "video2idx": {
        "castle_s01e02_seg02_clip_09": 19614,
        ...
    },
    "VCMR": [{
            "desc_id": 90200,
            "desc": "Phoebe puts one of her ponytails in her mouth.",
            "predictions": [
                [19614, 9.0, 12.0, 1.7275],
                [20384, 12.0, 18.0, 1.7315],
                [20384, 15.0, 21.0, 1.7351],
                ...
            ]
        },
        ...
    ],
    "SVMR": [{
            "desc_id": 90200,
            "desc": "Phoebe puts one of her ponytails in her mouth.",
            "predictions": [
                [20092, 36.0, 42.0, -1.9082],
                [20092, 18.0, 24.0, -1.9145],
                [20092, 51.0, 54.0, -1.922],
                ...
            ]
        },
        ...
    ],
    "VR": [{
            "desc_id": 90200,
            "desc": "Phoebe puts one of her ponytails in her mouth.",
            "predictions": [
                [19614, 0, 0, 1.7275],
                [20384, 0, 0, 1.7315],
                [20384, 0, 0, 1.7351],
                ...
            ]
        },
        ...
    ]
}
``` 

| entry | description |
| --- | ----|
| video2idx | `dict`, `{vid_name: vid_idx}`. A mapping of video names to unique video IDs for current set. From [tvr_video2dur_idx.json](../data/tvr_video2dur_idx.json). |
| VCMR | `list(dicts)`, stores predictions for the task `VCMR`. | 
| SVMR | `list(dicts)`, stores predictions for the task `SVMR`. Not required for `test-public` submission. | 
| VR | `list(dicts)`, stores predictions for the task `VR`. | 

The evaluation script will evaluate the predictions for tasks `[VCMR, SVMR, VR]` independently.
Each dict in VCMR/SVMR/VR list is:
```
{
    "desc": str,
    "desc_id": int,
    "predictions": [[vid_id (int), st (float), ed (float), score (float)], ...]
}
```

`predictions` is a `list` containing 100 `sublist`, each `sublist` has exactly 4 items: 
`[vid_id (int), st (float), ed (float), score (float)]`,
which are `vid_id` (video id), `st` and `ed` (moment start and end time, in seconds.), 
`score` (score of the prediction). 
The `score` item will not be used in the evaluation script, it is left here for record. 

 
### Run Evaluation
At project root, run
```
bash standalone_eval/eval_sample.sh 
```
This command will use [eval.py](eval.py) to evaluate the provided `sample_val_predictions.json` file, 
the output will be written into `sample_val_predictions_metrics.json`. 
Its content should be similar if not the same as `sample_val_predictions_metrics_raw.json` file.


### Codalab Submission
To test your model's performance on `test-public` set, 
please submit both `val` and `test-public` predictions to our 
[Codalab evaluation server](https://competitions.codalab.org/competitions/22780). 
The submission file should be a single `.zip ` file (no enclosing folder) 
that contains the two prediction files 
`tvr_test_public_submission.json` and `tvr_val_submission.json`, each of the `*submission.json` file 
should be formatted as instructed above. 
Note that `tvr_val_submission.json` will have all the 4 entries, while 
`tvr_test_public_submission.json` will have only 3 entries, without `SVMR`.


