Language Model Fine-tuning and Feature Extraction
====

### Install Dependencies

The code requires installing [transformers](https://github.com/huggingface/transformers) package as well as [tensorboardX](https://github.com/lanpa/tensorboardX):
```
# install transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout e1b2949ae6cb34cc39e3934ca87423474f8c8d02
pip install .

# install tensorboardX
pip install tensorboardX
```

###  Language Model Fine-tuning

We fine-tune pre-trained [RoBERTa](https://arxiv.org/abs/1907.11692) base Model on TVR text with Masked Language Model (MLM) objective for 1 epoch:
```
bash utils/text_feature/train_lm_finetuning_single_sentence.sh FINETUNE_MODE OUTPUT_ROOT
```
`FINETUNE_MODE` could be `query_only` where only query text (in train set) is used to fine-tune the pre-trained model, 
this feature is used when we want to test model performance without subtitles. It can also be `sub_query` where 
both subtitle and query text are used in the fine-tuning process. `OUTPUT_ROOT` is a directory used to store the 
fine-tuned model and extracted features. You can append an additional `--debug` flag after the command to do 
a fast run of the code to test your configuration before actually running fine-tuning.

At fine-tuning, each query is treated as a single sequence, each subtitle is split into max-length=256 segments 
where each of the resulting segments wil be treated as a single sequence.

### Feature Extraction
After fine-tuning, you will get fine-tuned model at `OUTPUT_ROOT/FINETUNE_MODE/roberta-base_tuned_model`. 

Extract features at token-level:
```
bash utils/text_feature/extract_single_sentence_embeddings.sh \
OUTPUT_ROOT FINETUNE_MODE EXTRACTION_MODE SAVE_FILEPATH
```
`EXTRACTION_MODE` could be `sub` or `query`, 
`SAVE_FILEPATH` is a `.h5` filepath that will save the extracted features.

To get the tokens that correspond to these feature vectors, run
```
bash utils/text_feature/extract_single_sentence_tokens.sh \
OUTPUT_ROOT FINETUNE_MODE EXTRACTION_MODE SAVE_FILEPATH
```
`SAVE_FILEPATH` is a `.jsonl` filepath that stores the extracted tokens. 
This is useful if you want to visualize attentions from the attended feature vectors back to the word tokens.

The extracted query features can be directly used for training our XML model,
while subtitle features needs one additional step: convert token-level features to clip-level features. 
Specifically, we max-pool/avg-pool the subtitle token embeddings every 1.5 seconds to get the clip-level 
embeddings:
```
bash utils/text_feature/convert_sub_feature_word_to_clip.sh \
POOL_TYPE CLIP_LENGTH SUB_TOKEN_H5 SUB_CLIP_H5 VID_CLIP_H5  
```
`POOL_TYPE` could be `max` or `avg`, which defines how to aggregate token-level features to clip-level features.
`CLIP_LENGTH` is set to 1.5 (seconds). `SUB_TOKEN_H5` is the path to extracted subtitle token-level features.
`SUB_CLIP_H5` is the path to save the aggregated subtitle clip-level features. 
`VID_CLIP_H5` is the path to extracted video clip-level features, 
which is used to make sure each subtitle's clip-level features 
has the same length as the its corresponding video clip-level features.


