#!/usr/bin/env bash
# Usage:
# bash utils/text_feature/extract_single_sentence_tokens.sh \
# OUTPUT_ROOT FINETUNE_MODE EXTRACTION_MODE SAVE_FILEPATH
# Examples:
# bash utils/text_feature/extract_single_sentence_tokens.sh ${output_root} query query tvr_query_roberta_tokenized.jsonl --debug
output_root=$1
finetune_mode=$2  # sub_query or query_only
extraction_mode=$3  # sub or query
extracted_file_name=$4  # "*jsonl" file



data_root="data"
train_data_file="${data_root}/tvr_train_release.jsonl"
val_data_file="${data_root}/tvr_val_release.jsonl"
test_data_file1="${data_root}/tvr_test_public_release.jsonl"
test_data_file2="${data_root}/tvr_test_challenge_release.jsonl"
sub_data_file="${data_root}/tvqa_preprocessed_subtitles.jsonl"

output_dir="${output_root}/${finetune_mode}"
model_type="roberta"
model_name_or_path="${output_dir}/roberta-base_tuned_model"


if [[ ${extraction_mode} == query ]]; then
    max_length=30
    extra_args=(--train_data_file)
    extra_args+=(${train_data_file})
    extra_args+=(${val_data_file})
    extra_args+=(${test_data_file1})
    extra_args+=(${test_data_file2})
#elif [[ ${extraction_mode} == sub ]]; then
#    max_length=256
#    extra_args=(--use_sub)
#    extra_args+=(--sub_data_file)
#    extra_args+=(${sub_data_file})
fi

python utils/text_feature/lm_finetuning_on_single_sentences.py \
--output_dir ${output_dir} \
--model_type ${model_type} \
--model_name_or_path ${model_name_or_path} \
--do_tokenize \
--extracted_file_name ${extracted_file_name} \
${extra_args[@]} \
${@:5}
