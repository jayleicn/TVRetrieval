#!/usr/bin/env bash
model=$1
ctx_batch_size=$2
save_dir=baselines/profiling/cache

python baselines/profiling/profile_main.py \
--model ${model} \
--ctx_batch_size ${ctx_batch_size} \
--query_batch_size 100 \
--save_dir ${save_dir}

