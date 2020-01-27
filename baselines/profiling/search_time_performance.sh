#!/usr/bin/env bash

mode=$1
#dt=$(date '%Y_%m_%d_%H_%M_%S');
#echo "$dt"
python baselines/profiling/search_time_performance.py \
--mode ${mode} \
--cache_dir baselines/profiling/cache

#| tee baselines/profiling/cache/${mode}_${dt}.log