#!/bin/bash
# 生成 LongBench-Pro 的 GPU / API / OMC 三套用例（同一批题）。
# MAX_CASE_LENGTH: chat template 后 prompt token 上限。
#   KV cache 32K、max_tokens=5000 时配 27000，更长的题直接丢掉。
# MAX_CASES_PER_VERTICAL: 每个 secondary_task 最多 N 条；0 = 长度过滤后全收。

CONFIG_PATH="../model_config/Qwen3.5-4B-test-config.json"
# CONFIG_PATH="../model_config/Qwen3.5-9B-test-config.json"
MAX_CASE_LENGTH=27000
MAX_CASES_PER_VERTICAL=1000
ENABLE_THINKING=false
DATASET=""

CMD="python Gen_longbench_pro_cases.py \"$CONFIG_PATH\" \"$MAX_CASE_LENGTH\" \"$MAX_CASES_PER_VERTICAL\" --enable_thinking \"$ENABLE_THINKING\""
if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset \"$DATASET\""
fi

eval $CMD
