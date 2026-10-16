#!/bin/bash

INPUT_JSONL="../../model-eval-storage/Qwen3.5-4B/project-1/project-1-LBP-GPU-xxx.jsonl"
VLLM_IP="127.0.0.1"
VLLM_PORT=8895
VLLM_MODEL_ID="qwen35_4b"
# 一次进程里连跑几轮。再启动一次会续写进当天同一个 GPU-LBP-日期 目录，
# 已有 GPU-Results-1.parquet 时下一轮是 2，最后对全部轮次求平均。
# 第 N 轮的 seed = jsonl 里的 seed + (N-1)，避免多轮打出同一份结果。
VERSION_FLAG=1
MAX_WORKERS=2
SHOW_DETAIL="true"
# Summary 语义相似度可选。不填则 T4 只用 Rouge-L。
EMBEDDING_MODEL=""

CMD="python Eval_GPU_results.py \"$INPUT_JSONL\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\" \"$VERSION_FLAG\" --max_workers \"$MAX_WORKERS\""
if [ "$SHOW_DETAIL" = "true" ]; then
    CMD="$CMD --show_detail"
fi
if [ -n "$EMBEDDING_MODEL" ]; then
    CMD="$CMD --embedding_model \"$EMBEDDING_MODEL\""
fi

eval $CMD
