#!/bin/bash

INPUT_JSONL="../../model-eval-storage/Qwen3.5-4B/project-1/project-1-LBP-GPU-xxx.jsonl"
VLLM_IP="127.0.0.1"
VLLM_PORT=8895
VLLM_MODEL_ID="qwen35_4b"
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
