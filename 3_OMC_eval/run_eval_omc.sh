#!/bin/bash

TOKENIZER_PATH="../AIPC_LLM_eval_supplyment/Model_file/Qwen3.5-4B"
TEST_RESULT_PATH="../../model-eval-storage/Qwen3.5-4B/project-1/project-1-LBP-OMC-xxx.txt"
SHOW_DETAIL=true
EMBEDDING_MODEL=""

CMD="python Eval_OMC_results.py \"$TOKENIZER_PATH\" \"$TEST_RESULT_PATH\""
if [ "$SHOW_DETAIL" = true ]; then
    CMD="$CMD --show_detail"
fi
if [ -n "$EMBEDDING_MODEL" ]; then
    CMD="$CMD --embedding_model \"$EMBEDDING_MODEL\""
fi

eval $CMD
