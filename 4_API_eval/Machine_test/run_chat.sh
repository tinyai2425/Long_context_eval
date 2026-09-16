#!/bin/sh
# HarmonyOS / 端侧：只依赖 sh + curl，不要 python。
#
# Usage:
#   sh run_chat.sh <input.jsonl> <output.txt>
#
# jsonl 每行已是完整的 /v1/chat/completions 请求（messages 含 role=user）。
# 长上下文请用 curl -d @file，避免撑爆命令行。
#
# 环境变量:
#   API_URL   默认 http://localhost:11434/v1/chat/completions

if [ "$#" -ne 2 ]; then
    echo "Usage: sh run_chat.sh <input.jsonl> <output.txt>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
API_URL="${API_URL:-http://localhost:11434/v1/chat/completions}"

echo "$INPUT_FILE" | grep -qE '\.jsonl$' || { echo "Input must end with .jsonl"; exit 1; }
echo "$OUTPUT_FILE" | grep -qE '\.txt$'   || { echo "Output must end with .txt";  exit 1; }
[ -f "$INPUT_FILE" ] || { echo "Input file not found: $INPUT_FILE"; exit 1; }

TMP_REQ="tmp_req.json"
TMP_RESP="tmp_response.raw"

: > "$OUTPUT_FILE"
total=`wc -l < "$INPUT_FILE" | tr -d ' '`
i=1

while [ "$i" -le "$total" ]; do
    sed -n "${i}p" "$INPUT_FILE" > "$TMP_REQ"
    if [ ! -s "$TMP_REQ" ]; then
        i=$((i + 1))
        continue
    fi

    echo ">> Sending request $i / $total to $API_URL ..."

    response_time=$(curl -sS -w "%{time_total}" -o "$TMP_RESP" \
        -H "Content-Type: application/json" \
        -X POST "$API_URL" \
        -d @"$TMP_REQ")

    cat "$TMP_REQ" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"
    tr -d '\r\n' < "$TMP_RESP" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"
    printf "API_total_time: %s\n" "$response_time" >> "$OUTPUT_FILE"

    i=$((i + 1))
done

rm -f "$TMP_REQ" "$TMP_RESP"
echo "All requests completed. Responses saved to: $OUTPUT_FILE"
