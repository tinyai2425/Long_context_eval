"""Parse HarmonyOS curl logs: request JSON / response / API_total_time.

请求行是 API jsonl（messages 为 role/user）。响应可能是完整 JSON 或压成一行的 SSE。
"""

import json
import math
from typing import Dict, List, Union

import pandas as pd

GET_answer = None
calculate_repetition_rate = None
calculate_token_entropy = None
score_case = None
parse_test_case_name = None
attach_task_fields = None
iter_lines_safely = None


def _messages_text(messages):
    parts = []
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text") or "")
    return "\n".join(parts)


def _try_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _iter_sse_payloads(text):
    marker = "data:"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        idx += len(marker)
        nxt = text.find(marker, idx)
        payload = text[idx:] if nxt < 0 else text[idx:nxt]
        payload = payload.strip()
        start = nxt if nxt >= 0 else len(text)
        if not payload or payload == "[DONE]":
            continue
        obj = _try_json_object(payload)
        if obj is not None:
            yield obj
            continue
        brace_s = payload.find("{")
        brace_e = payload.rfind("}")
        if brace_s >= 0 and brace_e > brace_s:
            obj = _try_json_object(payload[brace_s:brace_e + 1])
            if obj is not None:
                yield obj


def _collapse_chunks(chunks):
    if not chunks:
        return {
            "object": "error",
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        }
    if len(chunks) == 1 and chunks[0].get("object") != "chat.completion.chunk":
        return chunks[0]

    content_parts = []
    reasoning_parts = []
    usage = None
    model = None
    finish_reason = None
    for chunk in chunks:
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            content_parts.append(piece)
        reason = delta.get("reasoning_content") or delta.get("reasoning")
        if reason:
            reasoning_parts.append(reason)
        msg = choice.get("message") or {}
        if msg.get("content") and not piece:
            content_parts.append(msg["content"])

    result = {
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage:
        result["usage"] = usage
    return result


def coerce_response(raw):
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    obj = _try_json_object(text)
    if obj is not None:
        return obj
    if "data:" in text:
        return _collapse_chunks(list(_iter_sse_payloads(text)))
    return {
        "object": "error",
        "message": text[:4000],
        "choices": [{"message": {"role": "assistant", "content": ""}}],
    }


def _detect_format(resp):
    if not isinstance(resp, dict):
        return "unknown"
    if resp.get("object") in ("chat.completion", "chat.completion.chunk"):
        return "chat_completion"
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        delta = choices[0].get("delta") or {}
        if "content" in msg or "content" in delta:
            return "chat_completion"
    if isinstance(resp.get("message"), dict):
        return "ollama"
    return "unknown"


def _extract_response_fields(resp):
    fmt = _detect_format(resp)
    if fmt == "chat_completion":
        choices = resp.get("choices") or []
        response_text = ""
        reasoning = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            response_text = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
            if not response_text:
                delta = choices[0].get("delta") or {}
                response_text = delta.get("content", "") or ""
        usage = resp.get("usage") or {}
        return (
            response_text,
            reasoning,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
    if fmt == "ollama":
        msg = resp.get("message") or {}
        return (
            msg.get("content", "") or "",
            msg.get("reasoning_content", "") or "",
            resp.get("prompt_eval_count"),
            resp.get("eval_count"),
        )
    return "", "", None, None


def parse_llm_api_results(input_path: str) -> pd.DataFrame:
    results: List[Dict[str, Union[str, float, int]]] = []
    lines = [line.strip() for line in iter_lines_safely(input_path) if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"输入文件 {input_path} 行数 ({len(lines)}) 不是 3 的倍数，请检查格式。"
        )

    fmt_counts: Dict[str, int] = {}

    for i in range(0, len(lines), 3):
        req_line = json.loads(lines[i])
        resp_line = coerce_response(lines[i + 1])
        api_time_line = lines[i + 2]

        case: Dict[str, Union[str, float, int]] = {}
        case["testCaseName"] = req_line.get("testCaseName", "")
        project_name, flavor, vertical = ("", "", "")
        if parse_test_case_name:
            project_name, flavor, vertical = parse_test_case_name(case["testCaseName"])
        case["project_name"] = project_name
        case["flavor"] = flavor
        case["vertical"] = vertical
        case["benchmark"] = req_line.get("benchmark", "LongBench-Pro")
        case["index"] = req_line.get("index")
        case["sample_id"] = req_line.get("sample_id", "")
        case["language"] = req_line.get("language", "")
        case["token_length"] = req_line.get("token_length", "")
        case["primary_task"] = req_line.get("primary_task", "")
        case["secondary_task"] = req_line.get("secondary_task", "")
        case["contextual_requirement"] = req_line.get("contextual_requirement", "")
        case["difficulty"] = req_line.get("difficulty", "")
        case["metric_name"] = req_line.get("metric_name", "")
        case["expect"] = req_line.get("expect", "")

        fmt = _detect_format(resp_line)
        fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1
        response_text, reasoning, prompt_tok, completion_tok = _extract_response_fields(resp_line)
        case["resp_format"] = fmt
        case["response"] = response_text
        case["reasoning_content"] = reasoning

        prompt_text = _messages_text(req_line.get("messages"))
        if prompt_tok is not None:
            case["prompt_token_len"] = int(prompt_tok)
        else:
            case["prompt_token_len"] = int(req_line.get("prompt_token_len") or len(prompt_text))
        if completion_tok is not None:
            case["response_token_len"] = int(completion_tok)
        else:
            case["response_token_len"] = len(response_text)

        api_total_time = float(str(api_time_line).replace("API_total_time:", "").strip())
        case["first_token_time"] = math.nan
        case["decode_time"] = api_total_time
        case["total_time"] = api_total_time

        if attach_task_fields:
            attach_task_fields(case)

        score_text = response_text or reasoning
        case["get_ans"] = GET_answer(score_text)
        case["repeat"] = calculate_repetition_rate(score_text)
        case["entropy"] = calculate_token_entropy(score_text)
        metric, passed, prediction = score_case(
            case.get("secondary_task") or "",
            case.get("expect"),
            score_text,
            case.get("language") or "",
        )
        case["metric"] = float(metric)
        case["correct"] = bool(passed)
        case["prediction"] = prediction
        case["n_pred_lines"] = len(
            [ln for ln in (prediction or "").splitlines() if ln.strip()]
        )
        case["enable_thinking"] = req_line.get("enable_thinking")
        if isinstance(case.get("expect"), (list, dict)):
            case["expect"] = json.dumps(case["expect"], ensure_ascii=False)
        results.append(case)

    print(f"{len(results)} tests processed in results {input_path}")
    if fmt_counts:
        fmt_summary = ", ".join(f"{k}={v}" for k, v in fmt_counts.items())
        print(f"[INFO] response formats detected: {fmt_summary}")
    return pd.DataFrame(results)
