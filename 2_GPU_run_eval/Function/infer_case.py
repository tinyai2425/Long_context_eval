"""读取 GPU jsonl（Chat Completions + messages），打 vLLM。

只把 GPU_OPENAI_KEYS + extra_body 传给 openai 客户端；评分元数据不会进请求。
"""

import copy
import json
import time

GPU_OPENAI_KEYS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "presence_penalty",
    "seed",
    "max_tokens",
)

verify_ans = None
extend_metrics = None


def load_jsonl(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _prompt_text_from_messages(messages):
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


def enable_thinking_of(case):
    extra = case.get("extra_body") or {}
    kwargs = extra.get("chat_template_kwargs") or {}
    return bool(kwargs.get("enable_thinking", False))


def openai_create_kwargs(case, model_id, messages):
    kwargs = {"model": model_id}
    for key in GPU_OPENAI_KEYS:
        if key == "model":
            continue
        if key == "messages":
            kwargs["messages"] = messages
            continue
        if key in case:
            kwargs[key] = case[key]
    extra = case.get("extra_body")
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def measure_performance(client, model_id, case, timeout=1200):
    messages = copy.deepcopy(case.get("messages") or [])
    prompt_text = _prompt_text_from_messages(messages)
    create_kwargs = openai_create_kwargs(case, model_id, messages)
    create_kwargs["timeout"] = timeout

    start_time = time.time()
    first_token_time = None
    full_response = ""
    reasoning_content = ""
    usage = None

    response = client.chat.completions.create(**create_kwargs)
    for chunk in response:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        piece = getattr(delta, "content", None)
        if piece:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            full_response += piece
        reason_piece = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reason_piece:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            reasoning_content += reason_piece

    total_time = time.time() - start_time
    decode_time = (
        total_time - first_token_time if first_token_time is not None else None
    )

    if usage is not None:
        prompt_token_len = int(getattr(usage, "prompt_tokens", 0) or 0)
        response_token_len = int(getattr(usage, "completion_tokens", 0) or 0)
    else:
        prompt_token_len = int(case.get("prompt_token_len") or len(prompt_text))
        response_token_len = len(full_response)

    score_text = full_response or reasoning_content
    attached = verify_ans.attach_task_fields(dict(case))
    passed, metric = verify_ans.verify_case(attached, score_text)
    _, _, prediction = verify_ans.score_case(
        attached.get("secondary_task") or "",
        attached.get("expect"),
        score_text,
        attached.get("language") or "",
    )
    project_name, flavor, vertical = verify_ans.parse_test_case_name(
        attached.get("testCaseName", "")
    )

    row = {
        "testCaseName": attached.get("testCaseName", ""),
        "project_name": project_name,
        "flavor": flavor,
        "vertical": vertical,
        "index": attached.get("index"),
        "sample_id": attached.get("sample_id", ""),
        "benchmark": attached.get("benchmark", "LongBench-Pro"),
        "language": attached.get("language", ""),
        "token_length": attached.get("token_length", ""),
        "primary_task": attached.get("primary_task", ""),
        "secondary_task": attached.get("secondary_task", ""),
        "contextual_requirement": attached.get("contextual_requirement", ""),
        "difficulty": attached.get("difficulty", ""),
        "metric_name": attached.get("metric_name")
        or verify_ans.metric_name_of(attached.get("secondary_task") or ""),
        "expect": attached.get("expect")
        if isinstance(attached.get("expect"), str)
        else json.dumps(attached.get("expect"), ensure_ascii=False),
        "response": full_response,
        "reasoning_content": reasoning_content,
        "prediction": prediction,
        "n_pred_lines": len(
            [ln for ln in (prediction or "").splitlines() if ln.strip()]
        ),
        "prompt_token_len": prompt_token_len,
        "response_token_len": response_token_len,
        "first_token_time": first_token_time,
        "total_time": total_time,
        "decode_time": decode_time,
        "metric": float(metric),
        "correct": bool(passed),
        "get_ans": extend_metrics.GET_answer(score_text),
        "repeat": extend_metrics.calculate_repetition_rate(score_text),
        "entropy": extend_metrics.calculate_char_entropy(score_text),
        "enable_thinking": enable_thinking_of(case),
    }
    return verify_ans.attach_task_fields(row)
