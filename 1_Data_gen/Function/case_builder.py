"""把 model_config 映射成 GPU / API / OMC 三条请求。同一条样本的 messages 文本一致。

GPU / vLLM（openai.ChatCompletions.create）
  顶层: model, messages, stream, stream_options, temperature, top_p,
        presence_penalty, seed, max_tokens
  extra_body: top_k, min_p, repetition_penalty,
              chat_template_kwargs.enable_thinking

鸿蒙 OpenAI-compatible API（全部顶层）
  model, messages, stream, seed, top_p, temperature,
  presence_penalty, enable_thinking, max_tokens

OMC：顶层与 Ceval 一致；sentences[0].prompt 是 apply_chat_template 后的单串。
"""

import json
import re

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

GPU_EXTRA_BODY_KEYS = (
    "top_k",
    "min_p",
    "repetition_penalty",
)

API_OPENAI_KEYS = (
    "model",
    "messages",
    "stream",
    "seed",
    "top_p",
    "temperature",
    "presence_penalty",
    "enable_thinking",
    "max_tokens",
)


def sanitize_token(value):
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value))
    return text.strip("_") or "unknown"


def to_native(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def expect_list(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    try:
        return [str(x) for x in list(raw) if x is not None]
    except TypeError:
        return [str(raw)]


def _copy_present(dst, src, keys):
    for key in keys:
        if key in src and src[key] is not None:
            dst[key] = src[key]


def read_sampling(config, enable_thinking):
    sampling = config.get("sampling") or {}
    sampled = {
        "model_name": config["model_name"],
        "enable_thinking": bool(enable_thinking),
    }
    if "max_tokens" in config:
        sampled["max_tokens"] = int(config["max_tokens"])
    if "seed" in config:
        sampled["seed"] = int(config["seed"])
    for key, value in sampling.items():
        if key.startswith("_"):
            continue
        sampled[key] = value
    return sampled


def build_messages(prompt):
    return [{"role": "user", "content": prompt}]


def case_labels(test_case_name, expect, extra_meta):
    """Inline scoring labels into GPU/API cases. No sidecar file."""
    meta = {
        "testCaseName": test_case_name,
        "expect": expect,
        "benchmark": "LongBench-Pro",
    }
    meta.update(extra_meta)
    return meta


def build_gpu_case(sampled, prompt, test_case_name, expect, extra_meta):
    case = {
        "model": sampled["model_name"],
        "messages": build_messages(prompt),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _copy_present(
        case, sampled, ("temperature", "top_p", "presence_penalty", "seed", "max_tokens")
    )
    extra = {}
    _copy_present(extra, sampled, GPU_EXTRA_BODY_KEYS)
    extra["chat_template_kwargs"] = {"enable_thinking": sampled["enable_thinking"]}
    case["extra_body"] = extra
    case.update(case_labels(test_case_name, expect, extra_meta))
    return case


def build_api_case(sampled, prompt, test_case_name, expect, extra_meta):
    case = {
        "model": sampled["model_name"],
        "messages": build_messages(prompt),
        "stream": True,
        "enable_thinking": sampled["enable_thinking"],
    }
    _copy_present(
        case,
        sampled,
        ("seed", "top_p", "temperature", "presence_penalty", "max_tokens"),
    )
    case.update(case_labels(test_case_name, expect, extra_meta))
    return case


def dump_expect(expect):
    if isinstance(expect, (list, dict)):
        return json.dumps(expect, ensure_ascii=False)
    return str(expect)
