# 解析 OMC 日志 -> DataFrame，再用与 GPU/API 同一套 verify_ans 打分。

import json
import re
from typing import Dict, List, Optional, Union

import pandas as pd
from transformers import AutoTokenizer

TOKENIZER_CONFIG_PATH = None
tokenizer = None

GET_answer = None
calculate_repetition_rate = None
calculate_token_entropy = None
score_case = None
TEST_CASE_NAME_PATTERN = None
parse_test_case_name = None
attach_task_fields = None
iter_lines_safely = None

# Keep parquet aligned with GPU results. The OMC log contains the full
# long-context `prompt` plus engine paths; those must not be written out.
PARQUET_KEYS = (
    "testCaseName",
    "project_name",
    "flavor",
    "vertical",
    "index",
    "sample_id",
    "benchmark",
    "language",
    "token_length",
    "primary_task",
    "secondary_task",
    "contextual_requirement",
    "difficulty",
    "metric_name",
    "expect",
    "response",
    "prediction",
    "n_pred_lines",
    "prompt_token_len",
    "response_token_len",
    "first_token_time",
    "total_time",
    "decode_time",
    "metric",
    "correct",
    "get_ans",
    "repeat",
    "entropy",
)


def _slim_result(case_data):
    return {key: case_data.get(key) for key in PARQUET_KEYS}


def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def convert_to_ms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "us" or unit == "µs":
        return value / 1000
    if unit == "ns":
        return value / 1_000_000
    if unit == "s":
        return value * 1000
    return value


def sanitize_all_generation(all_gen, curr_gen=""):
    """去掉 OMC 写 all generation 时多打在全文末尾的 '.'。

    真实输出在 curr generation 里是正常的，例如最后一行是 ``8``；
    all generation 却变成 ``8.``。这个点不是模型 token，会让按行
    精确匹配的 NDCG / F1 / Accuracy / SubEM 判错。

    规则：若 all 相对 curr 只是末尾多一个 '.'，或 curr 不可用时
    全文末尾仍有那个点，则剥掉恰好一个。
    """
    if all_gen is None:
        all_gen = ""
    if not all_gen:
        return curr_gen or ""

    had_nl = all_gen.endswith("\n") or all_gen.endswith("\r\n")
    all_body = all_gen.rstrip("\r\n")
    curr_body = (curr_gen or "").rstrip("\r\n")

    if all_body.endswith("."):
        stripped = all_body[:-1]
        if curr_body:
            curr_last = curr_body.rsplit("\n", 1)[-1]
            all_last = all_body.rsplit("\n", 1)[-1]
            extra_dot = (
                all_body == curr_body + "."
                or all_last == curr_last + "."
            )
            if extra_dot:
                all_body = stripped
        else:
            all_body = stripped

    if had_nl:
        return all_body + "\n"
    return all_body


def _to_seconds(case_data, key):
    value = case_data.get(key)
    if value is None or value == "":
        return float("nan")
    return float(value) / 1000


def _finish_case(case_data, case_name):
    start_name = case_data.get("testCaseName") or case_name
    if case_name and start_name and case_name != start_name:
        print(
            f"[WARN] start/end testCaseName mismatch: {start_name} vs {case_name}; "
            "using start name"
        )
        case_name = start_name
    case_data["testCaseName"] = start_name

    if parse_test_case_name:
        project_name, flavor, vertical = parse_test_case_name(start_name)
        if project_name:
            case_data.setdefault("project_name", project_name)
        if flavor:
            case_data.setdefault("flavor", flavor)
        if vertical:
            case_data.setdefault("vertical", vertical)

    case_data["decode_time"] = _to_seconds(case_data, "decodeTimeMs")
    case_data["first_token_time"] = _to_seconds(case_data, "prefillTimeMs")
    case_data["total_time"] = case_data["decode_time"] + case_data["first_token_time"]

    raw_all = case_data.get("all generation", "") or ""
    response = sanitize_all_generation(
        raw_all, case_data.get("curr generation", "") or ""
    )
    if response != raw_all:
        case_data["all generation raw"] = raw_all
        case_data["all generation"] = response
    case_data["response"] = response

    if "inputTokenCount" in case_data:
        case_data["prompt_token_len"] = int(case_data["inputTokenCount"])
    else:
        case_data["prompt_token_len"] = len(
            tokenizer(case_data.get("prompt", "") or "")["input_ids"]
        )
    if "outputTokenCount" in case_data:
        case_data["response_token_len"] = int(case_data["outputTokenCount"])
    else:
        case_data["response_token_len"] = len(tokenizer(response)["input_ids"])

    if attach_task_fields:
        attach_task_fields(case_data)

    case_data["get_ans"] = GET_answer(response)
    case_data["repeat"] = calculate_repetition_rate(response)
    case_data["entropy"] = calculate_token_entropy(response)

    metric, passed, prediction = score_case(
        case_data.get("secondary_task") or "",
        case_data.get("expect"),
        response,
        case_data.get("language") or "",
    )
    case_data["metric"] = float(metric)
    case_data["correct"] = bool(passed)
    case_data["prediction"] = prediction
    case_data["n_pred_lines"] = len(
        [ln for ln in (prediction or "").splitlines() if ln.strip()]
    )
    case_data["benchmark"] = "LongBench-Pro"
    if isinstance(case_data.get("expect"), (list, dict)):
        case_data["expect"] = json.dumps(case_data["expect"], ensure_ascii=False)
    return _slim_result(case_data)


def parse_llm_test_results(output_path: str) -> pd.DataFrame:
    # Names may contain '.', which the old [A-Za-z0-9_-] charset dropped entirely.
    case_start_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] ## testCaseName: ([^\s,]+), start.*"
    )
    case_end_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] ## testCaseName: ([^\s,]+), Done"
    )
    log_prefix_pattern = re.compile(
        r"^\[(?:INFO|WARNING|ERROR|EXCEPTION|ERR|EXCEPT)\]"
    )
    numeric_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] *([a-zA-Z0-9_ \-]+) *(?::|, time =|=) *(\d+(?:\.\d*)?)\s*$"
    )
    time_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] *([a-z0-9A-Z_ \-]+) *: *(\d+(?:\.\d*)) *([mun]?s)\.?\s*$"
    )
    text_field_start_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] *([a-zA-Z0-9_ \-]+) *: *(.*\s)$"
    )

    results: List[Dict[str, Union[str, int, float]]] = []
    case_data: Optional[dict] = None
    current_text_field: Optional[str] = None
    text_buffer: List[str] = []

    def _close_text_field():
        nonlocal current_text_field, text_buffer
        if current_text_field:
            case_data[current_text_field] = "".join(text_buffer)
            current_text_field = None
            text_buffer = []

    for line in iter_lines_safely(output_path):
        if case_data:
            if log_prefix_pattern.match(line):
                _close_text_field()
                if start_match := case_start_pattern.match(line):
                    print(
                        f"[WARN] new testCaseName started before previous Done: "
                        f"{case_data.get('testCaseName')} -> {start_match.group(1)}"
                    )
                    results.append(
                        _finish_case(case_data, case_data.get("testCaseName"))
                    )
                    case_data = {"testCaseName": start_match.group(1)}
                elif end_match := case_end_pattern.match(line):
                    results.append(_finish_case(case_data, end_match.group(1)))
                    case_data = None
                elif time_match := time_pattern.match(line):
                    field, value, unit = time_match.groups()
                    case_data[field] = convert_to_ms(float(value), unit)
                elif num_match := numeric_pattern.match(line):
                    field, value = num_match.groups()
                    case_data[field] = float(value)
                elif text_match := text_field_start_pattern.match(line):
                    field, initial_content = text_match.groups()
                    current_text_field = field
                    text_buffer = [initial_content]
            elif current_text_field:
                text_buffer.append(line)
        elif start_match := case_start_pattern.match(line):
            case_data = {"testCaseName": start_match.group(1)}

    if case_data:
        _close_text_field()
        if case_data.get("all generation") or case_data.get("curr generation"):
            print(
                f"[WARN] last test case has no Done line; scoring "
                f"{case_data.get('testCaseName')}"
            )
            results.append(_finish_case(case_data, case_data.get("testCaseName")))
        else:
            print(f"[WARN] dropped unclosed test case {case_data.get('testCaseName')}")

    n_name = sum(1 for row in results if row.get("_task_source") == "name")
    n_payload = sum(1 for row in results if row.get("_task_source") == "payload")
    n_missing = sum(1 for row in results if row.get("_task_source") == "missing")
    n_no_expect = sum(1 for row in results if not str(row.get("expect") or "").strip())
    print(f"{len(results)} tests processed in results {output_path}")
    print(
        f"[INFO] task fields: payload={n_payload}, name={n_name}, missing={n_missing}; "
        f"empty expect={n_no_expect}"
    )
    if n_missing:
        print(
            "[WARN] some cases have no secondary_task; they default to Accuracy. "
            "Keep the original `*-lbp-test-{vertical}-{idx}` suffix in testCaseName."
        )
    return pd.DataFrame(results)
