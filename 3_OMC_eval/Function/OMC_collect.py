# 解析 OMC 日志 -> DataFrame，再用与 GPU/API 同一套 verify_ans 打分。

import json
import os
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
iter_lines_safely = None
merge_meta = None
load_eval_meta = None


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


def parse_llm_test_results(output_path: str) -> pd.DataFrame:
    case_start_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] ## testCaseName: ([a-zA-Z0-9_\-]+), start.*"
    )
    case_end_pattern = re.compile(
        r"\[(?:INFO|WARNING|ERROR|EXCEPTION)\] ## testCaseName: ([a-zA-Z0-9_\-]+), Done"
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

    meta_map = load_eval_meta(os.path.dirname(os.path.abspath(output_path))) if load_eval_meta else {}

    results: List[Dict[str, Union[str, int, float]]] = []
    case_data: Optional[dict] = None
    current_text_field: Optional[str] = None
    text_buffer: List[str] = []

    for line in iter_lines_safely(output_path):
        if case_data:
            if log_prefix_pattern.match(line):
                if current_text_field:
                    assert text_buffer
                    case_data[current_text_field] = "".join(text_buffer)
                    current_text_field = None
                    text_buffer = []

                if end_match := case_end_pattern.match(line):
                    case_name = end_match.group(1)
                    matched = re.match(TEST_CASE_NAME_PATTERN, case_name) if TEST_CASE_NAME_PATTERN else None
                    if matched:
                        case_data["project_name"] = matched.group("project_name")
                        case_data["flavor"] = matched.group("flavor")
                        case_data["vertical"] = matched.group("vertical")
                    assert case_data and case_data["testCaseName"] == case_name

                    case_data["decode_time"] = case_data["decodeTimeMs"] / 1000
                    case_data["first_token_time"] = case_data["prefillTimeMs"] / 1000
                    case_data["total_time"] = (
                        case_data["decode_time"] + case_data["first_token_time"]
                    )

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
                            tokenizer(case_data.get("prompt", ""))["input_ids"]
                        )
                    if "outputTokenCount" in case_data:
                        case_data["response_token_len"] = int(case_data["outputTokenCount"])
                    else:
                        case_data["response_token_len"] = len(tokenizer(response)["input_ids"])

                    if merge_meta:
                        merge_meta(case_data, meta_map)

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
                    case_data["benchmark"] = "LongBench-Pro"
                    if isinstance(case_data.get("expect"), (list, dict)):
                        case_data["expect"] = json.dumps(
                            case_data["expect"], ensure_ascii=False
                        )

                    results.append(case_data)
                    case_data = None
                elif time_match := time_pattern.match(line):
                    field, value, unit = time_match.groups()
                    case_data[field] = convert_to_ms(float(value), unit)
                elif num_match := numeric_pattern.match(line):
                    field, value = num_match.groups()
                    case_data[field] = float(value)
                elif text_match := text_field_start_pattern.match(line):
                    field, initial_content = text_match.groups()
                    assert not current_text_field and not text_buffer
                    current_text_field = field
                    text_buffer = [initial_content]
            elif current_text_field:
                assert text_buffer
                text_buffer.append(line)
        elif start_match := case_start_pattern.match(line):
            case_name = start_match.group(1)
            assert not case_data
            case_data = {"testCaseName": case_name}

    print(f"{len(results)} tests processed in results {output_path}")
    return pd.DataFrame(results)
