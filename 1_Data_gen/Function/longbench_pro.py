# 从 LongBench-Pro 生成同一批 GPU / API / OMC 用例。
# prompt = context + "\\n\\n\\n\\n" + question_{thinking|nonthinking}（官方 inference.py）

from collections import defaultdict

import case_builder
import test_utils

BUCKET_TO_TOKENS = {
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
}

TASK_METRIC_CONFIG = {
    "T1.1 Global Cohesive Retrieval": "NDCG",
    "T1.2 Key-Snippet Retrieval": "NDCG",
    "T2.1 Global Timeline Reconstruction": "Pairwise_Accuracy",
    "T2.2 Local Causal Chain Sorting": "Pairwise_Accuracy",
    "T3.1 Multi-Doc Integration QA": "Accuracy",
    "T3.2 Single-Hop Fact QA": "Accuracy",
    "T4.1 Global-Coverage Constrained Summary": "Summary",
    "T4.2 Query-Focused Summary": "Summary",
    "T5.1 Full-Sentence Citation Alignment": "F1_Score",
    "T5.2 Key-Statement Citation Alignment": "F1_Score",
    "T6.1 Large-Scale Document Clustering": "SubEM",
    "T6.2 Targeted Subset Cluster Identification": "F1_Score",
    "T6.3 Global Frequency Analysis": "Pairwise_Accuracy",
    "T7.1 Global Conflict & Inconsistency Localization": "F1_Score",
    "T7.2 Targeted Rule or Condition Violation Detection": "F1_Score",
    "T7.3 Comprehensive Error & Anomaly Sweep": "F1_Score",
    "T8.1 Structured Multi-Source Consistency Verification": "SubEM",
    "T8.2 Single-Source Targeted Aggregation": "SubEM",
    "T8.3 Long-Context Procedural State Tracking": "SubEM",
    "T9.1 Dependency-Aware Multi-Version Impact Analysis": "F1_Score",
    "T9.2 Localized Interface Change Detection": "F1_Score",
    "T10.1 Large-Scale In-Context Rule Induction": "SubEM",
    "T10.2 Targeted Example-Based Rule Induction": "SubEM",
    "T11.1 Long-Range Entity & Commitment Tracking": "Accuracy",
    "T11.2 Short-Range Reference Resolution & State Query": "Accuracy",
}


def bucket_tokens(token_length):
    key = str(token_length).strip().lower()
    if key in BUCKET_TO_TOKENS:
        return BUCKET_TO_TOKENS[key]
    digits = "".join(ch for ch in key if ch.isdigit())
    if digits:
        return int(digits) * 1024
    return None


def build_user_prompt(item, enable_thinking):
    context = item.get("context") or ""
    if enable_thinking:
        question = item.get("question_thinking") or item.get("question_nonthinking") or ""
    else:
        question = item.get("question_nonthinking") or item.get("question_thinking") or ""
    return f"{context}\n\n\n\n{question}"


def extra_meta(item, prompt_token_len, metric_name):
    return {
        "sample_id": item.get("id", ""),
        "language": item.get("language", ""),
        "token_length": item.get("token_length", ""),
        "primary_task": item.get("primary_task", ""),
        "secondary_task": item.get("secondary_task", ""),
        "contextual_requirement": item.get("contextual_requirement", ""),
        "difficulty": item.get("difficulty", ""),
        "metric_name": metric_name,
        "prompt_token_len": prompt_token_len,
    }


def select_items(dataset, max_case_length, max_per_vertical, enable_thinking):
    """先按官方 token_length 桶粗筛，再按 chat template 后的真实 token 数过滤。

    max_case_length <= 0 表示不过滤长度。
    max_per_vertical <= 0 表示每个 secondary_task 全收。
    同一 vertical 内按数据集原顺序取前 N 条（无随机）。
    """
    grouped = defaultdict(list)
    skipped_bucket = 0
    skipped_token = 0

    for item in dataset:
        secondary = item.get("secondary_task") or "unknown"
        labeled = bucket_tokens(item.get("token_length"))
        if max_case_length > 0 and labeled is not None and labeled > max_case_length:
            skipped_bucket += 1
            continue

        prompt = build_user_prompt(item, enable_thinking)
        messages = case_builder.build_messages(prompt)
        n_tokens = test_utils.count_chat_tokens(messages, enable_thinking=enable_thinking)
        if max_case_length > 0 and n_tokens > max_case_length:
            skipped_token += 1
            continue

        grouped[secondary].append((item, prompt, n_tokens))

    selected = []
    for secondary in sorted(grouped.keys()):
        rows = grouped[secondary]
        if max_per_vertical > 0:
            rows = rows[:max_per_vertical]
        selected.extend(rows)

    stats = {
        "skipped_bucket": skipped_bucket,
        "skipped_token": skipped_token,
        "kept": len(selected),
        "verticals": {k: min(len(v), max_per_vertical) if max_per_vertical > 0 else len(v)
                      for k, v in grouped.items()},
    }
    return selected, stats


def generate_cases(project_name, selected, sampled, enable_thinking):
    gpu_cases = []
    api_cases = []
    omc_cases = []
    meta_rows = []
    max_tokens = int(sampled.get("max_tokens", 5000))

    for idx, (item, prompt, n_tokens) in enumerate(selected):
        secondary = item.get("secondary_task") or "unknown"
        vertical = case_builder.sanitize_token(secondary)
        test_case_name = f"{project_name}-lbp-test-{vertical}-{idx}"
        expect = case_builder.expect_list(item.get("answer"))
        metric_name = TASK_METRIC_CONFIG.get(secondary, "Accuracy")
        extra = extra_meta(item, n_tokens, metric_name)
        extra["index"] = idx

        gpu_cases.append(
            case_builder.build_gpu_case(sampled, prompt, test_case_name, expect, extra)
        )
        api_cases.append(
            case_builder.build_api_case(sampled, prompt, test_case_name, expect, extra)
        )

        omc_prompt = test_utils.render_chat_template_messages(
            case_builder.build_messages(prompt), enable_thinking=enable_thinking
        )
        omc_cases.append(
            test_utils.create_omc_test(
                test_case_name,
                omc_prompt,
                expect=case_builder.dump_expect(expect),
                max_gen_tokens=max_tokens,
            )
        )

        meta_rows.append(
            {
                "testCaseName": test_case_name,
                "expect": expect,
                **extra,
            }
        )

    return gpu_cases, api_cases, omc_cases, meta_rows
