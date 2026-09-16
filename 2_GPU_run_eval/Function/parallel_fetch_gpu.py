import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd

import infer_case
import verify_ans


def _error_result(case, exc):
    name = case.get("testCaseName", "")
    project_name, flavor, vertical = verify_ans.parse_test_case_name(name)
    return {
        "testCaseName": name,
        "project_name": project_name,
        "flavor": flavor,
        "vertical": vertical,
        "index": case.get("index"),
        "sample_id": case.get("sample_id", ""),
        "benchmark": case.get("benchmark", "LongBench-Pro"),
        "language": case.get("language", ""),
        "token_length": case.get("token_length", ""),
        "primary_task": case.get("primary_task", ""),
        "secondary_task": case.get("secondary_task", ""),
        "contextual_requirement": case.get("contextual_requirement", ""),
        "difficulty": case.get("difficulty", ""),
        "metric_name": case.get("metric_name", ""),
        "expect": json.dumps(case.get("expect"), ensure_ascii=False)
        if not isinstance(case.get("expect"), str)
        else case.get("expect", ""),
        "response": "",
        "reasoning_content": "",
        "prediction": json.dumps({"error": str(exc)}, ensure_ascii=False),
        "prompt_token_len": int(case.get("prompt_token_len") or 0),
        "response_token_len": 0,
        "first_token_time": None,
        "total_time": None,
        "decode_time": None,
        "metric": 0.0,
        "correct": False,
        "get_ans": False,
        "repeat": 0.0,
        "entropy": 0.0,
        "enable_thinking": False,
    }


def parallel_fetch_reference_model(
    jsonl_path,
    server_ip,
    server_port,
    model_id,
    max_workers=2,
    timeout=1200,
):
    client = openai.OpenAI(
        base_url=f"http://{server_ip}:{server_port}/v1",
        api_key="no-api-key-needed",
        timeout=timeout,
    )
    cases = infer_case.load_jsonl(jsonl_path)
    total = len(cases)
    if total == 0:
        raise ValueError(f"No samples found in {jsonl_path}")

    max_workers = max(1, int(max_workers))
    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {}
        for seq, case in enumerate(cases):
            future = executor.submit(
                infer_case.measure_performance,
                client,
                model_id,
                case,
                timeout,
            )
            future_to_seq[future] = seq

        for done_i, future in enumerate(as_completed(future_to_seq)):
            seq = future_to_seq[future]
            case = cases[seq]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\n[ERROR] sample {seq} {case.get('testCaseName')} failed: {exc}")
                result = _error_result(case, exc)
            result["_seq"] = seq
            results.append(result)
            ttft = result.get("first_token_time") or 0.0
            ttc = result.get("total_time") or 0.0
            avg_elapsed = (time.time() - start_time) / (done_i + 1)
            remaining = avg_elapsed * (total - done_i - 1)
            print(
                f"\r{done_i + 1}/{total} samples processed "
                f"(TTFT:{ttft:.2f}s/TTC:{ttc:.2f}s), "
                f"{remaining:.2f}s remaining ...",
                end="",
                flush=True,
            )

    results.sort(key=lambda r: r.get("_seq", 0))
    for row in results:
        row.pop("_seq", None)
    df_results = pd.DataFrame(results)
    print(
        f"\r{len(results)} samples processed "
        f"({time.time() - start_time:.2f}s spent) in {jsonl_path}"
    )
    return df_results
