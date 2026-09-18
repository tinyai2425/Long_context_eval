# 用法:
#   python Gen_longbench_pro_cases.py <model_config.json> <MAX_CASE_LENGTH> <MAX_CASES_PER_VERTICAL> \
#       [--enable_thinking true|false] [--dataset PATH]
#
# MAX_CASE_LENGTH: chat template 后的 prompt token 上限。<=0 表示不过滤。
#   例：KV cache 32K、max_tokens=5000 时配 27000。
# MAX_CASES_PER_VERTICAL: 每个 secondary_task 最多取 N 条。<=0 表示全收（通过长度过滤的）。
#
# 产出（../../model-eval-storage/{model_name}/project-N/）:
#   project-N-LBP-GPU-{n}.jsonl
#   project-N-LBP-API-{n}.jsonl
#   project-N-LBP-OMC-{n}.json
#   sampling_config.json
#   api_config.json

import json
import os
import shutil
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath("Function"))
import case_builder
import longbench_pro
import test_utils


USAGE = (
    "Usage: python Gen_longbench_pro_cases.py <model_config.json> "
    "<MAX_CASE_LENGTH> <MAX_CASES_PER_VERTICAL> "
    "[--enable_thinking true|false] [--dataset PATH]"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_DATASET = os.path.join(
    REPO_ROOT, "Data_set", "LongBench-Pro", "longbench_pro.json"
)
STORAGE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "model-eval-storage"))


def _flag_val(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


def _parse_bool(value, default):
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def get_unique_subdir(base_dir, prefix):
    counter = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}-{counter}")):
        counter += 1
    return f"{prefix}-{counter}"


def resolve_path(config_path, maybe_rel):
    if not maybe_rel:
        return maybe_rel
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(config_path)), maybe_rel))


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_jsonl(path, cases):
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            json.dump(jsonable(case), f, ensure_ascii=False)
            f.write("\n")
    print(f"{len(cases)} cases saved to {path}")


def write_api_config(project_path, mp, max_tokens, enable_thinking):
    tokenizer = test_utils._get_tokenizer()
    chat_template = getattr(tokenizer, "chat_template", "")
    license_text = ""
    license_file = os.path.join(mp.TOKENIZER_CONFIG_PATH, "LICENSE")
    if os.path.isfile(license_file):
        with open(license_file, "r", encoding="utf-8") as f:
            license_text = f.read()
    api_config = {
        "inferType": mp.INFER_TYPE,
        "tokenizerType": mp.TOKENIZER_TYPE,
        "tokenizerPath": mp.TOKENIZER_PATH,
        "modelType": mp.MODEL_TYPE,
        "modelPath": mp.MODEL_PATH,
        "weightDir": mp.WEIGHT_DIR,
        "prefixPrompt": mp.PREFIX_PROMPT,
        "pmtCacheOperation": mp.PMT_CACHE_OP,
        "pfxInitTokenLen": mp.PFX_INIT_TOKEN_LEN,
        "loraCfgPath": mp.LORA_CFG_PATH,
        "expect": "",
        "callbackFreq": mp.CALLBACK_FREQ,
        "sampleFlag": mp.SAMPLE_FLAG,
        "seed": mp.SEED,
        "topK": mp.TOPK,
        "topP": mp.TOPP,
        "temperature": mp.TEMPERATURE,
        "maxGenTokens": max_tokens,
        "repetitionPenalty": mp.REPETITIONPENALTY,
        "initTokenLen": mp.INIT_TOKEN_LEN,
        "stopSeq": mp.STOP_SEQ,
        "isAsync": mp.IS_ASYNC,
        "enableThinking": bool(enable_thinking),
        "chatTemplate": chat_template,
        "modelInfo": {"license": license_text},
    }
    path = os.path.join(project_path, "api_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(api_config, f, indent=4, ensure_ascii=False)
    print(f"api_config.json saved to {path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(USAGE)
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        max_case_length = int(sys.argv[2])
        max_per_vertical = int(sys.argv[3])
    except ValueError as exc:
        raise ValueError("MAX_CASE_LENGTH 和 MAX_CASES_PER_VERTICAL 必须是整数") from exc

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    enable_thinking = _parse_bool(
        _flag_val("--enable_thinking"),
        bool(config.get("enable_thinking", config.get("ENABLE_THINKING", False))),
    )
    dataset_path = _flag_val("--dataset") or DEFAULT_DATASET
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"找不到数据集: {dataset_path}")

    tokenizer_path = resolve_path(config_path, config.get("TOKENIZER_CONFIG_PATH"))
    if not tokenizer_path or not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(
            f"找不到 tokenizer: {tokenizer_path}。"
            "请检查 model_config.TOKENIZER_CONFIG_PATH。"
        )
    config["TOKENIZER_CONFIG_PATH"] = tokenizer_path
    config["ENABLE_THINKING"] = enable_thinking

    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("model_config 缺少 model_name")

    mp = SimpleNamespace(**config)
    test_utils.set_model_config(mp)

    project_base = os.path.join(STORAGE_ROOT, model_name)
    os.makedirs(project_base, exist_ok=True)
    project_name = get_unique_subdir(project_base, "project")
    project_path = os.path.join(project_base, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"PROJECT_NAME {project_name}")
    print(f"Full path to project dir: {project_path}")
    print(f"[INFO] enable_thinking={enable_thinking}")
    print(f"[INFO] MAX_CASE_LENGTH={max_case_length}")
    print(f"[INFO] MAX_CASES_PER_VERTICAL={max_per_vertical}")
    print(f"[INFO] dataset={dataset_path}")
    print(f"[INFO] tokenizer={tokenizer_path}")

    print("[INFO] loading LongBench-Pro ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] dataset size={len(dataset)}")

    sampled = case_builder.read_sampling(config, enable_thinking)
    selected, stats = longbench_pro.select_items(
        dataset, max_case_length, max_per_vertical, enable_thinking
    )
    print(
        f"[INFO] kept={stats['kept']} "
        f"skipped_bucket={stats['skipped_bucket']} "
        f"skipped_token={stats['skipped_token']}"
    )
    for vertical, n in sorted(stats["verticals"].items()):
        print(f"  {vertical}: {n}")

    if not selected:
        raise RuntimeError("过滤后没有用例。请加大 MAX_CASE_LENGTH 或检查数据集。")

    token_lens = [n for _, _, n in selected]
    print(
        f"[INFO] kept prompt tokens min={min(token_lens)} "
        f"max={max(token_lens)} mean={sum(token_lens) / len(token_lens):.1f}"
    )

    gpu_cases, api_cases, omc_cases = longbench_pro.generate_cases(
        project_name, selected, sampled, enable_thinking
    )
    n = len(gpu_cases)

    write_jsonl(os.path.join(project_path, f"{project_name}-LBP-GPU-{n}.jsonl"), gpu_cases)
    write_jsonl(os.path.join(project_path, f"{project_name}-LBP-API-{n}.jsonl"), api_cases)

    omc_path = os.path.join(project_path, f"{project_name}-LBP-OMC-{n}.json")
    with open(omc_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(omc_cases), f, indent=4, ensure_ascii=False)
    print(f"{len(omc_cases)} OMC test cases saved to {omc_path}")

    archived = dict(config)
    archived["enable_thinking"] = enable_thinking
    archived["ENABLE_THINKING"] = enable_thinking
    archived["MAX_CASE_LENGTH"] = max_case_length
    archived["MAX_CASES_PER_VERTICAL"] = max_per_vertical
    sampling_path = os.path.join(project_path, "sampling_config.json")
    with open(sampling_path, "w", encoding="utf-8") as f:
        json.dump(archived, f, indent=2, ensure_ascii=False)
    shutil.copy2(config_path, os.path.join(project_path, os.path.basename(config_path)))
    print(f"sampling_config.json saved to {sampling_path}")

    max_tokens = int(sampled.get("max_tokens", config.get("max_tokens", 5000)))
    write_api_config(project_path, mp, max_tokens, enable_thinking)
    print("[ALL DONE]")
