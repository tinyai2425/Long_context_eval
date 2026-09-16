import json
import os


def load_eval_meta(project_dir):
    path = os.path.join(project_dir, "eval_meta.jsonl")
    mapping = {}
    if not os.path.isfile(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row.get("testCaseName")
            if name:
                mapping[name] = row
    return mapping


def merge_meta(case, meta_map):
    meta = meta_map.get(case.get("testCaseName") or "") or {}
    for key in (
        "sample_id",
        "language",
        "token_length",
        "primary_task",
        "secondary_task",
        "contextual_requirement",
        "difficulty",
        "metric_name",
        "index",
        "expect",
    ):
        if key not in case or case.get(key) in (None, ""):
            if key in meta:
                case[key] = meta[key]
        elif key == "expect" and not case.get("expect") and meta.get("expect") is not None:
            case[key] = meta[key]
    return case
