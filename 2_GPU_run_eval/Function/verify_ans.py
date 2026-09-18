"""LongBench-Pro 官方指标（modules/utils.py），GPU / OMC / API 共用。

Summary 默认 Rouge-L；若 init_embedding(path) 成功则改为 0.5 * cosine + 0.5 * Rouge-L。
pass 口径与官方 pass@n 一致：Summary > 0.65，其余 metric == 1.0。
"""

import math
import os
import re
from itertools import combinations

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

PRIMARY_TASK_BY_CODE = {
    "T1": "T1. Retrieval & Ranking",
    "T2": "T2. Sequencing & Structure Reconstruction",
    "T3": "T3. Evidence-Grounded QA",
    "T4": "T4. Summarization & Synthesis",
    "T5": "T5. Attribution & Citation Alignment",
    "T6": "T6. Aggregation & Clustering",
    "T7": "T7. Consistency & Compliance Checking",
    "T8": "T8. Structured & Numeric Reasoning",
    "T9": "T9. Version & Code Diff Analysis",
    "T10": "T10. Rule Induction & In-Context Learning",
    "T11": "T11. Dialogue Memory & Long-Horizon Tracking",
}

# 生成侧是 `{project}-lbp-test-{vertical}-{8k|16k}-{idx}`。project 前缀允许任意字符
# （含点号、改过的序号），评分身份只用 flavor-test-vertical-idx。
TEST_CASE_NAME_PATTERN = re.compile(
    r"^(?P<project_name>.+)-(?P<flavor>[A-Za-z0-9_]+)-test-"
    r"(?P<vertical>.+)-(?P<index>\d+)$"
)
TEST_CASE_NAME_FALLBACK = re.compile(
    r"(?:^|-)test-(?P<vertical>.+)-(?P<index>\d+)$"
)

LENGTH_BUCKETS = ("8k", "16k", "32k", "64k", "128k", "256k")
LENGTH_BUCKET_ORDER = list(LENGTH_BUCKETS)
LENGTH_BUCKET_TOKENS = (8192, 16384, 32768, 65536, 131072, 262144)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
SUMMARY_PASS_THRESHOLD = 0.65

CJK_RANGES = "\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"
CJK_CHAR = re.compile("[" + CJK_RANGES + "]")
CJK_SPACE = re.compile(r"(?<=[" + CJK_RANGES + r"])\s+|\s+(?=[" + CJK_RANGES + r"])")
LETTER_LIST = re.compile(r"^[a-z](?:[\s/、,，|\\]+[a-z])+$")
LETTER_SEP = re.compile(r"[\s/、,，|\\]+")
CJK_LANG_RATIO = 0.05
LANG_SNIFF_CHARS = 4000

# 官方 NDCG@k / SubEM 忽略多余行，量化模型把 1..N 全打出来反而比短而准的答案高分。
# 默认按 k/n_pred 稀释；想拿与官方 leaderboard 可比的数就关掉。
OVERGEN_PENALTY = True

_embedding_model = None
_embedding_enabled = False


def sanitize_task_token(value):
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value or ""))
    return text.strip("_") or "unknown"


SLUG_TO_SECONDARY = {
    sanitize_task_token(name): name for name in TASK_METRIC_CONFIG
}


def init_embedding(path):
    global _embedding_model, _embedding_enabled
    if not path:
        return False
    if not os.path.isdir(path):
        print(f"[WARN] embedding model path not found: {path}; Summary uses Rouge-L only")
        return False
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[WARN] sentence-transformers not installed; Summary uses Rouge-L only")
        return False
    print(f"[INFO] loading embedding model from {path}")
    _embedding_model = SentenceTransformer(
        path, tokenizer_kwargs={"padding_side": "left"}
    )
    _embedding_enabled = True
    return True


def normalize_length_bucket(value):
    text = str(value or "").strip().lower().replace(" ", "")
    if text in LENGTH_BUCKETS:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        key = f"{digits}k"
        if key in LENGTH_BUCKETS:
            return key
    return ""


def split_length_from_vertical(vertical):
    text = str(vertical or "")
    lower = text.lower()
    for bucket in LENGTH_BUCKETS:
        suffix = "-" + bucket
        if lower.endswith(suffix):
            return text[: -len(suffix)], bucket
    return text, ""


def token_length_from_prompt_len(n_tokens):
    try:
        n = float(n_tokens)
    except (TypeError, ValueError):
        return ""
    if n != n or n <= 0:
        return ""
    sizes = LENGTH_BUCKET_TOKENS
    for i, size in enumerate(sizes):
        nxt = sizes[i + 1] if i + 1 < len(sizes) else None
        if nxt is None or n <= (size + nxt) / 2:
            return LENGTH_BUCKETS[i]
    return LENGTH_BUCKETS[-1]


def parse_test_case_parts(name):
    """Split `{project}-{flavor}-test-{vertical}-{8k|16k}-{index}`.

    Older names omit the length bucket. project 不参与评分。
    """
    text = str(name or "").strip()
    matched = TEST_CASE_NAME_PATTERN.match(text)
    if matched:
        vertical, length = split_length_from_vertical(matched.group("vertical"))
        return (
            matched.group("project_name"),
            matched.group("flavor"),
            vertical,
            int(matched.group("index")),
            length,
        )
    fallback = TEST_CASE_NAME_FALLBACK.search(text)
    if fallback:
        vertical, length = split_length_from_vertical(fallback.group("vertical"))
        return "", "", vertical, int(fallback.group("index")), length
    return "", "", "", None, ""


def parse_test_case_name(name):
    project_name, flavor, vertical, _index, _length = parse_test_case_parts(name)
    return project_name, flavor, vertical


def canonical_case_key(name):
    """Stable identity: flavor-test-vertical-idx, ignoring project prefix and length."""
    project_name, flavor, vertical, index, _length = parse_test_case_parts(name)
    if vertical and index is not None:
        return f"{(flavor or 'lbp').lower()}-test-{vertical}-{index}".lower()
    text = str(name or "").strip().lower()
    return text


def secondary_task_from_vertical(vertical):
    text = str(vertical or "").strip()
    if not text:
        return ""
    if text in TASK_METRIC_CONFIG:
        return text
    return SLUG_TO_SECONDARY.get(sanitize_task_token(text), "")


def primary_task_from_secondary(secondary_task):
    text = str(secondary_task or "").strip()
    if not text:
        return ""
    matched = re.match(r"(T\d+)", text)
    if not matched:
        return ""
    return PRIMARY_TASK_BY_CODE.get(matched.group(1), "")


def _blank(value):
    if value is None:
        return True
    try:
        if value != value:
            return True
    except Exception:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in ("nan", "none", "<na>")


def set_overgen_penalty(enabled):
    global OVERGEN_PENALTY
    OVERGEN_PENALTY = bool(enabled)


def infer_language(*texts):
    """按 CJK 字符占比判语种。

    只看 expect 会把 gold 是 `["2","5"]` / `["C"]` 的中文题判成 English
    （500 条里 128 条），language breakdown 和 Summary 的 jieba 分词都会错。
    所以调用方要把 prompt / response 一起传进来。
    """
    blob = "".join(str(t or "")[:LANG_SNIFF_CHARS] for t in texts)
    blob = re.sub(r"\s+", "", blob)
    if not blob:
        return "English"
    if len(CJK_CHAR.findall(blob)) / len(blob) > CJK_LANG_RATIO:
        return "Chinese"
    return "English"


def attach_task_fields(case):
    """Fill project/flavor/vertical/tasks from the case itself.

    GPU/API jsonl already carry labels. OMC logs carry expect + testCaseName;
    the vertical slug in the name is enough to pick NDCG/F1/Accuracy.
    """
    if not isinstance(case, dict):
        return case
    name = case.get("testCaseName") or ""
    project_name, flavor, vertical, index, length_from_name = parse_test_case_parts(name)
    if _blank(case.get("project_name")) and project_name:
        case["project_name"] = project_name
    if _blank(case.get("flavor")) and flavor:
        case["flavor"] = flavor
    if _blank(case.get("vertical")) and vertical:
        case["vertical"] = vertical
    if _blank(case.get("index")) and index is not None:
        case["index"] = index

    if not _blank(case.get("token_length")):
        norm = normalize_length_bucket(case.get("token_length"))
        if norm:
            case["token_length"] = norm
    if _blank(case.get("token_length")) and length_from_name:
        case["token_length"] = length_from_name
    if _blank(case.get("token_length")):
        inferred = token_length_from_prompt_len(case.get("prompt_token_len"))
        if inferred:
            case["token_length"] = inferred

    secondary = case.get("secondary_task")
    if _blank(secondary):
        secondary = secondary_task_from_vertical(case.get("vertical") or vertical)
        if secondary:
            case["secondary_task"] = secondary
            case.setdefault("_task_source", "name")
        else:
            case["_task_source"] = "missing"
    else:
        case.setdefault("_task_source", "payload")
        secondary = str(secondary).strip()

    if secondary:
        if _blank(case.get("vertical")):
            case["vertical"] = sanitize_task_token(secondary)
        if _blank(case.get("primary_task")):
            case["primary_task"] = primary_task_from_secondary(secondary)
        if _blank(case.get("metric_name")):
            case["metric_name"] = metric_name_of(secondary)

    if _blank(case.get("language")):
        case["language"] = infer_language(
            case.get("prompt"), case.get("expect"), case.get("response")
        )

    if _blank(case.get("project_name")):
        case["project_name"] = "LongBench-Pro"
    if _blank(case.get("flavor")):
        case["flavor"] = flavor or "lbp"
    if _blank(case.get("vertical")):
        case["vertical"] = vertical or ""
    return case


def metric_name_of(secondary_task):
    return TASK_METRIC_CONFIG.get(secondary_task or "", "Accuracy")


def strip_thinking(text):
    if not text:
        return ""
    cleaned = THINK_BLOCK.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


def get_answer_area(text):
    text = strip_thinking(text or "")
    if "[Answer]" in text:
        last = text.rfind("[Answer]")
        return text[last + 8:].strip()
    if "[答案]" in text:
        last = text.rfind("[答案]")
        return text[last + 4:].strip()
    return text.strip()


def has_answer_marker(text):
    body = strip_thinking(text or "")
    return "[Answer]" in body or "[答案]" in body


def lower(text):
    return text.lower()


def fix_space(text):
    return " ".join(text.split())


def drop_cjk_space(text):
    """删掉紧贴 CJK 字符的空白。

    官方只有 `fix_space`（压缩连续空格），并提醒不能无条件删空格，
    因为 "1 11" != "11 1"。但 GPU 和 OMC 两条链路都会把中文写成
    `2024 年 3 月 31 日` / `句 5` / `文档 A 第三十二条`，gold 是
    `2024年3月31日` / `句5` / `文档A第三十二条`，按行精确匹配整格判 0
    （T8.1 上 GPU 20 题全 0）。只在空白至少一侧是 CJK 时删除，
    纯 ASCII 之间的空格（`A 38%`、`1 11`）保持原样。
    """
    return CJK_SPACE.sub("", text)


def fold_letter_separators(text):
    """`g/h`、`a、c、d` 归一成 `gh`、`acd`，对齐 gold 的 `GH` / `ACD`。

    只处理纯字母列表；带数字的一律不动，避免 "1 11" 和 "11 1" 撞车。
    """
    if LETTER_LIST.match(text):
        return LETTER_SEP.sub("", text)
    return text


def normalize_text(text):
    return fold_letter_separators(drop_cjk_space(fix_space(lower(str(text)).strip())))


def normalize_answers(answers):
    return [normalize_text(a) for a in answers]


def normalize_prediction(prediction):
    area = get_answer_area(prediction)
    lines = (normalize_text(p) for p in area.split("\n"))
    return [p for p in lines if p]


def normalize_prediction_lines(prediction):
    return [p for p in normalize_prediction(prediction) if p]


def parse_expect(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, (int, float)) and raw == raw:
        return [str(raw)]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        try:
            import json
            obj = json.loads(text)
            if isinstance(obj, list):
                return [str(x) for x in obj]
            return [str(obj)]
        except (TypeError, ValueError):
            pass
    return [text]


def Accuracy(answers, prediction):
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    return 1.0 if answers[0] == predictions[0] else 0.0


def F1_Score(answers, prediction):
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    answer_set = set(answers)
    prediction_set = set(predictions)
    common = answer_set & prediction_set
    if not common or not prediction_set or not answer_set:
        return 0.0
    precision = len(common) / len(prediction_set)
    recall = len(common) / len(answer_set)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def SubEM(answers, prediction):
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    score = 0.0
    for a in answers:
        if a in predictions:
            score += 1.0
    value = score / len(answers)
    return max(0.0, min(1.0, value * overgen_scale(len(answers), len(predictions))))


def overgen_scale(n_gold, n_pred_lines):
    """Dilute list metrics when the model dumps many more lines than asked.

    Official NDCG@k / SubEM ignore extra lines (or treat them as recall-only),
    so a quantized run that prints 1..N can beat a short, well-ranked GPU
    answer. Scale by n_gold / n_pred when n_pred > n_gold, i.e. list precision.

    设 OVERGEN_PENALTY=False 可退回官方口径。
    """
    if not OVERGEN_PENALTY:
        return 1.0
    if n_gold <= 0 or n_pred_lines <= n_gold:
        return 1.0
    return n_gold / float(n_pred_lines)


def NDCG(answers, prediction):
    """NDCG@k (k = len(answers)) with list-precision dilution.

    Core DCG matches official LongBench-Pro / pytrec_eval: graded relevance is
    the gold rank, only the first k unique prediction lines count. Extra dump
    lines after that are then penalized by overgen_scale, otherwise printing
    every ID in order scores ~0.8 while a short GPU ranking scores ~0.3.
    """
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    k = len(answers)
    rel_map = {a: k - i for i, a in enumerate(answers)}
    seen = set()
    ranked = []
    for pred in predictions:
        if pred in seen:
            continue
        seen.add(pred)
        ranked.append(pred)
        if len(ranked) >= k:
            break
    dcg = 0.0
    for i, pred in enumerate(ranked, start=1):
        rel = rel_map.get(pred, 0)
        dcg += rel / math.log2(i + 1)
    idcg = 0.0
    for i in range(1, k + 1):
        rel = k - i + 1
        idcg += rel / math.log2(i + 1)
    if idcg <= 0:
        return 0.0
    value = (dcg / idcg) * overgen_scale(k, len(predictions))
    return max(0.0, min(1.0, value))


def Pairwise_Accuracy(answers, prediction):
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    if len(answers) < 2 or len(predictions) < 2:
        return 0.0
    n_total = len(predictions) * (len(predictions) - 1) // 2
    if n_total <= 0:
        return 0.0
    prediction_indices = {p: i for i, p in enumerate(predictions)}
    n_correct = 0
    for a, b in combinations(answers, 2):
        if a in prediction_indices and b in prediction_indices:
            if prediction_indices[a] < prediction_indices[b]:
                n_correct += 1
    return n_correct / n_total


def _summary_max_rouge_l(answers, prediction, is_zh):
    try:
        from rouge import Rouge
    except ImportError:
        print("[WARN] rouge not installed; Summary Rouge-L = 0")
        return 0.0
    pred = prediction
    refs = list(answers)
    if is_zh:
        try:
            import jieba
        except ImportError:
            print("[WARN] jieba not installed; Chinese Summary uses char split")
            refs = [" ".join(list(a)) for a in refs]
            pred = " ".join(list(pred))
        else:
            refs = [" ".join(list(jieba.cut(a, cut_all=False))) for a in refs]
            pred = " ".join(list(jieba.cut(pred, cut_all=False)))
    if not pred.strip() or not any(r.strip() for r in refs):
        return 0.0
    try:
        scores = Rouge().get_scores([pred] * len(refs), refs, avg=False)
    except Exception:
        return 0.0
    return max(score["rouge-l"]["f"] for score in scores)


def _summary_max_semantic(answers, prediction):
    if _embedding_model is None:
        return None
    answer_embeddings = _embedding_model.encode(answers)
    prediction_embeddings = _embedding_model.encode([prediction])
    similarity = _embedding_model.similarity(answer_embeddings, prediction_embeddings)
    return float(similarity.max().cpu().numpy())


def Summary(answers, prediction, is_zh, alpha=0.5, beta=0.5):
    answers = normalize_answers(answers)
    prediction = normalize_text(get_answer_area(prediction))
    if not answers or not prediction:
        return 0.0
    rouge_l = _summary_max_rouge_l(answers, prediction, is_zh)
    if _embedding_enabled:
        sim = _summary_max_semantic(answers, prediction)
        if sim is None:
            return rouge_l
        return alpha * sim + beta * rouge_l
    return rouge_l


def calculate_metric(secondary_task, answers, prediction, language):
    if not prediction:
        return 0.0
    name = metric_name_of(secondary_task)
    is_zh = str(language).strip().lower() in ("chinese", "zh", "cn")
    if name == "NDCG":
        value = NDCG(answers, prediction)
    elif name == "Pairwise_Accuracy":
        value = Pairwise_Accuracy(answers, prediction)
    elif name == "Accuracy":
        value = Accuracy(answers, prediction)
    elif name == "F1_Score":
        value = F1_Score(answers, prediction)
    elif name == "SubEM":
        value = SubEM(answers, prediction)
    elif name == "Summary":
        value = Summary(answers, prediction, is_zh)
    else:
        value = Accuracy(answers, prediction)
    if value != value:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def is_pass(secondary_task, metric):
    if metric_name_of(secondary_task) == "Summary":
        return metric > SUMMARY_PASS_THRESHOLD
    return metric >= 1.0 - 1e-12


def score_case(secondary_task, expect, response, language):
    answers = parse_expect(expect)
    metric = calculate_metric(secondary_task, answers, response or "", language)
    prediction_text = "\n".join(normalize_prediction_lines(response or ""))
    return metric, bool(is_pass(secondary_task, metric)), prediction_text


def verify_case(case, response):
    secondary = case.get("secondary_task") or ""
    language = case.get("language") or ""
    metric, passed, _ = score_case(secondary, case.get("expect"), response, language)
    return passed, metric
