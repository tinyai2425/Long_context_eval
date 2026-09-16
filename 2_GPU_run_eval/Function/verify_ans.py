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

TEST_CASE_NAME_PATTERN = re.compile(
    r"^(?P<project_name>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*-[0-9]+)-"
    r"(?P<flavor>[a-zA-Z0-9_]+)-test-"
    r"(?P<vertical>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*)-[0-9]+$"
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
SUMMARY_PASS_THRESHOLD = 0.65

_embedding_model = None
_embedding_enabled = False


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


def parse_test_case_name(name):
    matched = TEST_CASE_NAME_PATTERN.match(name or "")
    if not matched:
        return "", "", ""
    return matched.group("project_name"), matched.group("flavor"), matched.group("vertical")


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


def normalize_answers(answers):
    return [fix_space(lower(str(a)).strip()) for a in answers]


def normalize_prediction(prediction):
    area = get_answer_area(prediction)
    return [fix_space(p.strip()) for p in lower(area).split("\n")]


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
    return score / len(answers)


def NDCG(answers, prediction):
    answers = normalize_answers(answers)
    predictions = normalize_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    k = len(answers)
    rel_map = {a: k - i for i, a in enumerate(answers)}
    dcg = 0.0
    for i, pred in enumerate(predictions[:k], start=1):
        rel = rel_map.get(pred, 0)
        dcg += rel / math.log2(i + 1)
    idcg = 0.0
    for i in range(1, k + 1):
        rel = k - i + 1
        idcg += rel / math.log2(i + 1)
    if idcg <= 0:
        return 0.0
    value = dcg / idcg
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
    prediction = fix_space(lower(get_answer_area(prediction)).strip())
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
