import math
import re
from collections import defaultdict
from difflib import SequenceMatcher


def GET_answer(answer):
    if not answer:
        return False
    return "[Answer]" in answer or "[答案]" in answer


def split_sentences(text):
    delimiters = r"[。！？；;.!?\n\r]+"
    return [s.strip() for s in re.split(delimiters, text or "") if s.strip()]


def calculate_repetition_rate(text, similarity_threshold=0.85):
    sentences = split_sentences(text)
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return 0.0

    groups = defaultdict(list)
    for sent in sentences:
        matched = False
        for key in groups:
            if SequenceMatcher(None, key, sent).ratio() >= similarity_threshold:
                groups[key].append(sent)
                matched = True
                break
        if not matched:
            groups[sent].append(sent)

    repeated_chars = sum(
        sum(len(sent) for sent in group[1:])
        for group in groups.values()
        if len(group) > 1
    )
    return repeated_chars / total_chars


def calculate_char_entropy(text):
    if not text:
        return 0.0
    counts = defaultdict(int)
    for ch in text:
        counts[ch] += 1
    total = len(text)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


calculate_token_entropy = calculate_char_entropy
