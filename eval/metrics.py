"""Evaluation metrics (pure functions, unit-tested).

Retrieval (document level - several chunks of one document count once):
- Recall@k: share of gold documents found in the top k.
- AllRecall@k ("complete evidence"): 1 if *every* gold document is in the top k. For multi-hop
  questions this is what matters: one hop is not enough to answer.
- MRR: 1 / rank of the first gold document (0 if none).
- nDCG@k: rank-discounted gain with binary relevance.

Answers:
- Exact match and token F1 (standard SQuAD/HotpotQA answer metrics).
- Key-fact recall: share of required facts present in the answer; a fact may list alternatives
  separated by "|" (e.g. "107 minutes|107 min").
- Citation precision: share of cited documents that are relevant (gold or explicitly acceptable).
- Citation recall: share of gold documents that the answer cites.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence


def unique_in_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def all_recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    return float(bool(gold) and gold <= set(retrieved[:k]))


def mrr(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    gold = set(gold)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, gold: str) -> float:
    """Lenient match for long-form answers: the normalized gold answer appears in the prediction."""
    g = normalize_answer(gold)
    return float(bool(g) and g in normalize_answer(prediction))


_UNICODE = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u202f": " "})


def _canon(text: str) -> str:
    """Lower-case, ASCII hyphens/spaces, no thousands separators, '80 %' == '80%'."""
    text = text.translate(_UNICODE).lower().replace(",", "")
    return re.sub(r"(\d)\s+%", r"\1%", text)


def fact_present(answer: str, fact: str) -> bool:
    low = _canon(answer)
    return any(_canon(alt.strip()) in low for alt in fact.split("|") if alt.strip())


def key_fact_recall(answer: str, facts: Sequence[str]) -> float:
    if not facts:
        return 1.0
    return sum(fact_present(answer, f) for f in facts) / len(facts)


def citation_precision(cited_docs: Iterable[str], relevant_docs: Iterable[str]) -> float | None:
    cited = set(cited_docs)
    if not cited:
        return None  # no citations -> undefined (reported separately)
    return len(cited & set(relevant_docs)) / len(cited)


def citation_recall(cited_docs: Iterable[str], gold_docs: Iterable[str]) -> float | None:
    gold = set(gold_docs)
    if not gold:
        return None
    return len(gold & set(cited_docs)) / len(gold)


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
