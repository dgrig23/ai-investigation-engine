"""Build the reproducible HotpotQA subset.

Source: hotpotqa/hotpot_qa, "distractor" config, validation split (7,405 questions). Every question
comes with 10 Wikipedia paragraphs: 2 "gold" paragraphs that contain the answer and 8 distractors
chosen to be similar. Pooling the paragraphs of all questions gives ~74k paragraphs; after
removing exact duplicates (the same Wikipedia paragraph appears in several questions) the corpus
has well over 50,000 unique documents, and the distractors make retrieval genuinely hard.

Outputs:
- data/processed/hotpotqa_docs.jsonl     one JSON document per unique paragraph
- eval/datasets/hotpotqa_questions.jsonl a fixed random sample (seed 42) of questions with gold doc ids

Usage:  python -m scripts.prepare_hotpotqa [--sample 300]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hotpotqa_validation.parquet"
URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
DOCS_OUT = ROOT / "data" / "processed" / "hotpotqa_docs.jsonl"
QUESTIONS_OUT = ROOT / "eval" / "datasets" / "hotpotqa_questions.jsonl"


def doc_id_for(title: str) -> str:
    return f"hotpotqa:{title}"


ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation"
RAW_JSONL = ROOT / "data" / "raw" / "hotpotqa_validation.jsonl"


def load_rows() -> list[dict]:
    """Read the split from parquet if pyarrow is installed, otherwise via the HF rows API (cached)."""
    try:
        import pyarrow.parquet as pq

        if not RAW.exists():
            RAW.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {URL}")
            RAW.write_bytes(httpx.get(URL, follow_redirects=True, timeout=300).content)
        return pq.read_table(RAW).to_pylist()
    except ImportError:
        pass
    if not RAW_JSONL.exists():
        RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
        rows, offset = [], 0
        while True:
            for attempt in range(8):  # the rows API rate-limits bursts: back off and retry
                resp = httpx.get(f"{ROWS_API}&offset={offset}&length=100", timeout=60)
                if resp.status_code != 429 and resp.status_code < 500:
                    break
                time.sleep(min(5 * 2 ** attempt, 60))
            page = resp.raise_for_status().json()
            rows += [r["row"] for r in page["rows"]]
            offset += 100
            print(f"  fetched {len(rows)}/{page['num_rows_total']}", end="\r")
            if offset >= page["num_rows_total"]:
                break
        with RAW_JSONL.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print()
    with RAW_JSONL.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300, help="number of evaluation questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows()
    rows.sort(key=lambda r: r["id"])  # same order regardless of source -> same random sample
    print(f"{len(rows)} questions")

    docs: dict[str, dict] = {}
    paragraphs_total = 0
    for row in rows:
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"], strict=True):
            paragraphs_total += 1
            text = "".join(sentences).strip()
            if not text:
                continue
            did = doc_id_for(title)
            if did not in docs:  # same title -> same Wikipedia paragraph: keep one copy
                docs[did] = {"doc_id": did, "title": title, "text": text}

    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DOCS_OUT.open("w") as f:
        for d in docs.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{paragraphs_total} paragraphs -> {len(docs)} unique documents -> {DOCS_OUT}")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.sample)
    QUESTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with QUESTIONS_OUT.open("w") as f:
        for row in sample:
            gold_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
            f.write(json.dumps({
                "id": f"hotpot-{row['id']}",
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],  # bridge (multi-hop chain) or comparison (multi-document)
                "level": row["level"],
                "gold_doc_ids": [doc_id_for(t) for t in gold_titles],
            }, ensure_ascii=False) + "\n")
    print(f"{len(sample)} evaluation questions -> {QUESTIONS_OUT}")


if __name__ == "__main__":
    main()
