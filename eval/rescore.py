"""Re-apply the deterministic answer metrics to saved results (no LLM calls).

Used when a metric is fixed after a run (e.g. '80 %' vs '80%'): the stored answers are re-scored so
runs made before and after the fix are compared with the same metric.

Usage:  python -m eval.rescore eval/results/answers_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval import metrics as m
from eval.run_eval import load_questions, summarize


def rescore(path: str) -> None:
    report = json.loads(Path(path).read_text())
    questions = {q["id"]: q for q in load_questions(report["dataset"], None)[0]}
    changed = 0
    for row in report["rows"]:
        qu = questions[row["id"]]
        if row["status"] == "llm_unavailable" or not qu["key_facts"] or qu["expect"] != "answer":
            continue
        kfr = m.key_fact_recall(row["answer"], qu["key_facts"])
        correct = float(kfr == 1.0 and row["status"] != "insufficient_evidence")
        if kfr > (row.get("key_fact_recall") or 0) or correct != row.get("correct"):
            changed += 1
            row["key_fact_recall"], row["correct"] = max(kfr, row.get("key_fact_recall") or 0), max(correct, row["correct"] or 0)
    keys = [k for k in next(iter(report["summary"].values())) if k != "n"]
    report["summary"] = summarize(report["rows"], keys)
    report["rescored"] = True
    Path(path).write_text(json.dumps(report, indent=1, default=str))
    print(f"{path}: {changed} rows re-scored, correct={report['summary']['overall']['correct']}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rescore(p)
