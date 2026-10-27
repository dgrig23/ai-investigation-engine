"""Score existing answer-evaluation results with one LLM judge, so every run is judged the same way.

Usage:
  python -m eval.rejudge --judge gemini:gemini-flash-lite-latest eval/results/answers_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.llm.client import LLMClient
from eval.judge import judge_answer
from eval.run_eval import load_questions, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, help="provider:model")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    provider, _, model = args.judge.partition(":")
    upd: dict[str, Any] = {"llm_providers": provider, "llm_max_output_tokens": 800}
    if model:
        upd[f"{provider}_model"] = model
    judge = LLMClient(get_settings().model_copy(update=upd))

    for file in args.files:
        report = json.loads(Path(file).read_text())
        questions = {q["id"]: q for q in load_questions(report["dataset"], None)[0]}
        for row in report["rows"]:
            qu = questions[row["id"]]
            if row["status"] == "llm_unavailable":
                continue
            reference = qu.get("answer") or "; ".join(qu["key_facts"])
            if qu["flag_contradiction"]:
                reference += " (and the answer must mention that sources disagree)"
            row.update(judge_answer(judge, qu["question"], qu["expect"], reference, row["answer"]))
        keys = list(next(iter(report["summary"].values())).keys())
        keys = [k for k in keys if k != "n"] + (["judge_score"] if "judge_score" not in keys else [])
        report["summary"] = summarize(report["rows"], keys)
        report["judge"] = args.judge
        Path(file).write_text(json.dumps(report, indent=1, default=str))
        print(f"{file}: judge_score={report['summary']['overall'].get('judge_score')}")


if __name__ == "__main__":
    main()
