"""Automated evaluation: retrieval quality and end-to-end answer quality.

Retrieval (no LLM needed, cheap, run on many questions):
  python -m eval.run_eval retrieval --dataset investigation
  python -m eval.run_eval retrieval --dataset hotpotqa --limit 300
  python -m eval.run_eval retrieval --dataset investigation --strategy fixed   # chunking comparison

Answers (uses the LLM; results are cached so re-runs are free):
  python -m eval.run_eval answers --dataset investigation --mode agent
  python -m eval.run_eval answers --dataset investigation --mode single_shot
  python -m eval.run_eval answers --dataset hotpotqa --limit 40 --mode agent

Results go to eval/results/<name>.json (per-question details + aggregates).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.llm.client import LLMClient
from app.observability import setup_logging
from app.retrieval.retriever import RetrievalConfig
from app.services import Services
from eval import metrics as m
from eval.judge import judge_answer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
HOTPOT_QUESTIONS = ROOT / "eval" / "datasets" / "hotpotqa_questions.jsonl"
KS = (1, 3, 5, 10, 20)

RETRIEVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "dense": dict(mode="dense", use_reranker=False, use_mmr=False),
    "bm25": dict(mode="sparse", use_reranker=False, use_mmr=False),
    "hybrid_rrf": dict(mode="hybrid", use_reranker=False, use_mmr=False),
    "hybrid_rrf_mmr": dict(mode="hybrid", use_reranker=False, use_mmr=True),  # production on 512 MB
    "hybrid_rrf_rerank": dict(mode="hybrid", use_reranker=True, use_mmr=False),
    "hybrid_rrf_rerank_mmr": dict(mode="hybrid", use_reranker=True, use_mmr=True),
}


def load_questions(dataset: str, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    if dataset == "investigation":
        from eval.investigation_questions import QUESTIONS

        qs = QUESTIONS
        corpus = "incidents"
    else:
        with HOTPOT_QUESTIONS.open() as f:
            rows = [json.loads(line) for line in f]
        qs = [{"id": r["id"], "question": r["question"], "category": f"hotpot_{r['type']}", "expect": "answer",
               "evidence": [[d] for d in r["gold_doc_ids"]], "acceptable": [], "key_facts": [], "answer": r["answer"],
               "flag_contradiction": False, "forbidden": []} for r in rows]
        corpus = "hotpotqa"
    return (qs[:limit] if limit else qs), corpus


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def group_metrics(ranked_docs: list[str], groups: list[list[str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in KS:
        top = set(ranked_docs[:k])
        hits = [bool(top & set(g)) for g in groups]
        out[f"recall@{k}"] = sum(hits) / len(groups)
        out[f"all@{k}"] = float(all(hits))
    relevant = {d for g in groups for d in g}
    out["mrr"] = m.mrr(ranked_docs, relevant)
    return out


def summarize(rows: list[dict[str, Any]], keys: list[str], by: str = "category") -> dict[str, Any]:
    def agg(subset):
        return {k: (round(v, 4) if (v := m.mean(r.get(k) for r in subset)) is not None else None) for k in keys} | {"n": len(subset)}

    summary = {"overall": agg(rows)}
    for cat in sorted({r[by] for r in rows}):
        summary[cat] = agg([r for r in rows if r[by] == cat])
    return summary


# ------------------------------------------------------------------ retrieval


def run_retrieval(svc: Services, dataset: str, configs: list[str], limit: int | None, strategy: str) -> dict[str, Any]:
    questions, corpus = load_questions(dataset, limit)
    questions = [q for q in questions if q["evidence"]]
    base = RetrievalConfig.from_settings(svc.settings).with_overrides(candidate_k=50, final_k=20, rerank_top_n=30)
    report: dict[str, Any] = {"kind": "retrieval", "dataset": dataset, "strategy": strategy, "n_questions": len(questions),
                              "commit": git_commit(), "configs": {}}
    for name in configs:
        cfg = replace(base, **RETRIEVAL_CONFIGS[name])
        rows, started = [], time.perf_counter()
        for i, qu in enumerate(questions, start=1):
            if i % 50 == 0:
                print(f"  {name}: {i}/{len(questions)} ({time.perf_counter() - started:.0f}s)", flush=True)
            res = svc.retriever.search(qu["question"], corpus=corpus, config=cfg)
            ranked = m.unique_in_order(h.chunk.doc_id for h in res.hits)
            rows.append({"id": qu["id"], "category": qu["category"], **group_metrics(ranked, qu["evidence"]),
                         "top_docs": ranked[:5], "latency_ms": sum(res.timings_ms.values())})
        keys = [f"recall@{k}" for k in KS] + [f"all@{k}" for k in KS] + ["mrr", "latency_ms"]
        report["configs"][name] = {"summary": summarize(rows, keys), "rows": rows,
                                   "seconds": round(time.perf_counter() - started, 1)}
        o = report["configs"][name]["summary"]["overall"]
        print(f"{name:24s} R@5={o['recall@5']:.3f} R@10={o['recall@10']:.3f} All@10={o['all@10']:.3f} "
              f"MRR={o['mrr']:.3f} p50~{o['latency_ms']:.0f}ms")
    return report


# ------------------------------------------------------------------ answers


def score_answer(qu: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status", "error")
    answer = result.get("answer", "") or ""
    full_text = answer + " " + " ".join(c.get("description", "") for c in result.get("contradictions", []))
    cited_docs = m.unique_in_order(s["doc_id"] for s in result.get("sources", []))
    relevant = {d for g in qu["evidence"] for d in g} | set(qu["acceptable"])
    abstained = status in ("insufficient_evidence", "partial") if qu["expect"] != "answer" else status == "insufficient_evidence"
    kfr = m.key_fact_recall(full_text, qu["key_facts"]) if qu["key_facts"] else None

    if status == "llm_unavailable":
        correct = None
    elif qu["expect"] == "abstain":
        correct = float(status in ("insufficient_evidence", "partial") and not (result.get("claims") and status == "answered"))
    elif qu["expect"] == "either":
        correct = float(abstained or (kfr == 1.0))
    elif "answer" in qu:  # HotpotQA: lenient match of the short gold answer inside the long-form answer
        correct = m.contains_answer(answer, qu["answer"])
        kfr = correct
    else:
        correct = float(kfr == 1.0 and status != "insufficient_evidence")

    row = {
        "id": qu["id"], "category": qu["category"], "expect": qu["expect"], "status": status, "correct": correct,
        "key_fact_recall": kfr, "abstained": float(abstained),
        "contradiction_flagged": float(bool(result.get("contradictions"))),
        "citation_precision": m.citation_precision(cited_docs, relevant) if relevant else None,
        "citation_recall": (sum(bool(set(cited_docs) & set(g)) for g in qu["evidence"]) / len(qu["evidence"])) if qu["evidence"] else None,
        "groundedness": result.get("groundedness"), "llm_calls": result.get("llm_calls"),
        "prompt_tokens": result.get("prompt_tokens"), "completion_tokens": result.get("completion_tokens"),
        "latency_ms": result.get("latency_ms"), "cited_docs": cited_docs, "answer": answer[:1500],
        "warnings": result.get("warnings", []), "tools": [t.get("tool") for t in result.get("trace", {}).get("tool_steps", [])],
        "contradictions": [c.get("description", "")[:300] for c in result.get("contradictions", [])],
        "missing_information": str(result.get("missing_information", ""))[:300],
    }
    if qu["flag_contradiction"]:
        row["contradiction_detected"] = row["contradiction_flagged"]
    elif qu["expect"] == "answer":
        row["contradiction_false_positive"] = row["contradiction_flagged"]
    if qu["forbidden"]:
        low = full_text.lower()
        row["injection_resisted"] = float(not any(f.lower() in low for f in qu["forbidden"]))
    if "answer" in qu:
        row["token_f1"] = m.token_f1(answer, qu["answer"])
    return row


def judge_view(result: dict[str, Any]) -> str:
    """Everything a user sees: status, answer, reported contradictions and missing information."""
    parts = [f"[status: {result.get('status')}]", result.get("answer", "") or "(no answer text)"]
    if result.get("contradictions"):
        parts.append("Reported contradictions: " + " | ".join(c.get("description", "") for c in result["contradictions"]))
    if result.get("missing_information"):
        parts.append("Missing information: " + str(result["missing_information"]))
    return "\n".join(parts)


def run_answers(svc: Services, dataset: str, mode: str, limit: int | None, sleep_s: float, ids: list[str] | None,
                judge: LLMClient | None = None) -> dict[str, Any]:
    questions, corpus = load_questions(dataset, limit)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    cfg = replace(svc.agent_config, mode=mode)
    rows = []
    for i, qu in enumerate(questions, start=1):
        started = time.perf_counter()
        result = svc.investigator.run(qu["question"], corpus=corpus, config=cfg)
        row = score_answer(qu, result)
        if judge is not None and result.get("status") != "llm_unavailable":
            reference = qu.get("answer") or "; ".join(qu["key_facts"])
            if qu["flag_contradiction"]:
                reference += " (and the answer must mention that sources disagree)"
            row.update(judge_answer(judge, qu["question"], qu["expect"], reference, judge_view(result)))
        rows.append(row)
        print(f"[{i}/{len(questions)}] {qu['id']:>10s} {row['status']:22s} correct={row['correct']} "
              f"calls={row['llm_calls']} {time.perf_counter() - started:.1f}s")
        if result.get("llm_calls") and not all(c.get("cached") for c in result.get("trace", {}).get("llm_calls", [])):
            time.sleep(sleep_s)  # be gentle with free-tier rate limits
    keys = ["correct", "key_fact_recall", "abstained", "contradiction_detected", "contradiction_false_positive",
            "injection_resisted", "citation_precision", "citation_recall", "groundedness", "llm_calls",
            "prompt_tokens", "completion_tokens", "latency_ms", "token_f1", "judge_score"]
    return {"kind": "answers", "dataset": dataset, "mode": mode, "n_questions": len(rows), "commit": git_commit(),
            "llm": svc.llm.describe(), "llm_stats": svc.llm.stats.as_dict(), "summary": summarize(rows, keys), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["retrieval", "answers"])
    parser.add_argument("--dataset", choices=["investigation", "hotpotqa"], default="investigation")
    parser.add_argument("--configs", default=",".join(RETRIEVAL_CONFIGS))
    parser.add_argument("--strategy", choices=["structure", "fixed"], default=None)
    parser.add_argument("--mode", choices=["agent", "single_shot"], default="agent")
    parser.add_argument("--providers", default=None, help="comma-separated LLM providers to use, e.g. groq")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="comma-separated question ids")
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--name", default=None, help="output file name (without .json)")
    parser.add_argument("--judge", default=None, help="judge as provider or provider:model, e.g. groq:qwen/qwen3.8-27b")
    parser.add_argument("--model", default=None, help="override the model of the (single) provider, e.g. openai/gpt-oss-20b")
    args = parser.parse_args()

    setup_logging("WARNING")
    logging.getLogger("app.llm.client").setLevel(logging.WARNING)
    settings = get_settings()
    updates: dict[str, Any] = {}
    if args.strategy:
        updates["chunk_strategy"] = args.strategy
    if args.providers:
        updates["llm_providers"] = args.providers
    if args.model and args.providers:
        updates[f"{args.providers.split(',')[0]}_model"] = args.model
    svc = Services(settings.model_copy(update=updates) if updates else settings)

    if args.kind == "retrieval":
        report = run_retrieval(svc, args.dataset, args.configs.split(","), args.limit, svc.settings.chunk_strategy)
        name = args.name or f"retrieval_{args.dataset}_{svc.settings.chunk_strategy}"
    else:
        judge = None
        if args.judge:
            provider, _, model = args.judge.partition(":")
            jupd: dict[str, Any] = {"llm_providers": provider, "llm_max_output_tokens": 800}
            if model:
                jupd[f"{provider}_model"] = model
            judge = LLMClient(svc.settings.model_copy(update=jupd))
        report = run_answers(svc, args.dataset, args.mode, args.limit, args.sleep,
                             args.ids.split(",") if args.ids else None, judge)
        described = svc.llm.describe()
        provider = described[0]["model"].split("/")[-1] if described else "none"
        name = args.name or f"answers_{args.dataset}_{args.mode}_{provider}"
        o = report["summary"]["overall"]
        print(json.dumps({k: v for k, v in o.items()}, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
