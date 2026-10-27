"""Turn eval/results/*.json into tables (docs/eval_tables.md) and charts (docs/img/*.png).

The written analysis lives in docs/EVALUATION.md and references these generated tables.
Usage:  python -m eval.report
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
DOCS = ROOT / "docs"
IMG = DOCS / "img"


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def retrieval_table(report: dict) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "recall@20", "all@5", "all@10", "mrr", "latency_ms"]
    lines = [f"**{report['dataset']} - chunking: {report['strategy']} - {report['n_questions']} questions**", "",
             "| config | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, data in report["configs"].items():
        o = data["summary"]["overall"]
        lines.append(f"| {name} | " + " | ".join(_fmt(o.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def retrieval_by_category(report: dict, config: str) -> str:
    summary = report["configs"][config]["summary"]
    cols = ["n", "recall@5", "recall@10", "all@10", "mrr"]
    lines = [f"**{report['dataset']} / {config} by category**", "", "| category | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for cat, o in summary.items():
        if cat != "overall":
            lines.append(f"| {cat} | " + " | ".join(_fmt(o.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def answers_table(reports: list[dict]) -> str:
    cols = ["correct", "key_fact_recall", "citation_precision", "citation_recall", "groundedness",
            "contradiction_detected", "contradiction_false_positive", "injection_resisted", "llm_calls", "latency_ms"]
    lines = ["| run | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for r in reports:
        o = r["summary"]["overall"]
        llm = "+".join(p["model"] for p in r.get("llm", [])[:1]) or "?"
        version = " (v1)" if r.get("_file", "").endswith("_v1.json") else ""
        lines.append(f"| {r['dataset']} / {r['mode']} / {llm}{version} | {o['n']} | " + " | ".join(_fmt(o.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def answers_by_category(report: dict) -> str:
    cols = ["n", "correct", "key_fact_recall", "abstained", "citation_precision", "llm_calls"]
    lines = [f"**{report['dataset']} / {report['mode']} by category**", "",
             "| category | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for cat, o in report["summary"].items():
        if cat != "overall":
            lines.append(f"| {cat} | " + " | ".join(_fmt(o.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def failures(report: dict, limit: int = 15) -> str:
    rows = [r for r in report["rows"] if r.get("correct") == 0.0]
    lines = [f"**Failures: {report['dataset']} / {report['mode']} ({len(rows)} incorrect)**", ""]
    for r in rows[:limit]:
        lines.append(f"- `{r['id']}` ({r['category']}, expected {r['expect']}, got {r['status']}, "
                     f"key facts {_fmt(r.get('key_fact_recall'))}): {r['answer'][:220].replace(chr(10), ' ')}…")
    return "\n".join(lines)


def chart(retrieval_reports: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    IMG.mkdir(parents=True, exist_ok=True)
    made = []
    for rep in retrieval_reports:
        names = list(rep["configs"])
        r10 = [rep["configs"][n]["summary"]["overall"]["recall@10"] for n in names]
        a10 = [rep["configs"][n]["summary"]["overall"]["all@10"] for n in names]
        mrr = [rep["configs"][n]["summary"]["overall"]["mrr"] for n in names]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        x = range(len(names))
        w = 0.27
        ax.bar([i - w for i in x], r10, w, label="Recall@10", color="#4C78A8")
        ax.bar(list(x), a10, w, label="All@10", color="#72B7B2")
        ax.bar([i + w for i in x], mrr, w, label="MRR", color="#F58518")
        ax.set_xticks(list(x), [n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Retrieval: {rep['dataset']} ({rep['strategy']} chunking, n={rep['n_questions']})", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = IMG / f"retrieval_{rep['dataset']}_{rep['strategy']}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(path.name)
    return made


ORDER = ["dense", "bm25", "hybrid_rrf", "hybrid_rrf_mmr", "hybrid_rrf_rerank", "hybrid_rrf_rerank_mmr"]


def merge_retrieval(reports: list[dict]) -> list[dict]:
    """Runs of the same dataset + chunking strategy (e.g. an extra config run later) become one table."""
    merged: dict[tuple[str, str], dict] = {}
    for r in reports:
        key = (r["dataset"], r["strategy"])
        if key in merged:
            merged[key]["configs"].update(r["configs"])
        else:
            merged[key] = {**r, "configs": dict(r["configs"])}
    for r in merged.values():
        r["configs"] = {k: r["configs"][k] for k in sorted(r["configs"], key=lambda c: ORDER.index(c) if c in ORDER else 99)}
    return list(merged.values())


def main() -> None:
    reports = []
    for p in sorted(RESULTS.glob("*.json")):
        if not p.name.startswith("smoke"):
            rep = json.loads(p.read_text())
            rep["_file"] = p.name
            reports.append(rep)
    retrieval = merge_retrieval([r for r in reports if r["kind"] == "retrieval"])
    answers = [r for r in reports if r["kind"] == "answers"]
    parts = ["# Generated evaluation tables", "", "_Generated by `python -m eval.report` from `eval/results/*.json`._", ""]
    if retrieval:
        parts += ["## Retrieval", ""]
        for r in retrieval:
            parts += [retrieval_table(r), ""]
            best = "hybrid_rrf_rerank_mmr" if "hybrid_rrf_rerank_mmr" in r["configs"] else list(r["configs"])[-1]
            parts += [retrieval_by_category(r, best), ""]
        for img in chart(retrieval):
            parts += [f"![{img}](img/{img})", ""]
    if answers:
        parts += ["## Answers", "", answers_table(answers), ""]
        for r in answers:
            parts += [answers_by_category(r), "", failures(r), ""]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "eval_tables.md").write_text("\n".join(parts))
    print(f"-> {DOCS / 'eval_tables.md'}")


if __name__ == "__main__":
    main()
