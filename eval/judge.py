"""LLM-as-judge for answer correctness (optional, complements the deterministic key-fact check).

The key-fact metric is strict and reproducible but can miss correct answers phrased differently
("about one and three-quarter hours" vs "107 minutes"). The judge reads the question, the reference
facts and the system's answer and returns a structured grade via a forced tool call. Use a different
model from the one being evaluated to reduce self-preference bias; agreement between the judge and the
key-fact metric is reported in the evaluation.
"""

from __future__ import annotations

from typing import Any

from app.llm.client import LLMClient, LLMError

GRADE_TOOL = {
    "type": "function",
    "function": {
        "name": "grade",
        "description": "Grade the system answer against the reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "1 = fully correct, 0.5 = partially correct, 0 = wrong or missing"},
                "reason": {"type": "string", "description": "One short sentence."},
            },
            "required": ["score", "reason"],
        },
    },
}

JUDGE_PROMPT = """You grade answers of a document-investigation system.
Question: {question}
Expected behaviour: {expect}
Reference facts (all must be conveyed for full credit): {reference}
System answer:
\"\"\"{answer}\"\"\"

Grade only factual correctness and completeness against the reference - not style or length.
If the expected behaviour is to abstain, give 1 only if the system clearly says the information is not available."""


def judge_answer(llm: LLMClient, question: str, expect: str, reference: str, answer: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(question=question, expect=expect, reference=reference or "(none)", answer=answer[:3000])
    try:
        resp = llm.chat([{"role": "user", "content": prompt}], tools=[GRADE_TOOL], tool_choice="required", max_tokens=200)
    except LLMError as exc:
        return {"judge_score": None, "judge_reason": f"judge unavailable: {exc}"[:200]}
    for tc in resp.tool_calls:
        if tc.name == "grade" and not tc.parse_error:
            try:
                score = max(0.0, min(1.0, float(tc.arguments.get("score"))))
            except (TypeError, ValueError):
                break
            return {"judge_score": score, "judge_reason": str(tc.arguments.get("reason", ""))[:300],
                    "judge_model": resp.model}
    return {"judge_score": None, "judge_reason": "judge returned no grade"}
