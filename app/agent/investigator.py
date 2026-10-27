"""The investigation agent (agentic RAG with tool calling).

    question ─ initial hybrid search ─┐
                                      ▼
            ┌──────────── LLM decides: search more? follow an ID? read a document? ──┐
            │   tools: search_documents · find_mentions · read_document              │
            └──────────── results (as numbered evidence) go back to the LLM ◄────────┘
                                      ▼  (max N steps, or when the model is confident)
                              submit_answer (structured: answer, claims, contradictions, status)
                                      ▼
            verification: citation check · exact-number grounding · conflict merge · abstention gate

Retrieval runs *before* the first LLM call, so simple questions need a single LLM call; only
multi-hop questions pay for extra steps. `mode="single_shot"` disables the tools entirely and is
the classic one-retrieval RAG baseline that the evaluation compares against.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.agent.contradictions import ConflictHint, find_conflicts
from app.agent.evidence import EvidenceRegistry
from app.agent.prompts import RESEARCH_TOOLS, SUBMIT_TOOL, SYSTEM_PROMPT
from app.agent.verify import verify_answer
from app.index.store import VectorStore, build_filter
from app.ingestion.metadata import ID_RE
from app.llm.client import LLMClient, LLMUnavailableError
from app.retrieval.retriever import HybridRetriever, RetrievalConfig

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    mode: str = "agent"  # "agent" | "single_shot"
    max_steps: int = 4
    snippet_chars: int = 700
    max_context_chars: int = 24000
    abstain_threshold: float = -4.0
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


class Investigator:
    def __init__(self, retriever: HybridRetriever, store: VectorStore, llm: LLMClient, embedder=None,
                 config: AgentConfig | None = None):
        self.retriever = retriever
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.config = config or AgentConfig()

    def run(self, question: str, corpus: str | None = None, config: AgentConfig | None = None) -> dict[str, Any]:
        final: dict[str, Any] = {}
        for event in self.stream(question, corpus, config):
            if event["type"] == "final":
                final = event["result"]
        return final

    # ------------------------------------------------------------------ the agent loop

    def stream(self, question: str, corpus: str | None = None, config: AgentConfig | None = None) -> Iterator[dict]:
        cfg = config or self.config
        started = time.perf_counter()
        registry = EvidenceRegistry()
        state = _RunState()
        llm_calls_before = self.llm.stats.calls

        # Step 0: retrieve before asking the model anything.
        yield {"type": "tool_call", "step": 0, "tool": "search_documents", "args": {"query": question}}
        initial = self._search(question, corpus, None, None, None, registry, state, cfg, "initial search")
        yield {"type": "tool_result", "step": 0, "tool": "search_documents", **initial["event"]}

        hints = find_conflicts(registry.all())
        state.reported_hints = {h.describe() for h in hints}
        user_msg = f"Question: {question}\n\nInitial search results:\n{initial['text']}"
        if hints:
            user_msg += "\n\n" + _hints_text(hints)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": user_msg}]

        submitted: dict[str, Any] | None = None
        max_steps = 1 if cfg.mode == "single_shot" else cfg.max_steps
        try:
            for step in range(1, max_steps + 1):
                last = step == max_steps or _context_chars(messages) > cfg.max_context_chars
                tools = [SUBMIT_TOOL] if last else RESEARCH_TOOLS
                response = self.llm.chat(messages, tools=tools, tool_choice="required" if last else "auto")
                state.llm_trace.append({"step": step, "provider": response.provider, "model": response.model,
                                        "prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens,
                                        "latency_ms": round(response.latency_ms), "cached": response.cached})
                if response.content.strip() and response.tool_calls:
                    yield {"type": "thought", "step": step, "text": response.content.strip()[:500]}

                if not response.tool_calls:  # model answered in plain text
                    submitted = {"answer": response.content, "status": "answered", "claims": []}
                    break

                messages.append({
                    "role": "assistant", "content": response.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                   for tc in response.tool_calls],
                })
                for tc in response.tool_calls:
                    if tc.name == "submit_answer" and not tc.parse_error:
                        submitted = tc.arguments
                        break
                    yield {"type": "tool_call", "step": step, "tool": tc.name, "args": tc.arguments}
                    result = self._execute(tc.name, tc.arguments, tc.parse_error, corpus, registry, state, cfg)
                    yield {"type": "tool_result", "step": step, "tool": tc.name, **result["event"]}
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result["text"]})
                if submitted is not None:
                    break
        except LLMUnavailableError as exc:
            log.error("LLM unavailable: %s", exc)
            yield {"type": "final", "result": self._degraded_result(question, registry, state, str(exc), started)}
            return

        if submitted is None:
            submitted = {"answer": "", "status": "insufficient_evidence", "claims": [],
                         "missing_information": "The agent did not reach a conclusion within its step budget."}

        all_hints = find_conflicts(registry.all())
        verified = verify_answer(submitted, registry, all_hints, state.best_rerank, cfg.abstain_threshold, self.embedder,
                                 question=question)
        result = {
            "question": question,
            "corpus": corpus,
            **verified,
            "sources": [registry.get(sid).to_dict() for sid in verified["cited_ids"] if registry.get(sid)],
            "evidence": [ev.to_dict(max_chars=400) for ev in registry.all()],
            "trace": {"tool_steps": state.tool_trace, "llm_calls": state.llm_trace},
            "retrieval_confidence": state.best_rerank,
            "llm_calls": self.llm.stats.calls - llm_calls_before,
            "prompt_tokens": sum(c["prompt_tokens"] for c in state.llm_trace),
            "completion_tokens": sum(c["completion_tokens"] for c in state.llm_trace),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "mode": cfg.mode,
        }
        yield {"type": "final", "result": result}

    # ------------------------------------------------------------------ tools

    def _execute(self, name: str, args: dict[str, Any], parse_error: str | None, corpus: str | None,
                 registry: EvidenceRegistry, state: _RunState, cfg: AgentConfig) -> dict[str, Any]:
        if parse_error:
            return _tool_text(f"Error: {parse_error}. Call the tool again with valid JSON arguments.")
        try:
            if name == "search_documents":
                query = str(args.get("query", "")).strip()
                if not query:
                    return _tool_text("Error: 'query' is required.")
                res = self._search(query, corpus, args.get("doc_type"), args.get("date_from"), args.get("date_to"),
                                   registry, state, cfg, f"search: {query}")
            elif name == "find_mentions":
                res = self._find_mentions(str(args.get("entity", "")).strip(), corpus, registry, state, cfg)
            elif name == "read_document":
                res = self._read_document(str(args.get("doc_id", "")).strip(), registry, state, cfg)
            else:
                return _tool_text(f"Error: unknown tool '{name}'.")
        except Exception as exc:  # a failing tool must not crash the whole investigation
            log.exception("tool %s failed", name)
            return _tool_text(f"Error while running {name}: {type(exc).__name__}. Try a different query.")

        new_hints = [h for h in find_conflicts(registry.all()) if h.describe() not in state.reported_hints]
        if new_hints:
            state.reported_hints.update(h.describe() for h in new_hints)
            res["text"] += "\n\n" + _hints_text(new_hints)
        return res

    def _search(self, query, corpus, doc_type, date_from, date_to, registry, state, cfg, label) -> dict[str, Any]:
        t = time.perf_counter()
        doc_types = [doc_type] if doc_type else None
        result = self.retriever.search(query, corpus=corpus, doc_types=doc_types, date_from=date_from,
                                       date_to=date_to, config=cfg.retrieval)
        note = ""
        if not result.hits and (doc_types or date_from or date_to):
            result = self.retriever.search(query, corpus=corpus, config=cfg.retrieval)
            note = "(No results with those filters; showing unfiltered results.)\n"
        if result.best_rerank_score is not None:
            state.best_rerank = max(state.best_rerank or -1e9, result.best_rerank_score)
        evs = [registry.add(h.chunk, h.scores, found_by=label) for h in result.hits]
        state.tool_trace.append({"tool": "search_documents", "query": query, "filters": {"doc_type": doc_type,
                                 "date_from": date_from, "date_to": date_to}, "evidence_ids": [e.sid for e in evs],
                                 "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
                                 "latency_ms": round((time.perf_counter() - t) * 1000)})
        if not evs:
            text = f"No results for '{query}'."
        else:
            text = note + "\n".join(EvidenceRegistry.render(e, cfg.snippet_chars) for e in evs)
        return {"text": text, "event": {"evidence": [e.to_dict(max_chars=300) for e in evs]}}

    def _find_mentions(self, entity, corpus, registry, state, cfg) -> dict[str, Any]:
        if not entity:
            return _tool_text("Error: 'entity' is required.")
        normalized = entity.upper() if ID_RE.fullmatch(entity.upper()) else entity.lower()
        chunks = self.store.find_chunks(build_filter(corpus=corpus, entities=[normalized]), limit=24)
        chunks.sort(key=lambda c: (c.metadata.get("date_int") or 0, c.doc_id, c.position))
        # Keep at most 2 passages per document so one long document cannot crowd out the others.
        per_doc: dict[str, int] = {}
        picked = []
        for c in chunks:
            if per_doc.get(c.doc_id, 0) < 2:
                picked.append(c)
                per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        picked = picked[:10]
        evs = [registry.add(c, found_by=f"mentions: {normalized}") for c in picked]
        state.tool_trace.append({"tool": "find_mentions", "entity": normalized, "evidence_ids": [e.sid for e in evs]})
        if not evs:
            return {"text": f"No passages mention '{entity}'.", "event": {"evidence": []}}
        text = f"Passages mentioning {normalized} (sorted by date, {len(per_doc)} documents):\n"
        text += "\n".join(EvidenceRegistry.render(e, cfg.snippet_chars) for e in evs)
        return {"text": text, "event": {"evidence": [e.to_dict(max_chars=300) for e in evs]}}

    def _read_document(self, doc_id, registry, state, cfg) -> dict[str, Any]:
        chunks = self.store.get_doc_chunks(doc_id)[:10]
        if not chunks:
            return _tool_text(f"No document with doc_id '{doc_id}'. Use a doc_id exactly as shown in results.")
        evs = [registry.add(c, found_by=f"read: {doc_id}") for c in chunks]
        state.tool_trace.append({"tool": "read_document", "doc_id": doc_id, "evidence_ids": [e.sid for e in evs]})
        text = f"Full document {doc_id} ({len(chunks)} passages):\n"
        text += "\n".join(EvidenceRegistry.render(e, cfg.snippet_chars + 300) for e in evs)
        return {"text": text, "event": {"evidence": [e.to_dict(max_chars=300) for e in evs]}}

    # ------------------------------------------------------------------ degraded mode

    def _degraded_result(self, question, registry, state, error, started) -> dict[str, Any]:
        """LLM down: still return the retrieved evidence so the user can investigate manually."""
        evidence = [ev.to_dict() for ev in registry.all()][:6]
        return {
            "question": question, "status": "llm_unavailable",
            "answer": "The language model is temporarily unavailable (free-tier limits or provider outage). "
                      "The most relevant passages found by the search are listed as sources below.",
            "claims": [], "contradictions": [], "missing_information": "", "groundedness": 0.0,
            "cited_ids": [e["id"] for e in evidence], "warnings": [error[:300]], "sources": evidence,
            "evidence": evidence, "trace": {"tool_steps": state.tool_trace, "llm_calls": state.llm_trace},
            "retrieval_confidence": state.best_rerank, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_ms": round((time.perf_counter() - started) * 1000), "mode": "degraded",
        }


@dataclass
class _RunState:
    best_rerank: float | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    llm_trace: list[dict[str, Any]] = field(default_factory=list)
    reported_hints: set[str] = field(default_factory=set)


def _tool_text(text: str) -> dict[str, Any]:
    return {"text": text, "event": {"evidence": [], "message": text}}


def _hints_text(hints: list[ConflictHint]) -> str:
    lines = "\n".join(f"- {h.describe()}" for h in hints)
    return f"Automated conflict check - these sources disagree; verify and report them:\n{lines}"


def _context_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)
