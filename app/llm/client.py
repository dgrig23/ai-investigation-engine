"""Provider-agnostic LLM client with failure handling.

All supported providers (Groq, Cerebras, Mistral, Gemini) expose an OpenAI-compatible
`/chat/completions` endpoint with tool calling, so one SDK covers them; only base URL, key and
model differ. Free tiers are rate-limited, so the client is built to degrade gracefully:

- Retries with exponential backoff + jitter for timeouts, connection errors and 5xx.
- 429 rate limits: honour `Retry-After` when it is short, otherwise fail over to the next provider.
- Circuit breaker: a provider that keeps failing (or rejects our key) is skipped for a cool-down.
- Response cache (SQLite): identical requests are never paid twice - makes evaluation reproducible
  and cheap to re-run.
- Every call records provider, model, tokens and latency for observability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import openai

from app.config import Settings

log = logging.getLogger(__name__)

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class LLMError(Exception):
    pass


class LLMUnavailableError(LLMError):
    """Every configured provider failed (or none is configured)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


@dataclass
class Provider:
    name: str
    model: str
    client: openai.OpenAI
    failures: int = 0
    open_until: float = 0.0  # circuit breaker: skip this provider until this timestamp


@dataclass
class CallStats:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    providers: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_json_lenient(raw: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in code fences or add trailing commas; repair the common cases."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        match = re.search(r"\{.*\}", repaired, re.S)
        value = json.loads(match.group(0) if match else repaired)
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


class ResponseCache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), time.time()))
            self._conn.commit()


class LLMClient:
    def __init__(self, settings: Settings, providers: list[str] | None = None, use_cache: bool = True):
        self.settings = settings
        names = providers or [p.strip() for p in settings.llm_providers.split(",") if p.strip()]
        self.providers: list[Provider] = []
        for name in names:
            key = getattr(settings, f"{name}_api_key", None)
            if name not in PROVIDER_BASE_URLS or key is None or not key.get_secret_value():
                continue
            client = openai.OpenAI(
                api_key=key.get_secret_value(), base_url=PROVIDER_BASE_URLS[name],
                timeout=settings.llm_timeout_s, max_retries=0,  # we do retries ourselves
            )
            self.providers.append(Provider(name=name, model=getattr(settings, f"{name}_model"), client=client))
        self.cache = ResponseCache(settings.llm_cache_path) if use_cache else None
        self.stats = CallStats()
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def describe(self) -> list[dict[str, str]]:
        return [{"provider": p.name, "model": p.model} for p in self.providers]

    # ------------------------------------------------------------------ main entry point

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             tool_choice: str = "auto", max_tokens: int | None = None) -> LLMResponse:
        if not self.providers:
            raise LLMUnavailableError("No LLM provider configured. Add at least one API key to .env.")
        errors: list[str] = []
        for provider in self.providers:
            if provider.open_until > time.time():
                errors.append(f"{provider.name}: circuit open")
                continue
            key = self._cache_key(provider, messages, tools, tool_choice)
            if self.cache and (hit := self.cache.get(key)):
                with self._lock:
                    self.stats.calls += 1
                    self.stats.cached_calls += 1
                return self._from_cache(hit)
            try:
                response = self._call_with_retries(provider, messages, tools, tool_choice, max_tokens)
            except LLMError as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            if self.cache:
                self.cache.put(key, asdict(response))
            return response
        with self._lock:
            self.stats.errors.extend(errors)
        raise LLMUnavailableError("All LLM providers failed: " + " | ".join(errors))

    # ------------------------------------------------------------------ internals

    def _call_with_retries(self, p: Provider, messages, tools, tool_choice, max_tokens) -> LLMResponse:
        attempts = self.settings.llm_max_retries
        choice = tool_choice
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": p.model, "messages": messages, "temperature": self.settings.llm_temperature,
                    "max_tokens": max_tokens or self.settings.llm_max_output_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = choice
                raw = p.client.chat.completions.create(**kwargs)
                p.failures = 0
                return self._parse(p, raw, (time.perf_counter() - started) * 1000)
            except openai.RateLimitError as exc:
                wait = self._retry_after(exc)
                log.warning("%s rate limited (retry-after=%s)", p.name, wait)
                if wait is None or wait > 20 or attempt == attempts - 1:
                    self._trip(p, cooldown=min(wait or 60, 300))
                    raise LLMError(f"rate limited ({wait or '?'}s)") from exc
                time.sleep(wait)
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
                self._trip(p, cooldown=3600)  # bad key or unknown model: no point retrying soon
                raise LLMError(f"rejected: {type(exc).__name__}") from exc
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if tools and choice != "auto" and "tool" in msg:
                    choice = "auto"  # some providers do not support forced tool choice
                    continue
                raise LLMError(f"bad request: {str(exc)[:200]}") from exc
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                if attempt == attempts - 1:
                    self._trip(p, cooldown=60)
                    raise LLMError(f"unavailable: {type(exc).__name__}") from exc
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                log.warning("%s %s, retrying in %.1fs", p.name, type(exc).__name__, backoff)
                time.sleep(backoff)
            except openai.APIError as exc:
                raise LLMError(f"api error: {str(exc)[:200]}") from exc
        raise LLMError("retries exhausted")

    def _trip(self, p: Provider, cooldown: float) -> None:
        p.failures += 1
        p.open_until = time.time() + cooldown
        log.warning("circuit opened for %s for %.0fs", p.name, cooldown)

    @staticmethod
    def _retry_after(exc: openai.RateLimitError) -> float | None:
        try:
            header = exc.response.headers.get("retry-after")
            return float(header) if header is not None else None
        except (AttributeError, ValueError):
            return None

    def _parse(self, p: Provider, raw, latency_ms: float) -> LLMResponse:
        message = raw.choices[0].message
        calls = []
        for i, tc in enumerate(message.tool_calls or []):
            raw_args = tc.function.arguments or "{}"
            try:
                args, err = parse_json_lenient(raw_args), None
            except (ValueError, json.JSONDecodeError) as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            # Portable ids: some providers (e.g. Mistral) require exactly 9 alphanumeric characters.
            call_id = hashlib.sha1(f"{tc.id}{i}".encode()).hexdigest()[:9]
            calls.append(ToolCall(id=call_id, name=tc.function.name, arguments=args, raw_arguments=raw_args, parse_error=err))
        usage = raw.usage
        response = LLMResponse(
            content=message.content or "", tool_calls=calls, provider=p.name, model=p.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0, latency_ms=latency_ms,
        )
        with self._lock:
            self.stats.calls += 1
            self.stats.prompt_tokens += response.prompt_tokens
            self.stats.completion_tokens += response.completion_tokens
            self.stats.providers[p.name] = self.stats.providers.get(p.name, 0) + 1
        return response

    @staticmethod
    def _from_cache(data: dict[str, Any]) -> LLMResponse:
        calls = [ToolCall(**tc) for tc in data.pop("tool_calls", [])]
        data["cached"] = True
        return LLMResponse(tool_calls=calls, **data)

    @staticmethod
    def _cache_key(p: Provider, messages, tools, tool_choice) -> str:
        blob = json.dumps({"m": p.model, "msgs": messages, "tools": tools, "tc": tool_choice}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()
