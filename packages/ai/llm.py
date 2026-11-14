"""
ai.llm — Provider-agnostic language-model client with a reproducibility cache.

Providers (chosen by QUERYOPT_LLM_PROVIDER, default "auto"):

    anthropic   Claude via the Messages API           (ANTHROPIC_API_KEY)
    openai      Any OpenAI-compatible /chat/completions endpoint — OpenAI,
                Ollama (http://localhost:11434/v1), vLLM, LM Studio …
                (QUERYOPT_LLM_BASE_URL, optional OPENAI_API_KEY)
    offline     No model. Every AI feature has a deterministic fallback so the
                platform, its tests and its demo work without network access.

"auto" picks anthropic if ANTHROPIC_API_KEY is set, else openai if
QUERYOPT_LLM_BASE_URL is set, else offline.

Every completion is cached on disk keyed by (provider, model, system, prompt,
temperature). Research runs that involve an LLM (RQ5) are therefore replayable
bit-for-bit: re-running an experiment re-uses the recorded answers instead of
sampling new ones. Set QUERYOPT_LLM_CACHE=false to force fresh calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

_cache_lock = threading.Lock()


class LLMUnavailableError(RuntimeError):
    """Raised when an AI feature needs a model and none is configured."""


class LLMCallError(RuntimeError):
    """Raised when the provider returned an error or an unusable answer."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None

    def provenance(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        return d


# ── .env support (no extra dependency) ─────────────────────────────────────


def _load_dotenv_once() -> None:
    if getattr(_load_dotenv_once, "_done", False):
        return
    _load_dotenv_once._done = True  # type: ignore[attr-defined]
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


# ── Clients ────────────────────────────────────────────────────────────────


class LLMClient(ABC):
    provider: str = "abstract"
    model: str = ""

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "available": self.available}

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.0,
        use_cache: bool | None = None,
    ) -> LLMResponse:
        if not self.available:
            raise LLMUnavailableError(
                "No language model is configured. Set ANTHROPIC_API_KEY (Claude) or "
                "QUERYOPT_LLM_BASE_URL (any OpenAI-compatible server, e.g. Ollama)."
            )
        cache = _ResponseCache.instance() if _cache_enabled(use_cache) else None
        key = _cache_key(self.provider, self.model, system, prompt, temperature, max_tokens)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return LLMResponse(**{**hit, "cached": True})
        t0 = time.perf_counter()
        resp = self._call(system, prompt, max_tokens, temperature)
        resp.latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if cache is not None:
            cache.put(key, asdict(resp))
        return resp

    @abstractmethod
    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> LLMResponse: ...


class OfflineClient(LLMClient):
    provider = "offline"
    model = "none"

    @property
    def available(self) -> bool:
        return False

    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        raise LLMUnavailableError("offline")


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, model: str | None = None, url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.url = url or ANTHROPIC_URL

    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        import httpx

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        try:
            r = httpx.post(self.url, json=body, headers=headers, timeout=90.0)
        except httpx.HTTPError as exc:
            raise LLMCallError(f"Could not reach Anthropic API: {exc}") from exc
        if r.status_code >= 400:
            raise LLMCallError(f"Anthropic API error {r.status_code}: {r.text[:400]}")
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=data.get("model", self.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


class OpenAICompatibleClient(LLMClient):
    provider = "openai"

    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or DEFAULT_OPENAI_MODEL
        self.api_key = api_key

    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        import httpx

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=120.0)
        except httpx.HTTPError as exc:
            raise LLMCallError(f"Could not reach {self.base_url}: {exc}") from exc
        if r.status_code >= 400:
            raise LLMCallError(f"LLM endpoint error {r.status_code}: {r.text[:400]}")
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            provider=self.provider,
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def get_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Build the configured client. Never raises; returns OfflineClient instead."""
    _load_dotenv_once()
    provider = (provider or os.environ.get("QUERYOPT_LLM_PROVIDER", "auto")).lower()
    model = model or os.environ.get("QUERYOPT_LLM_MODEL") or None
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    base_url = os.environ.get("QUERYOPT_LLM_BASE_URL", "").strip()

    if provider == "auto":
        provider = "anthropic" if anthropic_key else ("openai" if base_url else "offline")
    if provider == "anthropic" and anthropic_key:
        return AnthropicClient(anthropic_key, model)
    if provider == "openai" and base_url:
        return OpenAICompatibleClient(base_url, model, os.environ.get("OPENAI_API_KEY"))
    return OfflineClient()


# ── Cache ─────────────────────────────────────────────────────────────────


def _cache_enabled(override: bool | None) -> bool:
    if override is not None:
        return override
    return os.environ.get("QUERYOPT_LLM_CACHE", "true").lower() not in ("0", "false", "no", "off")


def _cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class _ResponseCache:
    _inst: _ResponseCache | None = None

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    @classmethod
    def instance(cls) -> _ResponseCache:
        from common.config import get_config

        path = Path(os.environ.get("QUERYOPT_LLM_CACHE_PATH") or Path(get_config().data_dir) / "llm_cache.json")
        if cls._inst is None or cls._inst.path != path:
            cls._inst = cls(path)
        return cls._inst

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with _cache_lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
            tmp.replace(self.path)


# ── Parsing helpers ───────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model answer (fenced or bare)."""
    m = _JSON_BLOCK.search(text)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise LLMCallError("Model answer did not contain a JSON object.")


_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    m = _SQL_BLOCK.search(text)
    sql = m.group(1) if m else text
    return sql.strip().rstrip(";").strip()
