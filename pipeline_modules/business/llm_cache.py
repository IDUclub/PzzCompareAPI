"""Disk-backed cache of LLM structured answers, shared by every pipeline process.

The zone check asks the model one question per distinct pair of parcel VRI
wording and zone regulation, so a run is already deduplicated in memory. That
in-memory map dies with the subprocess, while the questions themselves repeat:
re-running a municipality asks the very same things again. The backend is also
not reproducible — at ``temperature=0`` continuous batching still changes the
answer for a small share of questions between identical runs, which makes any
before/after quality comparison unreliable.

Entries are keyed by the *rendered question* — model, decoding parameters,
system prompt, response schema and user prompt — never by the input file or by
the caller's own dedup key. Two municipalities that both have a zone ``Ж-1``
carry different regulation text in the prompt, so they cannot collide; changing
the prompt template, the model or the schema misses instead of returning an
answer to a question nobody asked.

One JSON file per answer under ``LLM_CACHE_DIR`` (unset disables the cache),
written atomically so concurrent workers share the directory without locking.
``LLM_CACHE_TTL_DAYS`` (0 = keep forever) bounds how long a wrong answer can
survive, and ``CACHE_FORMAT_VERSION`` invalidates everything at once.

Each entry keeps the whole question it answers, not only the answer: prompts,
schema and decoding parameters. A reused verdict is otherwise unauditable —
there is no way to see what the model was asked, to re-ask it, or to tell a
stale answer from a wrong one. ``iter_entries`` walks them for exactly that,
and an entry carries everything ``complete_json`` needs to ask again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("pipeline_modules")

CACHE_ENV_VAR = "LLM_CACHE_DIR"
TTL_ENV_VAR = "LLM_CACHE_TTL_DAYS"
CACHE_FORMAT_VERSION = "v3"

_ENTRY_SUFFIX = ".json"
_KEY_SEPARATOR = "\x1f"
_SECONDS_PER_DAY = 86400.0

_instance_lock = threading.Lock()
_instance: Optional["LLMCache"] = None
_instance_root: Optional[str] = None


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0


_stats_lock = threading.Lock()
_stats = CacheStats()


def reset_stats() -> None:
    global _stats
    with _stats_lock:
        _stats = CacheStats()


def stats_snapshot() -> CacheStats:
    with _stats_lock:
        return CacheStats(**vars(_stats))


def format_summary() -> str:
    taken = stats_snapshot()
    looked_up = taken.hits + taken.misses
    if not looked_up:
        return "disabled"
    return (
        f"hits={taken.hits} misses={taken.misses} stores={taken.stores} "
        f"hit_rate={100 * taken.hits / looked_up:.0f}%"
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class LLMCache:
    """Structured LLM answers under one cache directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._writable = True

    def build_key(
        self,
        *,
        model: str,
        client_fingerprint: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        think_override: Any,
    ) -> str:
        material = _KEY_SEPARATOR.join(
            (
                CACHE_FORMAT_VERSION,
                model,
                _canonical(client_fingerprint),
                _canonical(think_override),
                system_prompt,
                _canonical(schema),
                user_prompt,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._count(hit=False)
            return None
        except (OSError, ValueError) as exc:
            # A truncated or half-written entry is indistinguishable from a miss:
            # answering it again overwrites the damaged file.
            logger.debug("LLM cache read failed for %s: %s", path, exc)
            self._count(hit=False)
            return None
        if self._is_expired(payload.get("created")):
            self._count(hit=False)
            return None
        value = payload.get("value")
        if not isinstance(value, dict):
            self._count(hit=False)
            return None
        self._count(hit=True)
        return value

    def put(
        self,
        key: str,
        *,
        model: str,
        decoding: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        think_override: Any,
        value: dict[str, Any],
    ) -> None:
        if not self._writable:
            return
        path = self._path(key)
        if self._holds_answer(path):
            # Two workers can miss the same key and answer it concurrently. Keep
            # whichever answer landed first so the entry stops moving, while a
            # damaged file still reads as a miss and gets overwritten below.
            return
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "created": time.time(),
            "model": model,
            "decoding": decoding,
            "think_override": think_override,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
            "value": value,
        }
        temp_path: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".tmp",
                delete=False,
                mode="w",
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False)
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        except OSError as exc:
            self._writable = False
            logger.warning(
                "LLM cache at %s is not writable (%s); running without it",
                self._root,
                exc,
            )
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    logger.debug("Could not remove cache temp file %s", temp_path)
            return
        with _stats_lock:
            _stats.stores += 1

    def iter_entries(self) -> Iterator[dict[str, Any]]:
        """Yield every readable entry, question included, for auditing and re-asking."""
        for path in sorted(self._root.rglob(f"*{_ENTRY_SUFFIX}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.debug("Skipping unreadable LLM cache entry %s", path)
                continue
            if isinstance(payload, dict) and isinstance(payload.get("value"), dict):
                yield {**payload, "path": str(path)}

    @staticmethod
    def _holds_answer(path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return False
        return isinstance(payload.get("value"), dict)

    @staticmethod
    def _is_expired(created: Any) -> bool:
        # Read per lookup, not per instance: a runtime config override changes
        # the TTL fleet-wide without restarting the workers.
        ttl_seconds = _ttl_days() * _SECONDS_PER_DAY
        if not ttl_seconds:
            return False
        if not isinstance(created, (int, float)):
            return True
        return (time.time() - float(created)) > ttl_seconds

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}{_ENTRY_SUFFIX}"

    @staticmethod
    def _count(*, hit: bool) -> None:
        with _stats_lock:
            if hit:
                _stats.hits += 1
            else:
                _stats.misses += 1


def _ttl_days() -> float:
    raw = (os.getenv(TTL_ENV_VAR) or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("%s=%r is not a number; caching answers forever", TTL_ENV_VAR, raw)
        return 0.0


def get_llm_cache() -> Optional[LLMCache]:
    """Cache instance, or None when ``LLM_CACHE_DIR`` is not configured."""
    global _instance, _instance_root
    root = (os.getenv(CACHE_ENV_VAR) or "").strip()
    if not root:
        return None
    with _instance_lock:
        if _instance is None or _instance_root != root:
            _instance = LLMCache(root)
            _instance_root = root
            logger.info("LLM answer cache enabled at %s", root)
        return _instance


_FINGERPRINT_ATTRS = (
    "mode",
    "temperature",
    "num_ctx",
    "num_predict",
    "max_tokens",
    "think",
    "runtime_presets",
)


class CachingLLMClient:
    """Wraps an LLM client so identical questions are answered from disk.

    Only successful answers are stored: a failed call raises through untouched,
    so a backend outage never freezes into a cached verdict.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._fingerprint = {
            "client": type(inner).__name__,
            **{
                name: getattr(inner, name)
                for name in _FINGERPRINT_ATTRS
                if hasattr(inner, name)
            },
        }

    def complete_json(
        self,
        user_prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        model: Optional[str] = None,
        think_override: Any = None,
    ) -> dict[str, Any]:
        cache = get_llm_cache()
        selected_model = model or getattr(self._inner, "default_model", "")
        if cache is None or not selected_model:
            return self._inner.complete_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=schema,
                model=model,
                think_override=think_override,
            )
        key = cache.build_key(
            model=selected_model,
            client_fingerprint=self._fingerprint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            think_override=think_override,
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        answer = self._inner.complete_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema,
            model=model,
            think_override=think_override,
        )
        if isinstance(answer, dict):
            cache.put(
                key,
                model=selected_model,
                decoding=self._fingerprint,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                think_override=think_override,
                value=answer,
            )
        return answer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
