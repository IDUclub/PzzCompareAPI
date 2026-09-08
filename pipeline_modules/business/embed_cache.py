"""Disk-backed cache of embedding vectors, shared by every pipeline process.

The pipeline runs as a short-lived subprocess per task, so an in-process cache
dies with the task while the texts it embedded repeat on every run: the whole
Rosreestr VRI catalogue is static, and the VRI phrasings of uploaded parcels
recur across municipalities. The embedder is the pipeline's bottleneck and does
not scale with batch size or parallelism, so the only lever is embedding fewer
texts.

One file per vector under ``EMBED_CACHE_DIR`` (unset disables the cache),
written atomically so concurrent workers can share the directory without
locking. The key covers the model name, so two models never collide, and a
format version, so a change in what is stored invalidates old entries instead of
reading them back as garbage.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger("pipeline_modules")

CACHE_ENV_VAR = "EMBED_CACHE_DIR"
CACHE_FORMAT_VERSION = "v1"

_VECTOR_DTYPE = "<f4"
_ENTRY_SUFFIX = ".f32"
_KEY_SEPARATOR = "\x1f"
_UNSAFE_MODEL_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

Vector = Sequence[float]
FetchVectors = Callable[[list[str]], list[Vector]]

_instances: dict[tuple[str, str], "EmbeddingCache"] = {}
_instances_lock = threading.Lock()


def _unique(texts: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(texts))


def _model_dir_name(model: str) -> str:
    """Readable per-model directory so a stale model's vectors can be dropped by hand."""
    slug = _UNSAFE_MODEL_CHARS.sub("-", model).strip("-")[:48]
    return f"{slug}-{hashlib.sha256(model.encode('utf-8')).hexdigest()[:8]}"


class EmbeddingCache:
    """Vectors for a single embedding model under one cache directory."""

    def __init__(self, root: str | os.PathLike[str], model: str) -> None:
        self._model = model
        self._root = Path(root) / _model_dir_name(model)
        self._writable = True

    def resolve(self, texts: Sequence[str], fetch: FetchVectors) -> list[Vector]:
        """Return one vector per text, embedding only what the cache is missing."""
        if not texts:
            return []
        vectors: list[Optional[Vector]] = [self._read(text) for text in texts]
        missing = _unique(
            [text for text, vector in zip(texts, vectors) if vector is None]
        )
        if missing:
            fresh = self._fetch_and_store(missing, fetch)
            vectors = [
                vector if vector is not None else fresh[text]
                for text, vector in zip(texts, vectors)
            ]
        resolved: list[Vector] = [vector for vector in vectors if vector is not None]
        if len({len(vector) for vector in resolved}) > 1:
            # The endpoint now answers with a different vector width under the same
            # model name. Cached entries are unusable — refill them from the server.
            logger.warning(
                "Embedding cache holds a different vector width for model %s; refetching",
                self._model,
            )
            fresh = self._fetch_and_store(_unique(texts), fetch)
            return [fresh[text] for text in texts]
        logger.debug(
            "Embedding cache: %d hit(s), %d miss(es), model=%s",
            len(texts) - len(missing),
            len(missing),
            self._model,
        )
        return resolved

    def _fetch_and_store(
        self, texts: list[str], fetch: FetchVectors
    ) -> dict[str, Vector]:
        fetched = fetch(texts)
        if len(fetched) != len(texts):
            raise ValueError(
                f"Unexpected embedding count: expected={len(texts)}, got={len(fetched)}"
            )
        for text, vector in zip(texts, fetched):
            self._write(text, vector)
        return dict(zip(texts, fetched))

    def _path(self, text: str) -> Path:
        key = _KEY_SEPARATOR.join((CACHE_FORMAT_VERSION, self._model, text))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._root / digest[:2] / f"{digest}{_ENTRY_SUFFIX}"

    def _read(self, text: str) -> Optional[np.ndarray]:
        path = self._path(text)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.debug("Embedding cache read failed for %s: %s", path, exc)
            return None
        if not payload or len(payload) % np.dtype(_VECTOR_DTYPE).itemsize:
            return None
        return np.frombuffer(payload, dtype=_VECTOR_DTYPE).copy()

    def _write(self, text: str, vector: Vector) -> None:
        if not self._writable:
            return
        path = self._path(text)
        payload = np.asarray(vector, dtype=_VECTOR_DTYPE).tobytes()
        temp_path: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        except OSError as exc:
            self._writable = False
            logger.warning(
                "Embedding cache at %s is not writable (%s); running without it",
                self._root,
                exc,
            )
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    logger.debug("Could not remove cache temp file %s", temp_path)


def get_embedding_cache(model: str) -> Optional[EmbeddingCache]:
    """Cache for ``model``, or None when ``EMBED_CACHE_DIR`` is not configured."""
    root = (os.getenv(CACHE_ENV_VAR) or "").strip()
    if not root or not model:
        return None
    key = (root, model)
    with _instances_lock:
        cache = _instances.get(key)
        if cache is None:
            cache = EmbeddingCache(root, model)
            _instances[key] = cache
            logger.info("Embedding cache enabled at %s (model=%s)", root, model)
    return cache
