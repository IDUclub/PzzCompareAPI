"""Synchronous client for the OpenAI-compatible embeddings service (vectorizer).

Used by the deterministic building runner to semantically match free-text
type/service names to the small type/service catalogue, so a human-written name
(«автомойка самообслуживания») can still resolve to a catalogue id. The runner
executes in a worker thread with no event loop, so this client is plain-sync httpx.

Endpoint contract (``POST {url}``): ``{"input": [str, ...], "model": name}`` ->
``{"data": [{"embedding": [float, ...]}, ...]}`` (OpenAI ``/v1/embeddings`` shape).
"""

from __future__ import annotations

import httpx


class EmbeddingsError(RuntimeError):
    """The vectorizer was unreachable or returned an unusable response."""


class EmbeddingsClient:
    """Minimal sync embeddings client with fixed-size batching."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not url:
            raise EmbeddingsError("vectorizer_url is not configured")
        self._url = url
        self._model = model
        self._batch_size = max(1, batch_size)
        self._timeout = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (order preserved)."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self._timeout) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                vectors.extend(self._embed_batch(client, batch))
        return vectors

    def _embed_batch(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        payload = {"input": batch, "model": self._model}
        try:
            resp = client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingsError(f"vectorizer request failed: {exc}") from exc
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(batch):
            raise EmbeddingsError("vectorizer response shape unexpected")
        out: list[list[float]] = []
        for item in items:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list) or not vec:
                raise EmbeddingsError("vectorizer returned an empty embedding")
            out.append(vec)
        return out
