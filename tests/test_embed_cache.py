from __future__ import annotations

import sys
import types

import numpy as np
import pytest

fake = types.ModuleType("iduconfig")


class _Config:
    def get(self, *_args, **_kwargs):
        return "1"


fake.Config = _Config
sys.modules.setdefault("iduconfig", fake)

from pipeline_modules.business import embed_cache as embed_cache_module
from pipeline_modules.business.embed_cache import EmbeddingCache, get_embedding_cache


class FetchSpy:
    """Deterministic stand-in for the embedding endpoint."""

    def __init__(self, dimension: int = 4) -> None:
        self.calls: list[list[str]] = []
        self.dimension = dimension

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text) + index)] * self.dimension for index, text in enumerate(texts)]

    @property
    def embedded(self) -> list[str]:
        return [text for call in self.calls for text in call]


def test_second_run_reuses_stored_vectors(tmp_path) -> None:
    texts = ["для индивидуального жилищного строительства", "магазины"]
    first_fetch = FetchSpy()
    first = EmbeddingCache(tmp_path, "model-a").resolve(texts, first_fetch)

    second_fetch = FetchSpy()
    second = EmbeddingCache(tmp_path, "model-a").resolve(texts, second_fetch)

    assert second_fetch.calls == []
    assert [np.asarray(vector).tolist() for vector in second] == [
        np.asarray(vector).tolist() for vector in first
    ]


def test_only_missing_texts_are_embedded(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path, "model-a")
    cache.resolve(["a", "b"], FetchSpy())

    fetch = FetchSpy()
    vectors = cache.resolve(["a", "b", "c"], fetch)

    assert fetch.calls == [["c"]]
    assert len(vectors) == 3


def test_duplicate_texts_are_embedded_once(tmp_path) -> None:
    fetch = FetchSpy()
    vectors = EmbeddingCache(tmp_path, "model-a").resolve(["a", "a", "b"], fetch)

    assert fetch.calls == [["a", "b"]]
    assert np.asarray(vectors[0]).tolist() == np.asarray(vectors[1]).tolist()


def test_vectors_of_another_model_are_not_reused(tmp_path) -> None:
    EmbeddingCache(tmp_path, "model-a").resolve(["a"], FetchSpy())

    fetch = FetchSpy()
    EmbeddingCache(tmp_path, "model-b").resolve(["a"], fetch)

    assert fetch.calls == [["a"]]


def test_truncated_entry_is_refetched(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path, "model-a")
    cache.resolve(["a"], FetchSpy())
    entry = next(tmp_path.rglob("*.f32"))
    entry.write_bytes(b"\x00\x01\x02")

    fetch = FetchSpy()
    vectors = cache.resolve(["a"], fetch)

    assert fetch.calls == [["a"]]
    assert len(vectors[0]) == 4


def test_changed_vector_width_refills_the_cache(tmp_path) -> None:
    EmbeddingCache(tmp_path, "model-a").resolve(["a"], FetchSpy(dimension=4))

    wider = FetchSpy(dimension=6)
    vectors = EmbeddingCache(tmp_path, "model-a").resolve(["a", "b"], wider)

    assert wider.embedded == ["b", "a", "b"]
    assert {len(vector) for vector in vectors} == {6}


def test_short_backend_response_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        EmbeddingCache(tmp_path, "model-a").resolve(["a", "b"], lambda texts: [[1.0]])


def test_unwritable_directory_does_not_break_the_run(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")

    fetch = FetchSpy()
    vectors = EmbeddingCache(blocked, "model-a").resolve(["a"], fetch)

    assert fetch.calls == [["a"]]
    assert len(vectors[0]) == 4


def test_cache_is_disabled_without_a_directory(monkeypatch) -> None:
    monkeypatch.setattr(embed_cache_module, "_instances", {})
    monkeypatch.delenv(embed_cache_module.CACHE_ENV_VAR, raising=False)
    assert get_embedding_cache("model-a") is None

    monkeypatch.setenv(embed_cache_module.CACHE_ENV_VAR, "   ")
    assert get_embedding_cache("model-a") is None


def test_vectorizer_client_embeds_each_text_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(embed_cache_module, "_instances", {})
    monkeypatch.setenv(embed_cache_module.CACHE_ENV_VAR, str(tmp_path))

    from pipeline_modules.business.clients import VectorizerClient

    fetch = FetchSpy()
    client = VectorizerClient(url="http://vectorizer.test/v1/embeddings", model="model-a")
    monkeypatch.setattr(
        VectorizerClient, "_embed_in_batches", lambda self, texts, batch_size: fetch(texts)
    )

    first = client.embed_many(["магазины", "склады"])
    second = client.embed_many(["магазины", "склады"])

    assert fetch.calls == [["магазины", "склады"]]
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(second, axis=1), 1.0)
