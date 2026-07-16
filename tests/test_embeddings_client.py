"""Tests for the sync embeddings (vectorizer) client."""
import json

import httpx

import service.infrastructure.embeddings_client as mod
from service.infrastructure.embeddings_client import EmbeddingsClient, EmbeddingsError


def _mock_httpx(handler, monkeypatch):
    real_client = httpx.Client

    def factory(**kw):
        kw.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(mod.httpx, "Client", factory)


def test_embed_returns_vectors(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "m" and body["input"] == ["a", "b"]
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]})

    _mock_httpx(handler, monkeypatch)
    c = EmbeddingsClient(url="http://vec/v1/embeddings", model="m", batch_size=10)
    assert c.embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_batches(monkeypatch):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(len(body["input"]))
        return httpx.Response(200, json={"data": [{"embedding": [0.0]} for _ in body["input"]]})

    _mock_httpx(handler, monkeypatch)
    c = EmbeddingsClient(url="http://vec.test/e", model="m", batch_size=2)
    out = c.embed(["a", "b", "c"])
    assert len(out) == 3
    assert seen == [2, 1]


def test_embed_http_error_raises(monkeypatch):
    _mock_httpx(lambda req: httpx.Response(500, text="boom"), monkeypatch)
    c = EmbeddingsClient(url="http://vec.test/e", model="m")
    try:
        c.embed(["a"])
    except EmbeddingsError:
        pass
    else:
        raise AssertionError("expected EmbeddingsError")


def test_embed_shape_mismatch_raises(monkeypatch):
    _mock_httpx(lambda req: httpx.Response(200, json={"data": []}), monkeypatch)
    c = EmbeddingsClient(url="http://vec.test/e", model="m")
    try:
        c.embed(["a"])
    except EmbeddingsError:
        pass
    else:
        raise AssertionError("expected EmbeddingsError")


def test_empty_input_no_call():
    assert EmbeddingsClient(url="http://vec.test/e", model="m").embed([]) == []


def test_missing_url_raises():
    try:
        EmbeddingsClient(url="", model="m")
    except EmbeddingsError:
        pass
    else:
        raise AssertionError("expected EmbeddingsError")
