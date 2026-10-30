"""Priority reaches vLLM for pipeline calls and survives subprocess settings."""

from types import SimpleNamespace

import pytest

from pipeline_modules.business import clients
from service.settings import _build_settings_cached


@pytest.mark.parametrize("priority", [-60, -20, 0])
def test_pipeline_factory_sends_priority_for_classification_and_rerank(
    monkeypatch, priority
):
    seen = []
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    original_optional = clients._config_optional
    monkeypatch.setattr(
        clients,
        "_config_optional",
        lambda key: (
            str(priority) if key == "VLLM_REQUEST_PRIORITY" else original_optional(key)
        ),
    )

    def post(url, **kwargs):
        seen.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "{}"}}]},
        )

    monkeypatch.setattr(
        clients, "get_shared_vllm_session", lambda: SimpleNamespace(post=post)
    )
    for token_budget in (256, 512):
        client = clients.build_llm_client(
            backend="vllm",
            timeout=5,
            default_model="test-model",
            temperature=0,
            num_ctx=1024,
            num_predict=token_budget,
            think=None,
        )
        assert client.complete_json("question", "system", {"type": "object"}) == {}
    assert [body["priority"] for body in seen] == [priority, priority]


def test_settings_load_priorities_from_config(monkeypatch):
    monkeypatch.setenv("VLLM_REQUEST_PRIORITY", "-20")
    monkeypatch.setenv("VLLM_CHAT_PRIORITY", "-80")
    _build_settings_cached.cache_clear()
    try:
        settings = _build_settings_cached()
        assert settings.vllm_request_priority == -20
        assert settings.vllm_chat_priority == -80
    finally:
        _build_settings_cached.cache_clear()
