"""Disk cache of LLM answers: keyed on the question, never on the input file."""

import json
from typing import Any, Optional

import pytest

from pipeline_modules.business import llm_cache
from pipeline_modules.business.llm_cache import CachingLLMClient

SYSTEM = "Ты проверяешь ВРИ участка."
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}}

GATCHINA_PROMPT = (
    "Код фактической зоны ПЗЗ: Ж-1\n"
    "Полное описание зоны: зона застройки индивидуальными жилыми домами Гатчины\n"
    "Кадастровый ВРИ: для индивидуального жилищного строительства\n"
)
DOLINSK_PROMPT = (
    "Код фактической зоны ПЗЗ: Ж-1\n"
    "Полное описание зоны: зона многоэтажной жилой застройки Долинска\n"
    "Кадастровый ВРИ: для индивидуального жилищного строительства\n"
)


class FakeLLM:
    """Answers every question with a different payload, so reuse is observable."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.default_model = "gpt-oss-20b"
        self.temperature = 0.0
        self.max_tokens = 1024
        self.think = "auto"

    def complete_json(
        self,
        user_prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        model: Optional[str] = None,
        think_override: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(user_prompt)
        return {"verdict": "allowed_main", "reason": f"answer #{len(self.calls)}"}


class FailingLLM(FakeLLM):
    def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("failed")
        raise RuntimeError("backend is down")


def ask(client: CachingLLMClient, prompt: str, **kwargs: Any) -> dict[str, Any]:
    return client.complete_json(
        user_prompt=prompt, system_prompt=SYSTEM, schema=SCHEMA, **kwargs
    )


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(llm_cache.TTL_ENV_VAR, raising=False)
    monkeypatch.setenv(llm_cache.CACHE_ENV_VAR, str(tmp_path / "llm_cache"))
    llm_cache.reset_stats()
    yield
    llm_cache.reset_stats()


def test_disabled_without_directory(monkeypatch):
    monkeypatch.delenv(llm_cache.CACHE_ENV_VAR, raising=False)
    inner = FakeLLM()
    client = CachingLLMClient(inner)

    ask(client, GATCHINA_PROMPT)
    ask(client, GATCHINA_PROMPT)

    assert len(inner.calls) == 2
    assert llm_cache.format_summary() == "disabled"


def test_repeated_question_is_answered_from_disk():
    inner = FakeLLM()
    client = CachingLLMClient(inner)

    first = ask(client, GATCHINA_PROMPT)
    second = ask(client, GATCHINA_PROMPT)

    assert len(inner.calls) == 1
    assert first == second


def test_answer_survives_a_new_process():
    first_inner = FakeLLM()
    first = ask(CachingLLMClient(first_inner), GATCHINA_PROMPT)

    second_inner = FakeLLM()
    second = ask(CachingLLMClient(second_inner), GATCHINA_PROMPT)

    assert second_inner.calls == []
    assert second == first


def test_same_zone_code_in_two_municipalities_does_not_collide():
    inner = FakeLLM()
    client = CachingLLMClient(inner)

    gatchina = ask(client, GATCHINA_PROMPT)
    dolinsk = ask(client, DOLINSK_PROMPT)

    assert len(inner.calls) == 2
    assert gatchina != dolinsk


def test_deeper_reasoning_is_a_different_question():
    inner = FakeLLM()
    client = CachingLLMClient(inner)

    ask(client, GATCHINA_PROMPT, think_override=None)
    ask(client, GATCHINA_PROMPT, think_override="medium")

    assert len(inner.calls) == 2


def test_another_model_is_a_different_question():
    inner = FakeLLM()
    client = CachingLLMClient(inner)

    ask(client, GATCHINA_PROMPT, model="gpt-oss-20b")
    ask(client, GATCHINA_PROMPT, model="qwen3")

    assert len(inner.calls) == 2


def test_decoding_parameters_are_part_of_the_key():
    cold = FakeLLM()
    ask(CachingLLMClient(cold), GATCHINA_PROMPT)

    warmer = FakeLLM()
    warmer.temperature = 0.7
    ask(CachingLLMClient(warmer), GATCHINA_PROMPT)

    assert warmer.calls != []


def test_failed_call_is_not_cached():
    failing = FailingLLM()
    with pytest.raises(RuntimeError):
        ask(CachingLLMClient(failing), GATCHINA_PROMPT)

    healthy = FakeLLM()
    answer = ask(CachingLLMClient(healthy), GATCHINA_PROMPT)

    assert len(healthy.calls) == 1
    assert answer["verdict"] == "allowed_main"


def test_expired_answer_is_asked_again(monkeypatch):
    inner = FakeLLM()
    ask(CachingLLMClient(inner), GATCHINA_PROMPT)

    monkeypatch.setenv(llm_cache.TTL_ENV_VAR, "1")
    real_time = llm_cache.time.time
    monkeypatch.setattr(
        llm_cache.time, "time", lambda: real_time() + 2 * llm_cache._SECONDS_PER_DAY
    )

    later = FakeLLM()
    ask(CachingLLMClient(later), GATCHINA_PROMPT)

    assert len(later.calls) == 1


def test_damaged_entry_is_treated_as_a_miss(tmp_path):
    inner = FakeLLM()
    ask(CachingLLMClient(inner), GATCHINA_PROMPT)

    entries = list((tmp_path / "llm_cache").rglob("*.json"))
    assert len(entries) == 1
    entries[0].write_text("{truncated", encoding="utf-8")

    recovered = FakeLLM()
    answer = ask(CachingLLMClient(recovered), GATCHINA_PROMPT)

    assert len(recovered.calls) == 1
    assert json.loads(entries[0].read_text(encoding="utf-8"))["value"] == answer


def test_first_stored_answer_is_kept():
    first = FakeLLM()
    kept = ask(CachingLLMClient(first), GATCHINA_PROMPT)

    racing = FakeLLM()
    cache = llm_cache.get_llm_cache()
    key = cache.build_key(
        model="gpt-oss-20b",
        client_fingerprint=CachingLLMClient(racing)._fingerprint,
        system_prompt=SYSTEM,
        user_prompt=GATCHINA_PROMPT,
        schema=SCHEMA,
        think_override=None,
    )
    cache.put(
        key,
        model="gpt-oss-20b",
        decoding={},
        system_prompt=SYSTEM,
        user_prompt=GATCHINA_PROMPT,
        schema=SCHEMA,
        think_override=None,
        value={"verdict": "not_allowed", "reason": "later"},
    )

    assert ask(CachingLLMClient(FakeLLM()), GATCHINA_PROMPT) == kept


def test_summary_reports_hit_rate():
    client = CachingLLMClient(FakeLLM())
    ask(client, GATCHINA_PROMPT)
    ask(client, GATCHINA_PROMPT)

    assert llm_cache.format_summary() == "hits=1 misses=1 stores=1 hit_rate=50%"


def test_entry_keeps_the_question_it_answers():
    client = CachingLLMClient(FakeLLM())
    answer = ask(client, GATCHINA_PROMPT, think_override="medium")

    entries = list(llm_cache.get_llm_cache().iter_entries())

    assert len(entries) == 1
    stored = entries[0]
    assert stored["user_prompt"] == GATCHINA_PROMPT
    assert stored["system_prompt"] == SYSTEM
    assert stored["think_override"] == "medium"
    assert stored["schema"] == SCHEMA
    assert stored["model"] == "gpt-oss-20b"
    assert stored["decoding"]["temperature"] == 0.0
    assert stored["value"] == answer


def test_iter_entries_skips_damaged_files(tmp_path):
    ask(CachingLLMClient(FakeLLM()), GATCHINA_PROMPT)
    ask(CachingLLMClient(FakeLLM()), DOLINSK_PROMPT)
    entries = sorted((tmp_path / "llm_cache").rglob("*.json"))
    entries[0].write_text("{truncated", encoding="utf-8")

    assert len(list(llm_cache.get_llm_cache().iter_entries())) == 1


def test_wrapper_delegates_unknown_attributes():
    inner = FakeLLM()
    assert CachingLLMClient(inner).default_model == "gpt-oss-20b"
