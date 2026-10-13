"""Resolve a natural-language service name («школа», «детсад») to a ВРИ code.

Used by the ``placement`` chat mode: the user names what they want to build and
the service picks the ВРИ (Rosreestr classifier code) it has to be checked
against. Resolution goes from the cheapest/most trustworthy step to the least:

1. an explicit code in the text («3.5.1»), validated against the classifier;
2. an exact (normalised) match against the service catalogue
   (``service_type_to_vri.json``), built-in synonyms and ВРИ names;
3. a stem match — every word of a catalogue name found (as a word-form) in the
   query, so «где построить школу» hits «Школа»;
4. semantic: embedding cosine over catalogue names (when a vectorizer is set);
5. LLM: one ``complete_json`` call whose enum is the real classifier codes, so
   the model cannot invent a code.

Returns ``None`` when nothing resolves — the caller asks the user to rephrase
instead of guessing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ...infrastructure.chat_llm_client import ChatLlmError, build_chat_llm_client
from ...infrastructure.embeddings_client import EmbeddingsClient, EmbeddingsError

if TYPE_CHECKING:
    from ...settings import Settings

logger = logging.getLogger("service.resolve_service_vri")

SOURCE_RU = {
    "code": "код указан явно",
    "catalog": "по справочнику сервисов",
    "vri_name": "по названию вида разрешённого использования",
    "semantic": "по смыслу названия",
    "llm": "подобрано языковой моделью",
}

# Common colloquial names that are not in the service catalogue. Value is
# (ВРИ code, canonical service name shown to the user).
_EXTRA_ALIASES: dict[str, tuple[str, str]] = {
    "детсад": ("3.5.1", "Детский сад"),
    "садик": ("3.5.1", "Детский сад"),
    "ясли": ("3.5.1", "Детский сад"),
    "доу": ("3.5.1", "Детский сад"),
    "дошкольное учреждение": ("3.5.1", "Детский сад"),
    "сош": ("3.5.1", "Школа"),
    "гимназия": ("3.5.1", "Школа"),
    "лицей": ("3.5.1", "Школа"),
    "колледж": ("3.5.2", "Колледж"),
    "техникум": ("3.5.2", "Техникум"),
    "университет": ("3.5.2", "Университет"),
    "вуз": ("3.5.2", "Вуз"),
    "амбулатория": ("3.4.1", "Амбулатория"),
    "фап": ("3.4.1", "Фельдшерско-акушерский пункт"),
    "фельдшерско акушерский пункт": ("3.4.1", "Фельдшерско-акушерский пункт"),
    "медпункт": ("3.4.1", "Медпункт"),
    "стационар": ("3.4.2", "Больница"),
    "госпиталь": ("3.4.2", "Больница"),
    "магазин": ("4.4", "Магазин"),
    "продуктовый магазин": ("4.4", "Магазин"),
    "азс": ("4.9.1.1", "Автозаправочная станция"),
    "заправка": ("4.9.1.1", "Автозаправочная станция"),
    "тц": ("4.2", "Торговый центр"),
    "торговый центр": ("4.2", "Торговый центр"),
    "спортзал": ("5.1.2", "Спортивный зал"),
    "фок": ("5.1.2", "Физкультурно-оздоровительный комплекс"),
    "почта": ("3.2.3", "Почтовое отделение"),
    "банк": ("4.5", "Отделение банка"),
    "дом культуры": ("3.6.1", "Дом культуры"),
    "дк": ("3.6.1", "Дом культуры"),
    "храм": ("3.7.1", "Храм"),
    "церковь": ("3.7.1", "Храм"),
    "кладбище": ("12.1", "Кладбище"),
    "общежитие": ("3.2.4", "Общежитие"),
    "отель": ("4.7", "Гостиница"),
    "склад": ("6.9", "Склад"),
    "ижс": ("2.1", "Индивидуальный жилой дом"),
}

# Words dropped from catalogue names before stem matching («Продукты (магазин у
# дома)» must not require the query to contain «у»).
_STOPWORDS = frozenset(
    "и в во на для с со по у к о об от до из за при или а также том числе".split()
)

_CODE_RE = re.compile(r"(?<![\d.])\d{1,2}(?:\.\d{1,2}){1,3}(?![\d.])")
_TOKEN_RE = re.compile(r"[0-9a-zа-я]+")


@dataclass(frozen=True)
class ResolvedService:
    """A service name resolved to a classifier ВРИ code."""

    code: str
    vri_name: str
    service_name: str
    source: str  # "code" | "catalog" | "vri_name" | "semantic" | "llm"
    query: str

    @property
    def source_ru(self) -> str:
        return SOURCE_RU.get(self.source, self.source)


@dataclass(frozen=True)
class _Candidate:
    name: str  # text that is matched against the query
    code: str
    label: str  # service name shown to the user
    kind: str  # "catalog" | "vri_name"


@dataclass(frozen=True)
class _Catalog:
    vri_names: dict[str, str]  # code -> ВРИ name
    candidates: tuple[_Candidate, ...]
    exact: dict[str, _Candidate | None]  # normalised name -> candidate (None = ambiguous)


def _normalise(text: str) -> str:
    text = str(text).casefold().replace("ё", "е")
    return " ".join(_TOKEN_RE.findall(text))


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


@lru_cache(maxsize=4)
def _load_catalog(vri_path: str, services_path: str) -> _Catalog:
    vri_raw = json.loads(Path(vri_path).read_text(encoding="utf-8"))
    vri_names: dict[str, str] = {}
    vri_plain: dict[str, str] = {}
    for code, entry in (vri_raw.get("by_code") or {}).items():
        name = str(entry.get("name") or "").strip()
        # The source table carries a junk header row ({"code": "3", "name": "1"}).
        if "." not in code or not name or name.isdigit():
            continue
        vri_names[code] = name
        vri_plain[code] = str(entry.get("name_plain") or name).strip()

    candidates: list[_Candidate] = []
    services_raw = json.loads(Path(services_path).read_text(encoding="utf-8"))
    for entry in (services_raw.get("by_service_type_id") or {}).values():
        code = entry.get("vri_code")
        name = str(entry.get("name") or "").strip()
        if not code or code not in vri_names or not name:
            continue
        candidates.append(_Candidate(name, code, name, "catalog"))
        for alias in entry.get("aliases") or ():
            candidates.append(_Candidate(str(alias), code, name, "catalog"))
    for alias, (code, label) in _EXTRA_ALIASES.items():
        if code in vri_names:
            candidates.append(_Candidate(alias, code, label, "catalog"))
    for code, name in vri_names.items():
        candidates.append(_Candidate(name, code, vri_plain[code], "vri_name"))
        if vri_plain[code] != name:
            candidates.append(
                _Candidate(vri_plain[code], code, vri_plain[code], "vri_name")
            )

    exact: dict[str, _Candidate | None] = {}
    for cand in candidates:
        key = _normalise(cand.name)
        if not key:
            continue
        prev = exact.get(key, cand)
        exact[key] = cand if prev is not None and prev.code == cand.code else None
    return _Catalog(vri_names, tuple(candidates), exact)


def _token_matches(query_token: str, name_token: str) -> bool:
    """Word-form match: «школу» ~ «школа», «детского» ~ «детский»."""
    if query_token == name_token:
        return True
    common = 0
    for a, b in zip(query_token, name_token):
        if a != b:
            break
        common += 1
    n = len(name_token)
    need = min(n, max(4, n - 2))
    max_extra = 1 if n <= 4 else 3
    return common >= need and len(query_token) - n <= max_extra


def _stem_match(query: str, catalog: _Catalog) -> _Candidate | None:
    q_tokens = _tokens(query)
    if not q_tokens:
        return None
    best: dict[str, tuple[int, int, _Candidate]] = {}
    for cand in catalog.candidates:
        name_tokens = [t for t in _tokens(cand.name) if t not in _STOPWORDS]
        if not name_tokens:
            continue
        positions: list[int] = []
        for nt in name_tokens:
            pos = next(
                (i for i, qt in enumerate(q_tokens) if _token_matches(qt, nt)), None
            )
            if pos is None:
                break
            positions.append(pos)
        else:
            score = (len(name_tokens), -min(positions))
            prev = best.get(cand.kind)
            if prev is None or score > prev[:2]:
                best[cand.kind] = (*score, cand)
    # A service-catalogue hit beats a ВРИ-name hit: «школа» is a service, and
    # ВРИ names are long legal phrases that rarely all appear in a query.
    for kind in ("catalog", "vri_name"):
        if kind in best:
            return best[kind][2]
    return None


def resolve_deterministic(query: str, settings: "Settings") -> ResolvedService | None:
    """Steps 1–3 (no network): explicit code, exact name, stem match."""
    catalog = _load_catalog(
        settings.default_vri_classifier_path, settings.service_type_to_vri_path
    )
    text = (query or "").strip()
    if not text:
        return None
    for match in _CODE_RE.finditer(text):
        code = match.group(0)
        if code in catalog.vri_names:
            name = catalog.vri_names[code]
            return ResolvedService(code, name, name, "code", text)
    cand = catalog.exact.get(_normalise(text))
    if cand is None:
        cand = _stem_match(text, catalog)
    if cand is None:
        return None
    return ResolvedService(
        cand.code, catalog.vri_names[cand.code], cand.label, cand.kind, text
    )


# Per-process cache of catalogue embeddings keyed by (url, model, paths).
_VECTOR_CACHE: dict[tuple[str, ...], tuple[list[_Candidate], Any]] = {}


def _semantic_resolve(query: str, settings: "Settings") -> ResolvedService | None:
    """Blocking embedding match of the query against catalogue names."""
    import numpy as np

    catalog = _load_catalog(
        settings.default_vri_classifier_path, settings.service_type_to_vri_path
    )
    client = EmbeddingsClient(
        url=settings.vectorizer_url,
        model=settings.embed_model,
        batch_size=settings.embed_batch_size,
    )
    key = (
        settings.vectorizer_url,
        settings.embed_model,
        settings.default_vri_classifier_path,
        settings.service_type_to_vri_path,
    )
    cached = _VECTOR_CACHE.get(key)
    if cached is None:
        cands = list(catalog.candidates)
        mat = np.asarray(client.embed([c.name for c in cands]), dtype=float)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        cached = (cands, mat)
        _VECTOR_CACHE[key] = cached
    cands, mat = cached
    qvec = np.asarray(client.embed([query])[0], dtype=float)
    qvec /= np.linalg.norm(qvec) + 1e-12
    sims = mat @ qvec
    j = int(sims.argmax())
    if float(sims[j]) < settings.building_semantic_threshold:
        return None
    cand = cands[j]
    return ResolvedService(
        cand.code, catalog.vri_names[cand.code], cand.label, "semantic", query
    )


async def _llm_resolve(
    query: str, settings: "Settings", model: str | None
) -> ResolvedService | None:
    catalog = _load_catalog(
        settings.default_vri_classifier_path, settings.service_type_to_vri_path
    )
    codes = list(catalog.vri_names)
    listing = "\n".join(f"{c} — {catalog.vri_names[c]}" for c in codes)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты эксперт по градостроительному зонированию РФ. Пользователь "
                "называет объект, который хочет разместить. Выбери ОДИН код вида "
                "разрешённого использования (ВРИ) из классификатора Росреестра, "
                "под который подпадает этот объект. Если запрос не про размещение "
                "объекта или подходящего кода нет — верни null.\n\n"
                f"Классификатор:\n{listing}"
            ),
        },
        {"role": "user", "content": query},
    ]
    schema = {
        "type": "object",
        "properties": {
            "vri_code": {"type": ["string", "null"], "enum": [*codes, None]}
        },
        "required": ["vri_code"],
    }
    async with build_chat_llm_client(settings) as client:
        answer = await client.complete_json(messages, schema=schema, model=model)
    code = answer.get("vri_code")
    if not isinstance(code, str) or code not in catalog.vri_names:
        return None
    name = catalog.vri_names[code]
    return ResolvedService(code, name, query, "llm", query)


async def resolve_service_vri(
    query: str, settings: "Settings", *, model: str | None = None
) -> ResolvedService | None:
    """Resolve ``query`` to a ВРИ code, or ``None`` when nothing fits."""
    text = (query or "").strip()
    if not text:
        return None
    resolved = resolve_deterministic(text, settings)
    if resolved is not None:
        return resolved
    if settings.vectorizer_url and settings.building_semantic_fallback:
        try:
            resolved = await asyncio.to_thread(_semantic_resolve, text, settings)
        except (EmbeddingsError, httpx.HTTPError, ValueError) as exc:
            logger.warning("semantic service->VRI resolution failed: %s", exc)
        if resolved is not None:
            return resolved
    try:
        return await _llm_resolve(text, settings, model)
    except (ChatLlmError, httpx.HTTPError) as exc:
        logger.warning("LLM service->VRI resolution failed: %s", exc)
        return None
