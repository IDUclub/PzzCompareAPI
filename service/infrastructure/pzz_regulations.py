"""The ПЗЗ of a territory, as NormGraph reads it from IDU_DVD, for the ВРИ check.

A ПЗЗ is adopted for a territory (e.g. an urban settlement) and tagged with it in
IDU_DVD. The territory comes from Urban API: either given directly or as the
project's territory (its geometry -> the deepest territory fully covering it,
``POST /common_territory``); then its parents up to the country. NormGraph is asked
for the ПЗЗ documents of that chain and the one of the deepest territory wins (the
settlement's rules over a district's).

Caches (per process): the territory chain of a project / territory for an hour,
the documents of a chain and the zones of a document for ten minutes, the zones of
every functional zone type (``pzz_zone_types``, one model call) per edition of the
document — the hash of its zones, so a re-parse with an unchanged version label is
a new edition too.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

import httpx
from cachetools import TTLCache

from service.application.use_cases.pzz_zone_types import (
    ZoneTypeMapping,
    base_zones,
    mapping_messages,
    mapping_schema,
    parse_mapping,
    prefix_mapping,
    vote,
)
from service.infrastructure.chat_llm_client import ChatLlmError
from service.infrastructure.normgraph_client import NormGraphClient
from service.infrastructure.urban_api_client import UrbanApiClient

if TYPE_CHECKING:
    from service.infrastructure.chat_llm_client import ChatLlmClient

logger = logging.getLogger("service.pzz_regulations")

_TERRITORIES: TTLCache = TTLCache(maxsize=512, ttl=3600)
_DOCUMENTS: TTLCache = TTLCache(maxsize=256, ttl=600)
_ZONES: TTLCache = TTLCache(maxsize=64, ttl=600)
_MAPPINGS: TTLCache = TTLCache(maxsize=64, ttl=24 * 3600)
# A failed model call is retried after a few minutes, not on every request.
_FALLBACK_MAPPINGS: TTLCache = TTLCache(maxsize=64, ttl=300)
_MAPPING_LOCKS: dict[str, asyncio.Lock] = {}
# Answers of the model per mapping, voted (``pzz_zone_types.vote``), and their
# temperature: a little, so the answers are independent.
_MAPPING_SAMPLES = 3
_MAPPING_TEMPERATURE = 0.3
# Deeper than a country -> region -> district -> settlement -> locality there is nothing.
_MAX_DEPTH = 8
_DOCUMENT_FIELDS = (
    "doc_id",
    "name",
    "title",
    "version",
    "territory_id",
    "territory_name",
    "effective_date",
)


class PzzNotFound(LookupError):
    """No territory or no ПЗЗ (with zones) for it; the message is for the caller."""


def _log(status: str, *, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(
        level,
        json.dumps(
            {"stage": "pzz_regulations", "status": status, **fields},
            ensure_ascii=False,
            default=str,
        ),
    )


async def territory_chain(
    urban: UrbanApiClient,
    *,
    project_id: int | None = None,
    territory_id: int | None = None,
    token: str | None,
) -> list[dict[str, Any]]:
    """``[{id, name}, …]`` from the territory (of the project) up to the root."""
    key = (
        ("project", project_id)
        if project_id is not None
        else ("territory", territory_id)
    )
    cached = _TERRITORIES.get(key)
    if cached is not None:
        return cached
    if project_id is not None:
        project_territory = await urban.get_project_territory(project_id, token=token)
        geometry = (project_territory or {}).get("geometry")
        if not geometry:
            return []
        territory = await urban.get_common_territory(geometry, token=token)
    else:
        territory = await urban.get_territory(territory_id, token=token)
    chain: list[dict[str, Any]] = []
    while territory and len(chain) < _MAX_DEPTH:
        current = territory.get("territory_id")
        if current is None or any(t["id"] == current for t in chain):
            break
        chain.append({"id": current, "name": territory.get("name")})
        parent = (territory.get("parent") or {}).get("id")
        if parent is None:
            break
        territory = await urban.get_territory(parent, token=token)
    if chain:
        _TERRITORIES[key] = chain
    return chain


def _pick_document(
    documents: list[dict[str, Any]], chain: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The ПЗЗ of the deepest territory; among several, the latest in force."""
    depth = {t["id"]: index for index, t in enumerate(chain)}
    candidates = [
        d for d in documents if d.get("territory_id") in depth and d.get("zones")
    ]
    candidates.sort(key=lambda d: d.get("effective_date") or "", reverse=True)
    candidates.sort(key=lambda d: depth[d["territory_id"]])
    return candidates[0] if candidates else None


async def resolve_territory_pzz(
    *,
    urban: UrbanApiClient,
    normgraph: NormGraphClient,
    project_id: int | None = None,
    territory_id: int | None = None,
    token: str | None,
) -> dict[str, Any]:
    """``{document, territory, requested_territory, zones}`` of the territory's ПЗЗ.

    Raises ``PzzNotFound`` when there is no territory or no ПЗЗ for it; Urban API /
    NormGraph failures propagate (``UrbanApiError``, ``NormGraphError``,
    ``httpx.HTTPError``).
    """
    chain = await territory_chain(
        urban, project_id=project_id, territory_id=territory_id, token=token
    )
    if not chain:
        raise PzzNotFound("Не удалось определить территорию в Urban API.")
    chain_key = tuple(t["id"] for t in chain)
    documents = _DOCUMENTS.get(chain_key)
    if documents is None:
        documents = await normgraph.regulation_documents(list(chain_key))
        _DOCUMENTS[chain_key] = documents
    document = _pick_document(documents, chain)
    if document is None:
        names = ", ".join(str(t["name"]) for t in chain)
        raise PzzNotFound(f"В NormGraph нет ПЗЗ для территорий: {names}.")
    zones_key = (document["doc_id"], document.get("version") or "")
    zones = _ZONES.get(zones_key)
    if zones is None:
        zones = await normgraph.zones(document["doc_id"])
        _ZONES[zones_key] = zones
    if not zones:
        raise PzzNotFound(
            f"В ПЗЗ «{document.get('name') or document['doc_id']}» не прочитаны зоны."
        )
    payload = {
        "document": {k: document.get(k) for k in _DOCUMENT_FIELDS},
        "territory": {
            "id": document.get("territory_id"),
            "name": document.get("territory_name"),
        },
        "requested_territory": chain[0],
        "zones": [{k: v for k, v in z.items() if k != "document"} for z in zones],
    }
    _log(
        "resolved",
        territory=chain[0],
        doc_id=document["doc_id"],
        version=document.get("version"),
        zones=len(zones),
    )
    return payload


def edition_key(payload: dict[str, Any]) -> str:
    """The document and the content of its zones, hashed: changes on any re-parse."""
    document = payload.get("document") or {}
    content = json.dumps(
        {
            "doc_id": document.get("doc_id"),
            "version": document.get("version"),
            "zones": [
                {k: z.get(k) for k in ("code", "name", "group", "uses")}
                for z in payload.get("zones") or []
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


async def _ask_model(
    llm_factory: Callable[[], "ChatLlmClient"],
    payload: dict[str, Any],
    *,
    types: list[dict[str, Any]],
    document_label: str,
) -> ZoneTypeMapping | None:
    """``_MAPPING_SAMPLES`` answers of the model at once, voted; None when none is usable."""
    key = edition_key(payload)
    zones = payload.get("zones") or []
    bases = base_zones(zones)
    messages = mapping_messages(types, bases, document_label)
    schema = mapping_schema(types, bases)
    try:
        async with llm_factory() as llm:
            # Low effort gives up on a few dozen zones and answers nothing.
            answers = await asyncio.gather(
                *(
                    llm.complete_json(
                        messages,
                        schema=schema,
                        temperature=_MAPPING_TEMPERATURE,
                        reasoning_effort="medium",
                    )
                    for _ in range(_MAPPING_SAMPLES)
                ),
                return_exceptions=True,
            )
    except RuntimeError as exc:  # the chat LLM is not configured
        _log("mapping_failed", level=logging.WARNING, edition=key, error=str(exc))
        return None
    mappings = []
    for answer in answers:
        if isinstance(answer, (ChatLlmError, httpx.HTTPError)):
            _log(
                "mapping_failed",
                level=logging.WARNING,
                edition=key,
                error=str(answer)[:500],
            )
        elif isinstance(answer, BaseException):
            raise answer
        elif (mapping := parse_mapping(answer, types, zones)) is not None:
            mappings.append(mapping)
    voted = vote(mappings, zones)
    if voted is None:
        _log("mapping_empty", level=logging.WARNING, edition=key)
    return voted


async def zone_type_mapping(
    payload: dict[str, Any],
    *,
    types: list[dict[str, Any]],
    document_label: str,
    llm_factory: Callable[[], "ChatLlmClient"] | None,
) -> ZoneTypeMapping:
    """The ПЗЗ zones of every functional zone type of this edition of the document.

    Asked once per edition (``_MAPPING_SAMPLES`` answers, voted); when the model is
    not configured, fails or names no zone, the zones are matched by code prefix
    (cached for a few minutes only).
    """
    key = edition_key(payload)
    cached = _MAPPINGS.get(key) or _FALLBACK_MAPPINGS.get(key)
    if cached is not None:
        return cached
    lock = _MAPPING_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        cached = _MAPPINGS.get(key) or _FALLBACK_MAPPINGS.get(key)
        if cached is not None:
            return cached
        zones = payload.get("zones") or []
        mapping = None
        if llm_factory is not None:
            mapping = await _ask_model(
                llm_factory, payload, types=types, document_label=document_label
            )
        if mapping is None:
            mapping = prefix_mapping(types, zones)
            _FALLBACK_MAPPINGS[key] = mapping
        else:
            _MAPPINGS[key] = mapping
            _log(
                "mapped",
                edition=key,
                by_type={k: len(v) for k, v in mapping.by_type.items()},
            )
        _MAPPING_LOCKS.pop(key, None)
        return mapping
