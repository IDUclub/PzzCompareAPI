"""The NormGraph context of a placement: the ПЗЗ document and the service's norms.

Built in the API process — NormGraph takes only the Keycloak service token, which
lives there — and handed to the worker as a task input, so the run itself stays
offline and repeatable.

- **ПЗЗ**: the document whose zone codes cover the uploaded zones best (at least
  ``MIN_ZONE_CODE_OVERLAP`` of the distinct codes); a ``territory_id`` narrows the
  candidates to the documents of that territory.
- **Norms**: the service name resolved to NormGraph entities (exact, alias or layer
  matches only — a text match like «спортивная школа» for «школа» is another
  service), then every restriction naming them, minus the ПЗЗ documents (their
  provisions are the zones' permitted uses, not norms), the excluded documents and
  duplicates, inverted into placement rules (``service.domain.placement``).

NormGraph being unavailable never fails a placement: the context then says why
and the worker falls back to the built-in zone mapping.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import httpx

from service.domain.placement import casefold_entity, rule_from_restriction
from service.infrastructure.normgraph_client import NormGraphError
from service.infrastructure.runners._deterministic_pzz import normalise_zone_code

logger = logging.getLogger("service.placement_context")

CONTEXT_VERSION = 1
MIN_ZONE_CODE_OVERLAP = 0.5
_MAX_DOCUMENTS = 10
_ENTITY_MATCHES = frozenset({"exact", "alias", "layer", "name", "stem"})
_DOCUMENT_FIELDS = (
    "doc_id",
    "name",
    "title",
    "version",
    "territory_id",
    "territory_name",
    "effective_date",
)


class NormGraphPort(Protocol):
    async def regulation_documents(
        self, territory_ids: list[int]
    ) -> list[dict[str, Any]]: ...

    async def zones(self, doc_id: str) -> list[dict[str, Any]]: ...

    async def resolve_entities(
        self, terms: list[str], *, limit: int = 10
    ) -> list[dict[str, Any]]: ...

    async def list_restrictions(
        self, entities: list[str], *, executable_only: bool = False
    ) -> list[dict[str, Any]]: ...


def _log(status: str, **fields: Any) -> None:
    logger.info(
        json.dumps(
            {"stage": "placement_context", "status": status, **fields},
            ensure_ascii=False,
            default=str,
        )
    )


def parse_excluded_documents(raw: str) -> frozenset[str]:
    return frozenset(
        casefold_entity(name) for name in (raw or "").split(";") if name.strip()
    )


def zone_code_overlap(uploaded_codes: set[str], zones: list[dict[str, Any]]) -> float:
    """Share of the uploaded (normalised) zone codes the document has a zone for."""
    if not uploaded_codes:
        return 0.0
    document_codes = {normalise_zone_code(z.get("code")) for z in zones}
    return len(uploaded_codes & document_codes) / len(uploaded_codes)


def _failure(exc: Exception) -> str:
    if isinstance(exc, NormGraphError):
        return f"ответ {exc.status}"
    return type(exc).__name__


async def _pick_regulations(
    normgraph: NormGraphPort,
    uploaded_codes: set[str],
    territory_id: int | None,
    notes: list[str],
) -> tuple[dict[str, Any] | None, frozenset[str]]:
    """The ПЗЗ for the uploaded zones, and the ids of every ПЗЗ document NormGraph has."""
    try:
        documents = await normgraph.regulation_documents([])
        regulation_ids = frozenset(str(d["doc_id"]) for d in documents)
        candidates = [
            d
            for d in documents
            if d.get("zones")
            and (territory_id is None or d.get("territory_id") == territory_id)
        ]
        candidates.sort(key=lambda d: d.get("effective_date") or "", reverse=True)
        best: tuple[float, dict[str, Any], list[dict[str, Any]]] | None = None
        for document in candidates[:_MAX_DOCUMENTS]:
            zones = await normgraph.zones(document["doc_id"])
            overlap = zone_code_overlap(uploaded_codes, zones)
            if best is None or overlap > best[0]:
                best = (overlap, document, zones)
    except (NormGraphError, httpx.HTTPError) as exc:
        logger.warning("NormGraph regulations unavailable: %s", exc)
        notes.append(
            f"ПЗЗ из NormGraph не получены ({_failure(exc)}); разрешённые виды "
            "использования взяты из справочника."
        )
        return None, frozenset()
    if best is None or best[0] < MIN_ZONE_CODE_OVERLAP:
        notes.append(
            "В NormGraph не найден документ ПЗЗ с кодами переданных зон; разрешённые "
            "виды использования взяты из справочника."
        )
        return None, regulation_ids
    overlap, document, zones = best
    _log("regulations", doc_id=document["doc_id"], overlap=round(overlap, 3))
    regulations = {
        "document": {k: document.get(k) for k in _DOCUMENT_FIELDS},
        "zone_code_overlap": round(overlap, 3),
        "zones": [{k: v for k, v in z.items() if k != "document"} for z in zones],
    }
    return regulations, regulation_ids


def _service_entities(
    resolutions: list[dict[str, Any]],
) -> tuple[list[str], frozenset[str]]:
    """Entity names to query by, and every casefolded name of the service."""
    names: list[str] = []
    aliases: set[str] = set()
    for resolution in resolutions:
        aliases.add(casefold_entity(resolution.get("term")))
        for candidate in resolution.get("candidates") or []:
            if candidate.get("match") not in _ENTITY_MATCHES:
                continue
            normalized = candidate.get("normalized") or ""
            if normalized and normalized not in names:
                names.append(normalized)
            for name in (
                normalized,
                candidate.get("name"),
                *(candidate.get("aliases") or []),
            ):
                aliases.add(casefold_entity(name))
    aliases.discard("")
    return names, frozenset(aliases)


def _dedupe(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One hit per duplicate group and per identical text of the same clause —
    NormGraph extracts some clauses twice under different ids."""
    seen: set[Any] = set()
    unique = []
    for hit in hits:
        provenance = hit.get("provenance") or {}
        text = " ".join(str(hit.get("extraction_text") or "").split())
        keys: set[Any] = {str(hit.get("duplicate_group") or hit.get("id") or "")}
        if text:
            keys.add((text, provenance.get("name"), provenance.get("numbering")))
        keys.discard("")
        if keys & seen:
            continue
        seen |= keys
        unique.append(hit)
    return unique


async def _service_rules(
    normgraph: NormGraphPort,
    service_name: str,
    search_terms: list[str],
    excluded_documents: frozenset[str],
    regulation_ids: frozenset[str],
    notes: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    terms = list(dict.fromkeys(t.strip() for t in [service_name, *search_terms]))
    try:
        resolutions = await normgraph.resolve_entities([t for t in terms if t])
        entities, aliases = _service_entities(resolutions)
        if not entities:
            notes.append(f"В NormGraph нет норм для объекта «{service_name}».")
            return [], []
        hits = await normgraph.list_restrictions(entities)
    except (NormGraphError, httpx.HTTPError) as exc:
        logger.warning("NormGraph norms unavailable: %s", exc)
        notes.append(f"Нормы из NormGraph не получены ({_failure(exc)}).")
        return [], []
    norms = [
        h
        for h in hits
        if str((h.get("provenance") or {}).get("doc_id") or "") not in regulation_ids
    ]
    if len(norms) < len(hits):
        notes.append(
            f"Положения ПЗЗ ({len(hits) - len(norms)}) не учтены как нормы: ПЗЗ "
            "учтены через разрешённые виды использования зон."
        )
    kept = [
        h
        for h in norms
        if casefold_entity((h.get("provenance") or {}).get("name"))
        not in excluded_documents
    ]
    if len(kept) < len(norms):
        notes.append(
            f"Не учтены нормы исключённых (тестовых) документов: {len(norms) - len(kept)}."
        )
    rules = [rule_from_restriction(h, aliases).to_dict() for h in _dedupe(kept)]
    _log("norms", entities=entities, restrictions=len(hits), rules=len(rules))
    return entities, rules


async def build_placement_context(
    normgraph: NormGraphPort | None,
    *,
    service_name: str,
    zone_codes: list[str],
    search_terms: list[str] | None = None,
    territory_id: int | None = None,
    excluded_documents: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """``{version, service_name, regulations, entities, rules, notes}`` for the worker.

    ``search_terms`` are other names of the service (the user's own words) to
    look its norms up by, next to ``service_name``.
    """
    notes: list[str] = []
    context: dict[str, Any] = {
        "version": CONTEXT_VERSION,
        "service_name": service_name,
        "regulations": None,
        "entities": [],
        "rules": [],
        "notes": notes,
    }
    if normgraph is None:
        notes.append(
            "NormGraph не подключён: нормы для объекта не проверялись, разрешённые "
            "виды использования взяты из справочника."
        )
        return context
    uploaded = {normalise_zone_code(c) for c in zone_codes} - {""}
    context["regulations"], regulation_ids = await _pick_regulations(
        normgraph, uploaded, territory_id, notes
    )
    context["entities"], context["rules"] = await _service_rules(
        normgraph,
        service_name,
        search_terms or [],
        excluded_documents,
        regulation_ids,
        notes,
    )
    return context
