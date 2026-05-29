"""Lookup of functional_zone_type_id → PZZ зона справка.

Loads the static mapping file once per process. Used by the scenarios
endpoint to enrich zone descriptions with PZZ permitted-use summary and
by /tasks/{id}/object-zone-fit to group per-zone results.

The mapping is currently city-specific (Долинск); future versions may
move to per-scenario mappings fetched from urban_api or uploaded by the
client. For now we have one global mapping.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _mapping_path() -> Path:
    from service.settings import get_settings

    return Path(get_settings().default_fz_to_pzz_mapping_path).resolve()


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


@lru_cache(maxsize=4)
def _version_cached(path_str: str, mtime: float) -> str:
    path = Path(path_str)
    if mtime < 0 or not path.is_file():
        return "none"
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:8]


def mapping_version() -> str:
    """Short content-hash of the mapping file, used as an idempotency-key
    component so cached scenario tasks invalidate transparently whenever the
    local PZZ mapping changes — keyed on file mtime, so an edit is picked up
    without restarting the API.

    Returns ``"none"`` when the mapping file is missing — keeps the key shape
    stable but means stale results won't auto-invalidate (degraded mode,
    acceptable for dev).
    """
    path = _mapping_path()
    return _version_cached(str(path), _mtime(path))


@lru_cache(maxsize=4)
def _load_mapping_cached(path_str: str, mtime: float) -> dict[str, Any]:
    path = Path(path_str)
    if mtime < 0 or not path.is_file():
        return {"functional_zone_mappings": []}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_mapping_raw() -> dict[str, Any]:
    """Load the JSON mapping, reloading transparently when the file changes."""
    path = _mapping_path()
    return _load_mapping_cached(str(path), _mtime(path))


@lru_cache(maxsize=4)
def _index_cached(path_str: str, mtime: float) -> dict[int, dict[str, Any]]:
    raw = _load_mapping_cached(path_str, mtime)
    out: dict[int, dict[str, Any]] = {}
    for entry in raw.get("functional_zone_mappings", []):
        type_id = entry.get("functional_zone_type_id")
        if type_id is None:
            continue
        try:
            out[int(type_id)] = entry
        except (TypeError, ValueError):
            continue
    return out


def _index_by_type_id() -> dict[int, dict[str, Any]]:
    path = _mapping_path()
    return _index_cached(str(path), _mtime(path))


def _empty_summary() -> dict[str, Any]:
    """The shape used when a zone has no mapping at all."""
    return {
        "mapping_status": "no_mapping",
        "mapping_confidence": None,
        "mapping_note": None,
        "db_zone_nickname": None,
        "source_pzz_zone_codes": None,
        "allowed_construction_summary": None,
        "main_vri": None,
        "conditional_vri": None,
        "auxiliary_vri": None,
    }


def _zone_label_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert one mapping entry into a pipeline labels-file record."""
    type_id = entry.get("functional_zone_type_id")
    zone_code = str(type_id) if type_id is not None else ""
    db_name = entry.get("db_zone_nickname") or entry.get("db_name") or ""
    profile = entry.get("averaged_pzz_profile") or {}
    summary = profile.get("allowed_construction_summary") or ""

    retrieval_parts: list[str] = []
    if db_name:
        retrieval_parts.append(f"Зона: {db_name}.")
    if entry.get("mapping_note"):
        retrieval_parts.append(entry["mapping_note"])
    if summary:
        retrieval_parts.append(summary)

    return {
        "zone_code": zone_code,
        "base_zone_code": zone_code,
        "zone_heading": f"{zone_code} {db_name}".strip() or zone_code,
        "zone_name": db_name,
        "zone_notes": [],
        "main": profile.get("main_vri") or [],
        "conditional": profile.get("conditional_vri") or [],
        "auxiliary": profile.get("auxiliary_vri") or [],
        "section_notes": "",
        "source_doc": "functional_zones_to_pzz_mapping (averaged)",
        "zone_summary": summary,
        "retrieval_text": "\n\n".join(retrieval_parts),
        "normalization_source": "functional_zone_mapping",
    }


def _stub_zone_label(zone_type_id: int | str, zone_name: str | None = None) -> dict[str, Any]:
    """Empty-permitted-VRI stub for zones that have no PZZ mapping.

    The pipeline still treats this as a known zone (so the verdict is
    ``not_allowed`` rather than ``no_zone_metadata``), but with no
    permitted uses there's nothing to match against.
    """
    zone_code = str(zone_type_id)
    name = zone_name or zone_code
    return {
        "zone_code": zone_code,
        "base_zone_code": zone_code,
        "zone_heading": f"{zone_code} {name}".strip(),
        "zone_name": name,
        "zone_notes": [],
        "main": [],
        "conditional": [],
        "auxiliary": [],
        "section_notes": "",
        "source_doc": "stub (no PZZ mapping)",
        "zone_summary": (
            f"Зона '{name}' — нет соответствия в маппинге ПЗЗ; "
            "разрешённые ВРИ неизвестны."
        ),
        "retrieval_text": (
            f"Зона: {name}.\nДля этой функциональной зоны нет описания "
            "разрешённых видов использования."
        ),
        "normalization_source": "stub",
    }


def build_pipeline_zone_labels(
    observed_zone_types: dict[int | str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build a list of labels records in the pipeline's expected format.

    Every mapping entry produces a record (keyed by
    ``functional_zone_type_id``). Additionally, when ``observed_zone_types``
    is supplied, any id present there but missing from the mapping gets
    a stub record so the pipeline doesn't emit ``no_zone_metadata`` —
    instead it can correctly say "not allowed" for those zones.

    Parameters
    ----------
    observed_zone_types
        Optional dict ``{functional_zone_type_id: zone_name}`` of zones
        actually present in the scenario being classified. Used to emit
        stubs for zones missing from the mapping.
    """
    raw = _load_mapping_raw()
    labels: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in raw.get("functional_zone_mappings", []):
        record = _zone_label_from_entry(entry)
        if not record["zone_code"]:
            continue
        labels.append(record)
        seen_ids.add(record["zone_code"])

    if observed_zone_types:
        for type_id, type_name in observed_zone_types.items():
            if type_id is None:
                continue
            zone_code = str(type_id)
            if zone_code in seen_ids:
                continue
            labels.append(_stub_zone_label(type_id, zone_name=type_name))
            seen_ids.add(zone_code)

    return labels


def lookup_zone_summary(functional_zone_type_id: int | str | None) -> dict[str, Any]:
    """Return a stable-shape summary dict for one functional_zone_type_id.

    ``mapping_status`` is one of:
      - ``"ok"`` — mapping exists with confidence high or medium
      - ``"low_confidence"`` — mapping exists but confidence is low/none
      - ``"no_mapping"`` — id not found in the mapping file

    The other fields are populated when ``mapping_status != "no_mapping"``
    and may be None for individual missing sub-fields.
    """
    if functional_zone_type_id is None:
        return _empty_summary()
    try:
        type_id = int(functional_zone_type_id)
    except (TypeError, ValueError):
        return _empty_summary()

    entry = _index_by_type_id().get(type_id)
    if entry is None:
        return _empty_summary()

    confidence = (entry.get("mapping_confidence") or "").lower()
    status = "low_confidence" if confidence in {"low", "none"} else "ok"
    profile = entry.get("averaged_pzz_profile") or {}

    return {
        "mapping_status": status,
        "mapping_confidence": entry.get("mapping_confidence"),
        "mapping_note": entry.get("mapping_note"),
        "db_zone_nickname": entry.get("db_zone_nickname"),
        "source_pzz_zone_codes": profile.get("source_pzz_zone_codes"),
        "allowed_construction_summary": profile.get("allowed_construction_summary"),
        "main_vri": profile.get("main_vri"),
        "conditional_vri": profile.get("conditional_vri"),
        "auxiliary_vri": profile.get("auxiliary_vri"),
    }
