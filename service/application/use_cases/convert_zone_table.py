"""Fold a tidy zone-description table into the PZZ label-schema JSON.

Input is one row per (zone, VRI, permission-kind), as a real ПЗЗ regulation is
naturally tabulated (see the Долинский градрегламент export). Output is the list
schema consumed by the building flow's ``load_pzz_label_mapping`` —
``[{zone_code, zone_name, main[], conditional[], auxiliary[]}]`` — so a converted
spreadsheet drops straight into the deterministic zone backend, no LLM.

The mapping from a permission label to its bucket is deterministic substring
matching on the Russian regulation vocabulary; the column→role mapping is resolved
upstream (heuristic + LLM) and passed in as ``column_map``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...infrastructure.runners._deterministic_pzz import normalise_zone_code

_SECTIONS = ("main", "conditional", "auxiliary")


@dataclass
class ConversionReport:
    """What the converter did, for the frontend preview before a run."""

    zones_count: int
    vri_count: int
    rows_total: int
    rows_used: int
    warnings: list[str] = field(default_factory=list)


class ConversionError(ValueError):
    """A required column (zone_code / vri_code) was not resolved."""


def _permission_bucket(value: str) -> str | None:
    """Map a permission label to ``main``/``conditional``/``auxiliary``, else None.

    Matches the regulation vocabulary: «Основной вид…», «Условно разрешённый»,
    «Вспомогательный». Returns None for an unrecognised (non-empty) value so the
    caller can warn and default it to ``main``.
    """
    n = value.strip().casefold().replace("ё", "е")
    if not n:
        return None
    if "основн" in n:
        return "main"
    if "условн" in n:
        return "conditional"
    if "вспомог" in n:
        return "auxiliary"
    return None


def convert_zone_table(
    rows: list[dict[str, str]],
    column_map: dict[str, str | None],
) -> tuple[list[dict[str, Any]], ConversionReport]:
    """Convert table ``rows`` to zone label entries using ``column_map``.

    ``column_map`` maps each role (``zone_code``, ``vri_code``, and optionally
    ``zone_name``, ``permission``, ``vri_name``) to a header name in ``rows``.
    ``zone_code`` and ``vri_code`` are required; missing them raises
    :class:`ConversionError`. When ``permission`` is unmapped, every VRI is filed
    under ``main`` and a warning is recorded.
    """
    zone_code_col = column_map.get("zone_code")
    vri_code_col = column_map.get("vri_code")
    if not zone_code_col:
        raise ConversionError("column for zone_code was not resolved")
    if not vri_code_col:
        raise ConversionError("column for vri_code was not resolved")
    zone_name_col = column_map.get("zone_name")
    permission_col = column_map.get("permission")
    vri_name_col = column_map.get("vri_name")

    warnings: list[str] = []
    if not permission_col:
        warnings.append(
            "Колонка типа разрешения не найдена — все виды разрешённого использования "
            "отнесены к основным."
        )

    zones: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    rows_used = 0
    skipped_no_zone = 0
    skipped_no_vri = 0
    unknown_permissions: set[str] = set()

    for row in rows:
        zone_raw = (row.get(zone_code_col) or "").strip()
        key = normalise_zone_code(zone_raw)
        if not key:
            skipped_no_zone += 1
            continue
        vri_code = (row.get(vri_code_col) or "").strip()
        if not vri_code:
            skipped_no_vri += 1
            continue

        if permission_col:
            perm_raw = (row.get(permission_col) or "").strip()
            bucket = _permission_bucket(perm_raw)
            if bucket is None:
                if perm_raw:
                    unknown_permissions.add(perm_raw)
                bucket = "main"
        else:
            bucket = "main"

        if key not in zones:
            zones[key] = {
                "zone_code": zone_raw,
                "zone_name": "",
                "buckets": {s: {} for s in _SECTIONS},
            }
            order.append(key)
        entry = zones[key]

        if zone_name_col and not entry["zone_name"]:
            name = (row.get(zone_name_col) or "").strip()
            if name:
                entry["zone_name"] = name

        vri_name = (row.get(vri_name_col) or "").strip() if vri_name_col else ""
        existing = entry["buckets"][bucket].get(vri_code)
        if existing is None or (not existing and vri_name):
            entry["buckets"][bucket][vri_code] = vri_name
        rows_used += 1

    if unknown_permissions:
        warnings.append(
            "Не распознан тип разрешения (отнесено к основным): "
            + ", ".join(sorted(unknown_permissions))
        )
    if skipped_no_zone:
        warnings.append(f"Пропущено строк без кода зоны: {skipped_no_zone}.")
    if skipped_no_vri:
        warnings.append(
            "Пропущено строк без кода вида разрешённого использования: "
            f"{skipped_no_vri}."
        )

    result: list[dict[str, Any]] = []
    vri_total = 0
    for key in order:
        entry = zones[key]
        zone_obj: dict[str, Any] = {
            "zone_code": entry["zone_code"],
            "zone_name": entry["zone_name"] or entry["zone_code"],
        }
        for section in _SECTIONS:
            items = [
                {"vri_code": code, "vri_name": name}
                for code, name in entry["buckets"][section].items()
            ]
            vri_total += len(items)
            zone_obj[section] = items
        result.append(zone_obj)

    report = ConversionReport(
        zones_count=len(result),
        vri_count=vri_total,
        rows_total=len(rows),
        rows_used=rows_used,
        warnings=warnings,
    )
    return result, report
