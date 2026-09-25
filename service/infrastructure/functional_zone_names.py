"""Human-readable Russian names for urban_api functional zones.

A zone's own ``name`` is the most precise label a project carries, so it wins;
otherwise the zone is named by its ``functional_zone_type``. Some urban_api types
have no Russian nickname (``basic``, ``unknown``) — those get a Russian label here
so an English code never reaches a map layer, a result file or a chat answer.
"""

from __future__ import annotations

import re
from typing import Any

UNKNOWN_ZONE_TYPE_LABEL = "Тип зоны не определён"

_TYPE_LABELS_BY_CODE: dict[str, str] = {
    "basic": "Зона без профиля",
    "unknown": UNKNOWN_ZONE_TYPE_LABEL,
}

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _russian_text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text if _CYRILLIC.search(text) else ""


def functional_zone_type_label(zone_type: Any) -> str:
    """Russian label of a ``functional_zone_type`` object: nickname → known code → description."""
    if not isinstance(zone_type, dict):
        return UNKNOWN_ZONE_TYPE_LABEL
    nickname = _russian_text(
        zone_type.get("nickname") or zone_type.get("zone_nickname")
    )
    if nickname:
        return nickname
    code = str(zone_type.get("name") or "").strip().casefold()
    if code in _TYPE_LABELS_BY_CODE:
        return _TYPE_LABELS_BY_CODE[code]
    return _russian_text(zone_type.get("description")) or UNKNOWN_ZONE_TYPE_LABEL


def functional_zone_display_name(properties: dict[str, Any]) -> str:
    """Display name of one urban_api zone feature: its own ``name``, else its type label."""
    own_name = properties.get("name")
    if isinstance(own_name, str) and own_name.strip():
        return own_name.strip()
    return functional_zone_type_label(properties.get("functional_zone_type"))
