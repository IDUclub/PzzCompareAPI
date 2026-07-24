"""Zone-review logic for the building_pzz_check letter-index flow.

When a user uploads a real ПЗЗ (letter indices «Ж-1», «СХ-3» …) but no zone
descriptions file, the deterministic runner falls back to the built-in template
mapping — which only approximates their municipality's regulations and covers a
subset of indices. This module decides, before the task runs, how to handle the
gap, mirroring the column-detection retry pattern:

  * ``proceed``        — numeric urban_api zones, or the user's own descriptions,
                         or the template already covers every zone (still flagged
                         approximate when template-based);
  * ``suggest_upload`` — many uncovered zones (≥ threshold): the template clearly
                         doesn't fit; ask the user to upload their descriptions;
  * ``confirm``        — a few uncovered zones: offer a per-zone LLM suggestion of
                         the nearest template zone for the user to confirm.

The LLM call itself lives in the endpoint (async); the prompt/response shaping and
the overlay construction are pure and live here so they can be unit-tested without
the SSE stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from service.infrastructure.runners._deterministic_pzz import (
    load_pzz_label_mapping,
    normalise_zone_code,
    zone_code_display_map,
    zone_codes_are_numeric,
)


@dataclass
class UncoveredZone:
    """A zone code present on the uploaded layer but absent from the mapping."""

    code: str  # user's verbatim display code, e.g. «СХ-3»
    name: str  # user's zone name (from the name column), may be ""
    norm: str  # normalised match key


@dataclass
class ZoneReview:
    numeric: bool
    approximate: bool
    action: str  # "proceed" | "suggest_upload" | "confirm"
    uncovered: list[UncoveredZone] = field(default_factory=list)

    @property
    def uncovered_codes(self) -> list[str]:
        return [z.code for z in self.uncovered]


def _zone_names(
    zones_fc: dict[str, Any], code_col: str, name_col: str | None
) -> dict[str, str]:
    """Map a normalised zone key to the first non-empty name seen for it."""
    names: dict[str, str] = {}
    if not name_col:
        return names
    for f in zones_fc.get("features") or []:
        props = f.get("properties") or {}
        key = normalise_zone_code(props.get(code_col))
        if not key or key in names:
            continue
        name = props.get(name_col)
        if name not in (None, ""):
            names[key] = str(name).strip()
    return names


def review_building_zones(
    *,
    zones_fc: dict[str, Any],
    code_col: str,
    name_col: str | None,
    user_uploaded_descriptions: bool,
    confirmed_map: dict[str, str] | None,
    template_path: str,
    threshold: int,
) -> ZoneReview:
    """Decide how to handle the uploaded zone layer before running the task."""
    if zone_codes_are_numeric(zones_fc, code_col):
        return ZoneReview(numeric=True, approximate=False, action="proceed")
    if user_uploaded_descriptions:
        # The user supplied authoritative descriptions for their own ПЗЗ.
        return ZoneReview(numeric=False, approximate=False, action="proceed")

    allowed, _nick = load_pzz_label_mapping(template_path)
    display = zone_code_display_map(zones_fc, code_col)
    names = _zone_names(zones_fc, code_col, name_col)
    uncovered = sorted(
        (
            UncoveredZone(code=display[k], name=names.get(k, ""), norm=k)
            for k in display
            if k not in allowed
        ),
        key=lambda z: z.code,
    )
    # Template- or overlay-based ⇒ approximate regardless of the branch below.
    if confirmed_map or not uncovered:
        return ZoneReview(
            numeric=False, approximate=True, action="proceed", uncovered=uncovered
        )
    if len(uncovered) >= threshold:
        return ZoneReview(
            numeric=False,
            approximate=True,
            action="suggest_upload",
            uncovered=uncovered,
        )
    return ZoneReview(
        numeric=False, approximate=True, action="confirm", uncovered=uncovered
    )


def remaining_uncovered(
    uncovered: list[UncoveredZone], confirmed_map: dict[str, str] | None
) -> list[str]:
    """Display codes still without a mapping after applying confirmations."""
    confirmed_norm = {normalise_zone_code(k) for k in (confirmed_map or {})}
    return [z.code for z in uncovered if z.norm not in confirmed_norm]


def build_disclaimer(remaining: list[str]) -> str:
    """The approximate-classification note prepended to the streamed answer."""
    lines = [
        "⚠️ Проверка выполнена по обобщённому шаблону ПЗЗ и является приблизительной. "
        "Для точного результата загрузите описание разрешённых ВРИ вашего ПЗЗ "
        "(поле pzz_descriptions_file)."
    ]
    if remaining:
        lines.append(
            "Зоны без описания в шаблоне (вердикт не вынесен): "
            + ", ".join(remaining)
            + "."
        )
    return "\n".join(lines)


def _template_entries(template_path: str) -> list[dict[str, Any]]:
    raw = json.loads(Path(template_path).read_text(encoding="utf-8"))
    entries = (
        raw
        if isinstance(raw, list)
        else (raw.get("zones") or raw.get("labels") or raw.get("pzz_zones") or [])
    )
    return [e for e in entries if isinstance(e, dict)]


def template_candidates(template_path: str) -> list[dict[str, str]]:
    """`[{code, name}]` of template zones, for the LLM suggestion prompt."""
    out: list[dict[str, str]] = []
    for e in _template_entries(template_path):
        code = e.get("zone_code")
        if code in (None, ""):
            continue
        out.append(
            {"code": str(code).strip(), "name": str(e.get("zone_name") or "").strip()}
        )
    return out


def build_suggestion_messages(
    uncovered: list[UncoveredZone], candidates: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the (messages, json-schema) for one batched LLM suggestion call.

    For each uncovered zone the model returns the code of the *closest* template
    zone (by name/meaning) from the candidate list, or null. It never invents a
    code — the schema enums the candidate codes. The result is advisory: the user
    confirms before it is applied (see ``build_confirmed_overlay``).
    """
    codes = [c["code"] for c in candidates]
    indexed = {f"z{i}": z for i, z in enumerate(uncovered)}
    schema = {
        "type": "object",
        "properties": {
            key: {"type": ["string", "null"], "enum": [*codes, None]} for key in indexed
        },
        "required": list(indexed),
        "additionalProperties": False,
    }
    catalogue = "\n".join(f"- {c['code']}: {c['name']}" for c in candidates)
    listing = "\n".join(
        f"{key} = «{z.code}»" + (f" — {z.name}" if z.name else "")
        for key, z in indexed.items()
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты сопоставляешь территориальные зоны ПЗЗ пользователя с зонами "
                "шаблона по смыслу их наименований. Для каждой зоны верни код "
                "ближайшей по смыслу зоны шаблона ИЗ ПРЕДЛОЖЕННОГО СПИСКА кодов, либо "
                "null, если подходящей зоны нет. Никогда не придумывай код. Это "
                "предварительная подсказка — окончательное решение принимает человек. "
                "Ответ строго JSON по схеме."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Зоны шаблона — код: наименование:\n{catalogue}\n\n"
                f"Зоны пользователя для сопоставления:\n{listing}"
            ),
        },
    ]
    return messages, schema


def parse_suggestions(
    uncovered: list[UncoveredZone],
    parsed: dict[str, Any],
    candidates: list[dict[str, str]],
) -> dict[str, str]:
    """Turn the LLM response into `{user_code: suggested_template_code}` for the
    confident hits (a valid candidate code); nulls / unknowns are dropped."""
    valid = {c["code"] for c in candidates}
    out: dict[str, str] = {}
    for i, z in enumerate(uncovered):
        pick = parsed.get(f"z{i}")
        if isinstance(pick, str) and pick in valid:
            out[z.code] = pick
    return out


def build_confirmed_overlay(
    template_path: str, confirmed_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Template entries plus, per confirmed `{user_code: template_code}`, a new
    entry keyed by the user's code carrying the template zone's permitted ВРИ.

    Written to a labels file and fed back as the descriptions upload, so the user's
    «СХ-3» resolves against the confirmed template zone's ВРИ set on the re-run.
    """
    entries = _template_entries(template_path)
    by_norm = {
        normalise_zone_code(e.get("zone_code")): e
        for e in entries
        if normalise_zone_code(e.get("zone_code"))
    }
    overlay: list[dict[str, Any]] = list(entries)
    for user_code, template_code in (confirmed_map or {}).items():
        src = by_norm.get(normalise_zone_code(template_code))
        if not src:
            continue
        overlay.append(
            {
                "zone_code": str(user_code).strip(),
                "zone_name": (
                    f"{src.get('zone_name') or template_code} "
                    f"(сопоставлено с {template_code})"
                ),
                "main": src.get("main") or [],
                "conditional": src.get("conditional") or [],
                "auxiliary": src.get("auxiliary") or [],
            }
        )
    return overlay
