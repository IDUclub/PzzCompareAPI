"""Which zones of a ПЗЗ correspond to each Urban API functional zone type.

An Urban API zone carries only one of its functional zone types (residential,
business, …), while a ПЗЗ has its own territorial zones (Ж-1 … Ж-6, О-1 …). The
language model reads the zones of the document — code, group, name and main ВРИ —
and the types as the built-in template describes them
(``functional_zones_to_pzz_mapping.json``: name, description, characteristic ВРИ),
and names the types of every zone (a mixed zone has several); several answers
vote. Subzones (Ж-1.10, Ж-1.10.2) follow their
base zone (Ж-1), so the model sees base zones only.

When the model is unavailable or answers nonsense, the zones are matched by code
prefix (Ж — residential, О/ОИ/ОД — business, И — industrial and transport, …);
``ZoneTypeMapping.method`` says which way a mapping was made.

Pure: the model call and the cache live in ``service.infrastructure.pzz_regulations``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

METHOD_LLM = "llm"
METHOD_CODE_PREFIX = "code_prefix"

# Urban API functional zone type (``db_name`` without the «_lowrise»-like suffix)
# -> ПЗЗ zone code prefixes of that kind. Only the fallback when the model fails.
ZONE_PREFIXES: dict[str, tuple[str, ...]] = {
    "residential": ("Ж",),
    "business": ("О", "ОИ", "ОД"),
    "mixed": ("О", "ОИ", "ОД"),
    "industrial": ("П", "ПП", "ПК", "И"),
    "recreation": ("Р",),
    "special": ("С", "СП", "СН", "ЛС"),
    "transport": ("Т", "ТД", "И"),
    "agriculture": ("СХ",),
}

# Main ВРИ listed per zone / type in the prompt; the rest is left out.
_MAX_VRI = 12
_PREFIX = re.compile(r"[^\W\d_]+")

# Types without a purpose of their own: no ПЗЗ zone corresponds to them.
UNPROFILED_TYPES = {"basic", "unknown"}

SYSTEM_PROMPT = """Ты относишь территориальные зоны правил землепользования и застройки (ПЗЗ) одного муниципального образования к типам функциональных зон Urban API.

Тип функциональной зоны — обобщённое назначение территории (жилая, общественно-деловая, промышленная и т. д.). Территориальная зона ПЗЗ — зона документа со своим кодом, группой, названием и основными видами разрешённого использования (ВРИ).

Для каждой зоны ПЗЗ укажи типы, которым соответствует её основное назначение. Решай по названию самой зоны, затем по её основным ВРИ; группа — лишь раздел документа и может объединять зоны разного назначения.
- Вид, который в зоне лишь допускается среди прочих (жильё в деловой зоне, спорт или сквер в жилой), тип зоны не меняет.
- Зона смешанного назначения относится ко всем типам, из которых она смешана (зона жилой и общественной застройки — к жилому, общественно-деловому и многофункциональному).
- Зона жилой застройки относится к подтипу своей застройки (ИЖС, малоэтажная, среднеэтажная, многоэтажная), если он ясен из названия, иначе к общему типу «Жилая зона».
- Зоны делового, торгового, социального назначения, здравоохранения, образования, спорта — общественно-деловые.
- Производственные и коммунально-складские зоны — промышленные; зоны транспорта и улично-дорожной сети — транспортные; зоны инженерной инфраструктуры — и промышленные, и транспортные.
- Зона, которой не соответствует ни один тип, получает пустой список.

Перечисли все зоны ПЗЗ из списка, каждую один раз. Ответь в формате JSON."""


@dataclass(frozen=True)
class ZoneTypeMapping:
    """``{functional_zone_type_id: [ПЗЗ zone codes, subzones included]}``."""

    method: str
    by_type: dict[int, list[str]] = field(default_factory=dict)

    def codes(self, fz_type_id: int) -> list[str]:
        return self.by_type.get(fz_type_id, [])


def code_prefix(code: str) -> str:
    """«Ж-2.15» -> «Ж», «ТД.10» -> «ТД», «ПП» -> «ПП»."""
    match = _PREFIX.match((code or "").strip().upper())
    return match.group(0) if match else ""


def base_code(code: str) -> str:
    """«Ж-1.10.2» -> «Ж-1», «ТД.10» -> «ТД»: the zone a subzone belongs to."""
    return (code or "").strip().split(".", 1)[0]


def zone_kind(fz_type_name: str | None) -> str | None:
    """«residential_lowrise» -> «residential»; None for a type without a prefix."""
    name = (fz_type_name or "").strip().lower()
    for kind in ZONE_PREFIXES:
        if name == kind or name.startswith(kind + "_"):
            return kind
    return None


def zone_types(template: dict[str, Any]) -> list[dict[str, Any]]:
    """The functional zone types of the template, with their characteristic ВРИ.

    A ВРИ permitted in the zones of most types (utilities, roads, …) says nothing
    about a type, so only ВРИ found in at most a third of the types are listed.
    """
    entries = [
        e
        for e in template.get("functional_zone_mappings") or []
        if e.get("functional_zone_type_id") is not None
    ]
    uses = {
        int(e["functional_zone_type_id"]): {
            v["vri_code"]: v.get("vri_name") or ""
            for v in (e.get("averaged_pzz_profile") or {}).get("main_vri") or []
            if v.get("vri_code")
        }
        for e in entries
    }
    spread = Counter(code for codes in uses.values() for code in codes)
    limit = max(1, len(entries) // 3)
    types = []
    for e in entries:
        type_id = int(e["functional_zone_type_id"])
        characteristic = [
            (code, name)
            for code, name in uses[type_id].items()
            if spread[code] <= limit
        ]
        types.append(
            {
                "id": type_id,
                "name": e.get("db_name") or "",
                "nickname": e.get("db_zone_nickname") or str(type_id),
                "description": e.get("db_description") or "",
                "vri": characteristic[:_MAX_VRI],
            }
        )
    return types


def base_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The base zones of a ПЗЗ: ``{code, group, name, main}`` in document order."""
    bases: dict[str, dict[str, Any]] = {}
    for zone in zones:
        code = base_code(zone.get("code") or "")
        if not code:
            continue
        main = [
            c
            for use in zone.get("uses") or []
            if use.get("section") == "main"
            for c in use.get("codes") or []
        ]
        entry = bases.get(code)
        if entry is None:
            bases[code] = {
                "code": code,
                "group": zone.get("group") or "",
                "name": zone.get("name") or "",
                "main": list(dict.fromkeys(main)),
            }
        elif zone.get("code") == code:
            # The base zone itself names the zone better than its subzones.
            entry["name"] = zone.get("name") or entry["name"]
            entry["main"] = list(dict.fromkeys(main + entry["main"]))
    return list(bases.values())


def _profiled(types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in types if t["name"] not in UNPROFILED_TYPES]


def mapping_messages(
    types: list[dict[str, Any]], bases: list[dict[str, Any]], document_label: str
) -> list[dict[str, str]]:
    type_lines = []
    for t in _profiled(types):
        line = f"- {t['id']} ({t['name']}) «{t['nickname']}»"
        if t["description"] and t["description"] != "--":
            line += f": {t['description']}"
        if t["vri"]:
            line += ". Характерные ВРИ: " + "; ".join(
                f"{code} {name}".strip() for code, name in t["vri"]
            )
        type_lines.append(line)
    zone_lines = []
    for z in bases:
        line = f"- {z['code']}"
        if z["group"]:
            line += f" [{z['group']}]"
        line += f" {z['name']}"
        if z["main"]:
            line += ". Основные ВРИ: " + ", ".join(z["main"][:_MAX_VRI])
        zone_lines.append(line)
    user = (
        "Типы функциональных зон:\n"
        + "\n".join(type_lines)
        + f"\n\nЗоны ПЗЗ «{document_label}»:\n"
        + "\n".join(zone_lines)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def mapping_schema(
    types: list[dict[str, Any]], bases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Structured output: the types of every base zone, ids and codes enumerated."""
    return {
        "type": "object",
        "properties": {
            "zones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "enum": [z["code"] for z in bases]},
                        "functional_zone_type_ids": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "enum": [t["id"] for t in _profiled(types)],
                            },
                        },
                    },
                    "required": ["code", "functional_zone_type_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["zones"],
        "additionalProperties": False,
    }


def _expand(base_codes: list[str], zones: list[dict[str, Any]]) -> list[str]:
    wanted = set(base_codes)
    return [z["code"] for z in zones if base_code(z.get("code") or "") in wanted]


def parse_mapping(
    answer: dict[str, Any], types: list[dict[str, Any]], zones: list[dict[str, Any]]
) -> ZoneTypeMapping | None:
    """The model's answer turned into the zones of every type; None when it is empty.

    Unknown zone codes and type ids are dropped (the schema should already rule them
    out); a base zone stands for its subzones too.
    """
    known_types = {t["id"] for t in _profiled(types)}
    by_type: dict[int, list[str]] = {t["id"]: [] for t in types}
    for item in answer.get("zones") or []:
        if not isinstance(item, dict) or not isinstance(item.get("code"), str):
            continue
        codes = _expand([item["code"]], zones)
        for type_id in item.get("functional_zone_type_ids") or []:
            if type_id in known_types:
                by_type[type_id] = list(dict.fromkeys(by_type[type_id] + codes))
    # A zone of a subtype (residential_lowrise) is a zone of the type (residential).
    names = {t["name"]: t["id"] for t in types}
    for t in types:
        parent = names.get(t["name"].split("_", 1)[0])
        if parent is not None and parent != t["id"]:
            by_type[parent] = list(dict.fromkeys(by_type[parent] + by_type[t["id"]]))
    if not any(by_type.values()):
        return None
    order = {z["code"]: index for index, z in enumerate(zones)}
    return ZoneTypeMapping(
        METHOD_LLM,
        {k: sorted(v, key=order.__getitem__) for k, v in by_type.items()},
    )


def vote(
    mappings: list[ZoneTypeMapping], zones: list[dict[str, Any]]
) -> ZoneTypeMapping | None:
    """The zones of every type that most of several answers agree on.

    One answer of the model wavers on borderline zones (a health zone as business or
    not); a zone counts for a type when more than half of the answers say so.
    """
    if not mappings:
        return None
    need = len(mappings) // 2 + 1
    votes = Counter(
        (type_id, code)
        for mapping in mappings
        for type_id, codes in mapping.by_type.items()
        for code in codes
    )
    type_ids = list(dict.fromkeys(t for m in mappings for t in m.by_type))
    by_type = {
        type_id: [z["code"] for z in zones if votes[(type_id, z["code"])] >= need]
        for type_id in type_ids
    }
    if not any(by_type.values()):
        return None
    return ZoneTypeMapping(METHOD_LLM, by_type)


def prefix_mapping(
    types: list[dict[str, Any]], zones: list[dict[str, Any]]
) -> ZoneTypeMapping:
    """Zones of every type by code prefix — the fallback without the model."""
    by_type: dict[int, list[str]] = {}
    for t in types:
        kind = zone_kind(t["name"])
        prefixes = ZONE_PREFIXES.get(kind, ()) if kind else ()
        by_type[t["id"]] = [
            z["code"]
            for z in zones
            if z.get("code") and code_prefix(z["code"]) in prefixes
        ]
    return ZoneTypeMapping(METHOD_CODE_PREFIX, by_type)
