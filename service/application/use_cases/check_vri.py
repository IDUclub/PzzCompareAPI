"""Point check of a ВРИ against the ПЗЗ of a territory, as NormGraph reads it.

The compliance agent asks whether a use — a ВРИ code, or an Urban API object type
or service type that maps to one — may stand in a zone. The zone is either

- **an exact ПЗЗ zone** (its code, e.g. «Ж-2.15»): the permitted uses of that zone;
- **an Urban API functional zone type**: the ПЗЗ zones of that type
  (``pzz_zone_types``). A type covers several ПЗЗ zones, so every one of them is
  checked and the answer says how they agree:

  - allowed in all of them -> allowed, in the weakest section of them
    (main < conditional < auxiliary);
  - allowed in none -> not_allowed;
  - allowed in some -> ``depends_on_zone``: the ПЗЗ zone the object stands in
    decides; ``zones`` lists the verdict of each.

The object's floors and height are checked against the limits of each zone.

Pure: the ПЗЗ itself is resolved in ``service.infrastructure.pzz_regulations``.
"""

from __future__ import annotations

from typing import Any

from service.infrastructure.runners._deterministic_pzz import (
    SECTION_RU,
    VERDICT_RU,
    is_allowed,
    normalise_zone_code,
    resolve_po_type_vri,
)

DEPENDS_ON_ZONE = "depends_on_zone"
CHECK_VERDICT_RU = {**VERDICT_RU, DEPENDS_ON_ZONE: "Зависит от зоны ПЗЗ"}

PARAMS_OK = "Соответствует"
PARAMS_EXCEEDED = "Превышены"
PARAMS_NOT_CHECKED = "Не проверено"

# ВРИ of residential buildings: a zone's limits for residential / non-residential
# buildings («для жилых домов», «для нежилых зданий») follow it.
RESIDENTIAL_VRI = {"2.0", "2.1", "2.1.1", "2.2", "2.3", "2.5", "2.6"}
_RESIDENTIAL_PO_TYPE = 4  # urban_api "жилой дом"

_SECTIONS = ("main", "conditional", "auxiliary")
_CHECKED_KINDS = {"max_floors": "этажность", "max_height": "высота"}
_KIND_UNIT = {"max_floors": "эт.", "max_height": "м"}
# Longest zone listing in a reason; the rest is counted.
_MAX_LISTED = 6


def _sections(zone: dict[str, Any]) -> dict[str, set[str]]:
    sections: dict[str, set[str]] = {section: set() for section in _SECTIONS}
    for use in zone.get("uses") or []:
        if use.get("section") in sections:
            sections[use["section"]].update(use.get("codes") or [])
    return sections


def _listing(codes: list[str]) -> str:
    shown = ", ".join(codes[:_MAX_LISTED])
    if len(codes) > _MAX_LISTED:
        shown += f" и ещё {len(codes) - _MAX_LISTED}"
    return shown


def _number(value: float) -> str:
    return f"{value:g}"


def _sentence(parts: list[str]) -> str:
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]


class Regulations:
    """The zones of one ПЗЗ, indexed by code."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.document: dict[str, Any] = payload.get("document") or {}
        self.zones: list[dict[str, Any]] = [
            z for z in payload.get("zones") or [] if z.get("code")
        ]
        self._by_code = {normalise_zone_code(z["code"]): z for z in self.zones}
        self._sections = {z["code"]: _sections(z) for z in self.zones}

    @property
    def label(self) -> str:
        """«Правила землепользования и застройки МО «Город Гатчина», 2019»."""
        name = self.document.get("name") or self.document.get("title") or "ПЗЗ"
        version = self.document.get("version")
        return f"{name}, {version}" if version else name

    def zone(self, code: str | None) -> dict[str, Any] | None:
        return self._by_code.get(normalise_zone_code(code))

    def use_name(self, vri: str) -> str:
        """The name the document gives a use, when one of its zones lists it."""
        for zone in self.zones:
            for use in zone.get("uses") or []:
                if vri in (use.get("codes") or []) and use.get("name"):
                    return use["name"]
        return ""

    def zone_verdict(self, vri: str, zone: dict[str, Any]) -> tuple[str, str]:
        """``(machine_verdict, reason)`` for a use in ``zone``."""
        where = f"зоне {zone['code']} «{zone.get('name') or ''}»"
        sections = self._sections[zone["code"]]
        if not any(sections.values()):
            return (
                "no_zone_metadata",
                f"Для зоны {zone['code']} в ПЗЗ не прочитаны виды разрешённого использования.",
            )
        for section in _SECTIONS:
            if is_allowed(vri, sections[section]):
                return (
                    f"allowed_{section}",
                    f"Тип использования {vri} разрешён в {where}: {SECTION_RU[section]}.",
                )
        return (
            "not_allowed",
            f"Тип использования {vri} не входит в разрешённые в {where}.",
        )

    def check_parameters(
        self,
        zone: dict[str, Any],
        vri: str | None,
        *,
        floors: int | None,
        height: float | None,
        residential: bool,
    ) -> tuple[str, str]:
        """``(label, reason)``: the object's floors and height against the zone's limits.

        A row limited to some uses («для вида 2.7.2», «кроме …») or to residential /
        non-residential buildings applies only to them; a row for the object's own use
        wins over the zone's general row of the same kind. «Не подлежит установлению»
        sets no limit.
        """
        measured = {"max_floors": floors, "max_height": height}
        exceeded: list[str] = []
        within: list[str] = []
        unknown: list[str] = []
        not_set: list[str] = []
        for kind, word in _CHECKED_KINDS.items():
            rows = [
                p
                for p in zone.get("parameters") or []
                if p.get("kind") == kind and _applies(p, vri, residential)
            ]
            if not rows:
                continue
            specific = [p for p in rows if p.get("vri_codes")]
            rows = specific or rows
            limits = [
                (limit, p)
                for p in rows
                if not p.get("not_set") and (limit := _limit(p)) is not None
            ]
            if not limits:
                not_set.append(word)
                continue
            limit, row = min(limits, key=lambda item: item[0])
            unit = _KIND_UNIT[kind]
            where = f"строка {row['number']}" if row.get("number") else "регламент зоны"
            note = ", см. примечание" if row.get("footnote") else ""
            value = measured[kind]
            if value is None:
                unknown.append(
                    f"{word} объекта неизвестна (предельная {_number(limit)} {unit})"
                )
            elif value > limit:
                exceeded.append(
                    f"{word} {_number(value)} {unit} больше предельной "
                    f"{_number(limit)} {unit} ({where}{note})"
                )
            else:
                within.append(
                    f"{word} {_number(value)} {unit} не больше предельной "
                    f"{_number(limit)} {unit} ({where}{note})"
                )
        where = f"Зона {zone['code']}."
        if exceeded:
            return (
                PARAMS_EXCEEDED,
                _sentence(exceeded + within + unknown) + f". {where}",
            )
        if within:
            return PARAMS_OK, _sentence(within + unknown) + f". {where}"
        if unknown:
            return PARAMS_NOT_CHECKED, _sentence(unknown) + f". {where}"
        if not_set:
            return (
                PARAMS_NOT_CHECKED,
                f"Предельная {' и '.join(not_set)} не подлежит установлению. {where}",
            )
        return (
            PARAMS_NOT_CHECKED,
            f"В регламенте зоны нет предельной этажности и высоты для этого объекта. {where}",
        )


def _limit(row: dict[str, Any]) -> float | None:
    value = row.get("value")
    if value is None:
        value = row.get("maximum")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _applies(row: dict[str, Any], vri: str | None, residential: bool) -> bool:
    building = row.get("building")
    if building == "residential" and not residential:
        return False
    if building == "non_residential" and residential:
        return False
    only = row.get("vri_codes") or []
    if only and not (vri and is_allowed(vri, set(only))):
        return False
    excluded = row.get("except_vri_codes") or []
    if excluded and vri and is_allowed(vri, set(excluded)):
        return False
    return True


def resolve_vri(
    *,
    vri_code: str | None,
    physical_object_type_id: int | None,
    service_type_id: int | None,
    floors: int | None,
    po2vri: dict[str, Any],
    service_map: dict[str, Any],
) -> dict[str, Any]:
    """``{code, name, basis}`` of the use asked about; ``code`` is None when unknown.

    An object type maps to a ВРИ like in the scenario check (a residential building by
    its floors), a service type by ``service_type_to_vri``.
    """
    if vri_code:
        return {"code": vri_code.strip(), "name": "", "basis": "ВРИ задан в запросе"}
    if service_type_id is not None:
        entry = service_map.get(str(service_type_id)) or {}
        return {
            "code": entry.get("vri_code") or None,
            "name": entry.get("vri_name") or "",
            "basis": f"по типу сервиса (service_type_id={service_type_id})",
        }
    code, name = resolve_po_type_vri(po2vri, int(physical_object_type_id), floors)
    if physical_object_type_id == _RESIDENTIAL_PO_TYPE:
        floors_text = f", {floors} эт." if floors else " (этажность не указана)"
        basis = f"жилой дом — по этажности{floors_text}"
    else:
        basis = f"по типу объекта (physical_object_type_id={physical_object_type_id})"
    return {"code": code or None, "name": name or "", "basis": basis}


def is_residential(vri: str | None, physical_object_type_id: int | None) -> bool:
    return physical_object_type_id == _RESIDENTIAL_PO_TYPE or vri in RESIDENTIAL_VRI


def _aggregate(vri: str, checked: list[dict[str, Any]], where: str) -> tuple[str, str]:
    """``(machine_verdict, reason)`` over the per-zone verdicts of a zone type."""
    allowed = [z for z in checked if z["verdict"].startswith("allowed_")]
    denied = [z for z in checked if z["verdict"] == "not_allowed"]
    unknown = [z for z in checked if z not in allowed and z not in denied]
    codes = _listing([z["code"] for z in checked])
    if allowed and not denied and not unknown:
        sections = {z["verdict"].removeprefix("allowed_") for z in allowed}
        weakest = max(sections, key=_SECTIONS.index)
        reason = f"Тип использования {vri} разрешён во всех зонах {where} ({codes})"
        if len(sections) > 1:
            reason += f"; в части из них только как {SECTION_RU[weakest]}."
        else:
            reason += f": {SECTION_RU[weakest]}."
        return f"allowed_{weakest}", reason
    if denied and not allowed and not unknown:
        return (
            "not_allowed",
            f"Тип использования {vri} не разрешён ни в одной из зон {where} ({codes}).",
        )
    if allowed:
        reason = (
            f"Тип использования {vri} разрешён в зонах "
            f"{_listing([z['code'] for z in allowed])}"
        )
        if denied:
            reason += f", не разрешён в {_listing([z['code'] for z in denied])}"
        if unknown:
            reason += (
                f", для {_listing([z['code'] for z in unknown])} виды использования "
                "не прочитаны"
            )
        return (
            DEPENDS_ON_ZONE,
            reason + f" — результат зависит от того, в какой из зон {where} "
            "стоит объект.",
        )
    return (
        "unclear",
        f"Для зон {where} ({codes}) не прочитаны виды разрешённого использования, "
        f"чтобы проверить тип {vri}.",
    )


def check(
    regs: Regulations,
    vri: dict[str, Any],
    zones: list[dict[str, Any]],
    *,
    where: str,
    floors: int | None,
    height: float | None,
    residential: bool,
) -> dict[str, Any]:
    """The verdict of a use over ``zones`` (one exact zone or the zones of a type).

    ``where`` names the zones of a type in the reason: «ПЗЗ типа «Жилая зона»».
    """
    code = vri.get("code")
    if code is None:
        return _verdict(
            "unclear",
            "Для объекта нет сопоставленного вида разрешённого использования.",
            [],
        )
    if not zones:
        return _verdict(
            "unclear",
            f"В документе нет зон {where} — тип использования {code} проверить "
            "не по чему.",
            [],
        )
    checked = []
    for zone in zones:
        machine_verdict, reason = regs.zone_verdict(code, zone)
        label, params_reason = regs.check_parameters(
            zone, code, floors=floors, height=height, residential=residential
        )
        checked.append(
            {
                "code": zone["code"],
                "name": zone.get("name") or "",
                "verdict": machine_verdict,
                "verdict_label": CHECK_VERDICT_RU[machine_verdict],
                "reason": reason,
                "parameters": {"status": label, "reason": params_reason},
            }
        )
    if len(checked) == 1:
        machine_verdict, reason = checked[0]["verdict"], checked[0]["reason"]
    else:
        machine_verdict, reason = _aggregate(code, checked, where)
    return _verdict(machine_verdict, reason, checked)


def _verdict(
    machine_verdict: str, reason: str, zones: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "verdict": machine_verdict,
        "verdict_label": CHECK_VERDICT_RU[machine_verdict],
        "reason": reason,
        "zones": zones,
    }
