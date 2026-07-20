"""Scaffold ``data/service_type_to_vri.json`` from ``data/services_hierarchy.json``.

Deterministic ``service_type_id -> Rosreestr VRI`` lookup, the service-object
counterpart of ``data/physical_object_type_to_vri.json``. Used by the uploaded
building PZZ check to resolve a building's ``service_type_id`` to a VRI code
before matching it against a zone's permitted-VRI set.

The VRI assignments below are a BEST-EFFORT first pass keyed on each service
type's English ``code`` (urban_api service_type catalogue, 103 types). They need
domain validation — an unmapped / ``null`` code is safe: it degrades to the
"Требуется ручная проверка" verdict downstream, exactly like an unresolved
physical_object_type. Re-run after editing ``_CODE_TO_VRI``:

    python -m scripts.build_service_type_to_vri
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SRC = Path("data/services_hierarchy.json")
_DST = Path("data/service_type_to_vri.json")

# service_type ``code`` (English, from urban_api) -> Rosreestr VRI code.
# Grouped by Rosreestr section for reviewability. ``None`` = needs manual review
# (ambiguous / not a building-shaped use); resolves to "manual check" downstream.
_CODE_TO_VRI: dict[str, str | None] = {
    # 3.1 коммунальное обслуживание
    "boiler room": "3.1.1",
    "water works": "3.1.1",
    "pumping station": "3.1.1",
    "wastewater plant": "3.1.1",
    "transformer substation": "3.1.1",
    # 3.2 социальное обслуживание
    "integrated social service center": "3.2.2",
    "nursing home": "3.2.2",
    "employment center": "3.2.2",
    "orphanage": "3.2.2",
    "post office": "3.2.3",
    # 3.3 бытовое обслуживание
    "hairdresser": "3.3",
    "beauty salon": "3.3",
    "bathhouse": "3.3",
    # 3.4 здравоохранение
    "polyclinic": "3.4.1",
    "dental clinic": "3.4.1",
    "female consultation": "3.4.1",
    "pharmacy": "3.4.1",
    "hospital": "3.4.2",
    "maternity": "3.4.2",
    "ambulance station": "3.4.2",
    "trauma center": "3.4.2",
    # 3.5 образование и просвещение
    "kindergarden": "3.5.1",
    "school": "3.5.1",
    "house of children's art": "3.5.1",
    "summer camp": "3.5.1",
    "college": "3.5.2",
    "university": "3.5.2",
    # 3.6 культурное развитие
    "library": "3.6.1",
    "palace of culture": "3.6.1",
    "museum": "3.6.1",
    "theatre": "3.6.1",
    "concert hall": "3.6.1",
    "cinema": "3.6.1",
    "botanical garden": "3.6.3",
    "circus": "3.6.3",
    "zoo": "3.6.3",
    # 3.7 религиозное использование
    "religion": "3.7.1",
    "monastery": "3.7.1",
    # 3.8 общественное управление
    "multifunctional center": "3.8.1",
    "registry office": "3.8.1",
    # 3.10 ветеринарное обслуживание
    "veterinary": "3.10.1",
    # 4.1 деловое управление
    "lawyer": "4.1",
    # 4.2 объекты торговли (ТЦ)
    "mall": "4.2",
    # 4.3 рынки
    "market": "4.3",
    # 4.4 магазины
    "supermarket": "4.4",
    "convenience": "4.4",
    "houseware": "4.4",
    "clothes": "4.4",
    "appliances": "4.4",
    "bookshop": "4.4",
    "baby goods": "4.4",
    "sport shop": "4.4",
    "petshop": "4.4",
    "dropsite": "4.4",
    "bakery": "4.4",
    # 4.5 банковская и страховая деятельность
    "bank": "4.5",
    "atm": "4.5",
    # 4.6 общественное питание
    "cafe": "4.6",
    "restaurant": "4.6",
    "bar": "4.6",
    # 4.7 гостиничное обслуживание
    "hotel": "4.7",
    "hostel": "4.7",
    # 4.8 развлечения
    "water park": "4.8",
    "theme park": "4.8",
    # 4.9 обслуживание автотранспорта / дорожный сервис
    "fuel": "4.9.1.1",
    "parking": "4.9",
    # 5.0 отдых (рекреация)
    "beach": "5.0",
    "recreation center": "5.0",
    "dog park": "5.0",
    # 5.1 спорт
    "stadium": "5.1.1",
    "ice arena": "5.1.1",
    "sports center": "5.1.1",
    "climbing gym": "5.1.1",
    "pitch": "5.1.3",
    "swimmimg pool": "5.1.3",
    "skating rink": "5.1.3",
    "skatepark": "5.1.3",
    "playground": "5.1.3",
    # 5.2 природно-познавательный туризм
    "eco path": "5.2",
    # 6.0 производственная деятельность
    "industrial area": "6.0",
    # 6.7 энергетика
    "power plant": "6.7",
    "nuclear plant": "6.7",
    "hydropower plant": "6.7",
    "thermal plant": "6.7",
    "renewable energy source": "6.7",
    # 7 транспорт
    "train station": "7.1",
    "train building": "7.1",
    "bus terminal": "7.2.1",
    "bus stop": "7.2.1",
    "airport": "7.4",
    # 8 обеспечение обороны/правопорядка
    "police": "8.3",
    "fire station": "8.3",
    "prison": "8.4",
    # 9 деятельность по особой охране / культурная
    "sanatorium": "9.2.1",
    "monunent": "9.3",  # sic: source spelling of "monument"
    # 12 ритуальная / специальная
    "cemetery": "12.1",
    "mortuary": "12.1",
    "crematorium": "12.1",
    # not building-shaped / ambiguous -> manual review
    "park": "3.6.2",
    "wood": None,
    "reserve": None,
    "protected area": None,
    "stable": None,
}


def _iter_leaves(nodes: list[dict[str, Any]]):
    for node in nodes:
        if "service_type_id" in node:
            yield node
        yield from _iter_leaves(node.get("children") or [])


def build() -> dict[str, Any]:
    hierarchy = json.loads(_SRC.read_text(encoding="utf-8"))
    by_id: dict[str, Any] = {}
    unmapped: list[str] = []
    for leaf in _iter_leaves(hierarchy):
        raw_code = leaf.get("code")
        # Source codes occasionally carry trailing whitespace ("nursing home ").
        code = raw_code.strip() if isinstance(raw_code, str) else raw_code
        vri = _CODE_TO_VRI.get(code, None)
        if code not in _CODE_TO_VRI:
            unmapped.append(f"{leaf.get('service_type_id')}:{code}")
        by_id[str(leaf["service_type_id"])] = {
            "name": leaf.get("name"),
            "code": code,
            "vri_code": vri,
            "vri_name": "",
        }
    resolved = sum(1 for v in by_id.values() if v["vri_code"])
    return {
        "metadata": {
            "description": (
                "Deterministic service_type_id -> Rosreestr VRI code, used by the "
                "uploaded-building PZZ check. Best-effort scaffold; needs domain "
                "validation. null vri_code -> 'manual check' verdict downstream."
            ),
            "source": "urban_api service_type catalogue (data/services_hierarchy.json)",
            "total": len(by_id),
            "resolved": resolved,
        },
        "by_service_type_id": by_id,
        "_unmapped_codes": sorted(set(unmapped)),
    }


def main() -> None:
    result = build()
    _DST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = result["metadata"]
    print(
        f"wrote {_DST}: {meta['resolved']}/{meta['total']} resolved; "
        f"unmapped codes: {result['_unmapped_codes'] or 'none'}"
    )


if __name__ == "__main__":
    main()
