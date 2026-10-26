"""Check scenario object uses against the cadastral parcels covering each object."""

import re
from typing import Any

CADASTRE_CHECK_KEY = "cadastre_check"
PARCEL_IDS = "Кадастровые_участки"
PARCEL_VRI = "ВРИ_участков"
PARCEL_CODES = "Коды_ВРИ_участков"
PARCEL_VERDICT = "Проверка_по_ВРИ_участков"
PARCEL_REASON = "Причина_проверки_по_ВРИ_участков"

_CODE = re.compile(r"(?<![\w.])\d{1,2}(?:\.\d{1,2}){0,3}(?![\w.])")


def _name(value: str) -> str:
    text = re.sub(r"[^\w]+", " ", value.casefold().replace("ё", "е")).strip()
    return text.removeprefix("для ")


class CadastreChecker:
    def __init__(self, doc: dict[str, Any], vri_col: str, names: dict[str, str]):
        from shapely.geometry import shape
        from shapely.strtree import STRtree

        self.features = doc.get("features") or []
        self.geometries = [shape(f["geometry"]) for f in self.features]
        self.tree = STRtree(self.geometries)
        self.vri_col = vri_col
        self.known_codes = set(names)
        self.by_name: dict[str, set[str]] = {}
        for code, name in names.items():
            self.by_name.setdefault(_name(name), set()).add(code)
        self.parsed = [
            self._codes((f.get("properties") or {}).get(vri_col)) for f in self.features
        ]

    def _codes(self, value: Any) -> tuple[set[str], bool]:
        """Resolve explicit codes or exact classifier names; never guess unknown text."""
        if isinstance(value, list):
            parts = [self._codes(item) for item in value]
            return set().union(*(codes for codes, _ in parts)), any(u for _, u in parts)
        text = str(value).strip() if value is not None else ""
        if not text:
            return set(), True
        tokens = set(_CODE.findall(text))
        codes = tokens & self.known_codes
        if codes:
            return codes, bool(tokens - self.known_codes)
        parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
        resolved = [self.by_name.get(_name(part), set()) for part in parts]
        # Duplicate classifier names with different codes are ambiguous.
        return set().union(*(c for c in resolved if len(c) == 1)), not resolved or any(
            len(c) != 1 for c in resolved
        )

    @staticmethod
    def _permits(object_code: str, codes: set[str]) -> bool:
        for code in codes:
            parent = code[:-2] if code.endswith(".0") else code
            if (
                object_code == code
                or object_code == parent
                or object_code.startswith(parent + ".")
            ):
                return True
        return False

    def check(
        self, feature: dict[str, Any], object_code: str | None
    ) -> tuple[bool | None, dict[str, Any]]:
        from shapely.geometry import shape
        from shapely.ops import unary_union

        geometry = shape(feature["geometry"])
        indices = []
        for index in sorted(self.tree.query(geometry, predicate="intersects")):
            intersection = geometry.intersection(self.geometries[index])
            if geometry.area and not intersection.area:
                continue  # A neighbouring parcel touched by the boundary only.
            if not geometry.area and geometry.length and not intersection.length:
                continue
            indices.append(int(index))

        ids, texts, parsed, checks = [], [], [], []
        for index in indices:
            feature_props = self.features[index].get("properties") or {}
            parcel_id = next(
                (
                    feature_props[k]
                    for k in (
                        "cadastral_number",
                        "cad_num",
                        "Кадастровый номер",
                        "Кадастровый_номер",
                        "cn",
                    )
                    if feature_props.get(k)
                ),
                self.features[index].get("id", f"участок №{index + 1}"),
            )
            value = feature_props.get(self.vri_col)
            codes, unresolved = self.parsed[index]
            ids.append(str(parcel_id))
            texts.append(value)
            parsed.append(sorted(codes))
            if object_code and self._permits(object_code, codes):
                checks.append(True)
            elif (
                not object_code
                or unresolved
                or any(code.startswith(object_code + ".") for code in codes)
            ):
                checks.append(None)
            else:
                checks.append(False)

        uncovered = geometry.difference(
            unary_union([self.geometries[i] for i in indices])
        )
        partial = (
            uncovered.area > geometry.area * 1e-6
            if geometry.area
            else (
                uncovered.length > geometry.length * 1e-6
                if geometry.length
                else not uncovered.is_empty
            )
        )
        if not indices:
            status, reason = (
                None,
                "Объект не попал ни в один загруженный кадастровый участок.",
            )
        elif False in checks:
            denied = ", ".join(
                ids[i] for i, result in enumerate(checks) if result is False
            )
            status, reason = (
                False,
                f"Тип использования {object_code} не соответствует ВРИ участка: {denied}.",
            )
        elif partial:
            status, reason = (
                None,
                "Объект частично выходит за границы загруженных кадастровых участков.",
            )
        elif not object_code:
            status, reason = (
                None,
                "Тип использования объекта не определён; сравнение с ВРИ участка требует ручной проверки.",
            )
        elif None in checks:
            status, reason = (
                None,
                "ВРИ участка не распознан однозначно по классификатору; требуется ручная проверка.",
            )
        else:
            status, reason = (
                True,
                f"Тип использования {object_code} соответствует ВРИ покрывающих объект участков.",
            )
        return status, {
            PARCEL_IDS: ids,
            PARCEL_VRI: texts,
            PARCEL_CODES: parsed,
            PARCEL_VERDICT: (
                "Соответствует"
                if status is True
                else (
                    "Не соответствует"
                    if status is False
                    else "Требуется ручная проверка"
                )
            ),
            PARCEL_REASON: reason,
        }
