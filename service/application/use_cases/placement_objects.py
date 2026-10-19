"""Normalize a user's actual-object layer without treating it as cadastral data."""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...infrastructure.runners._deterministic_pzz import resolve_po_type_vri

if TYPE_CHECKING:
    from ...settings import Settings

OBJECT_TYPE_TEXT = "placement_object_type"
TEXT_COLUMNS = (
    "тип объекта",
    "тип",
    "object_type",
    "service_type_name",
    "physical_object_type_name",
    "service_name",
    "название типа",
    "vri",
    "vri_name",
    "vri_code",
    "ВРИ_участка",
    "name",
    "название",
)


def _normalise_name(name: str) -> str:
    return re.sub(r"[\W_]+", "", name.casefold().replace("ё", "е"))


def _id_key(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_placement_objects(
    fc: dict[str, Any], settings: "Settings", type_col: str | None = None
) -> dict[str, Any]:
    """Accept Urban API nested/flat ids or an explicit/known text-type column.

    Empty layers are meaningful: they override cadastral guesses with no known
    existing target objects. The original properties are retained for the map.
    """
    physical = json.loads(
        Path(settings.physical_object_type_to_vri_path).read_text(encoding="utf-8")
    )
    services = json.loads(
        Path(settings.service_type_to_vri_path).read_text(encoding="utf-8")
    ).get("by_service_type_id", {})
    features = fc.get("features") or []
    columns = {k for f in features for k in (f.get("properties") or {})}
    if type_col and features and type_col not in columns:
        raise ValueError(f"Колонка типа объекта «{type_col}» отсутствует в файле.")
    by_name = {_normalise_name(k): k for k in columns}
    if not type_col:
        type_col = next(
            (
                by_name[_normalise_name(k)]
                for k in TEXT_COLUMNS
                if _normalise_name(k) in by_name
            ),
            None,
        )
    structured = {
        "service_type",
        "service_type_id",
        "physical_object_type",
        "physical_object_type_id",
    }
    if features and not type_col and not columns.intersection(structured):
        raise ValueError(
            "Не найдена колонка типа объекта. Укажите existing_objects_type_col или передайте типы Urban API."
        )
    normalized = []
    for f in features:
        geometry = f.get("geometry")
        if geometry and geometry.get("type") not in {
            "Point",
            "MultiPoint",
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError("Фактические объекты должны быть точками или полигонами.")
        props = dict(f.get("properties") or {})
        text = str(props.get(type_col) or "").strip() if type_col else ""
        code = None
        service = props.get("service_type") or {}
        service = service if isinstance(service, dict) else {"name": str(service)}
        sid = props.get(
            "service_type_id", service.get("service_type_id", service.get("id"))
        )
        if sid is not None or service:
            entry = services.get(_id_key(sid)) or {}
            code = entry.get("vri_code")
            text = " ".join(
                str(s) for s in [service.get("name"), entry.get("name"), text] if s
            )
        else:
            po = props.get("physical_object_type") or {}
            po = po if isinstance(po, dict) else {"name": str(po)}
            pid = props.get(
                "physical_object_type_id",
                po.get("physical_object_type_id", po.get("id")),
            )
            nested = props.get("properties") or {}
            floors = (
                nested.get("Количество этажей") if isinstance(nested, dict) else None
            )
            if pid is not None:
                try:
                    code, label = resolve_po_type_vri(
                        physical, int(pid), floors or props.get("Количество этажей")
                    )
                except (ValueError, TypeError):
                    label = None
                text = " ".join(str(s) for s in [po.get("name"), label, text] if s)
            elif po.get("name"):
                text = " ".join([str(po["name"]), text])
        props[OBJECT_TYPE_TEXT] = f"{text} (ВРИ {code})" if code else text
        normalized.append({**f, "properties": props})
    return {**fc, "features": normalized}
