"""LLM-assisted detection of the classifier's input columns.

The pipeline needs three column names to run against an uploaded layer:

- ``cadastral_vri_col``  — the parcel's VRI (permitted-use) text (cadastral layer);
- ``pzz_zone_code_col``  — the PZZ zone index/code (zones layer);
- ``pzz_zone_name_col``  — the PZZ zone name (zones layer).

Historically the frontend passed these explicitly. This module detects them from
the uploaded data so the chat agent can say "I recognised field X as the VRI
name" and proceed, letting the user correct it in a follow-up (phase 2).

Two-pass strategy per target:

1. a cheap heuristic — an EXACT (normalised) match of a column name against the
   target's known names — resolves the standard schemas without an LLM call;
2. for everything else, a single ``complete_json`` call with a JSON-schema whose
   ``enum`` is the layer's real column names, so the model physically cannot
   return a non-existent field.

Fuzzy name matching was deliberately dropped: layers commonly carry a numeric
"code companion" column named ``Код_<known>`` (e.g. ``Код_Индекс_зоны`` next to
``Индекс_зоны``). When the real column is named non-standardly, a fuzzy matcher
latches onto that companion and confidently picks the wrong (numeric) column —
and, worse, preempts the LLM. So non-standard names defer to the LLM, which is
the whole point of the feature.

``confidence`` is derived from the source (heuristic-exact 1.0 / llm 0.6 / none
0.0), not from the model — LLM self-reported confidence is unreliable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError

logger = logging.getLogger("service.detect_columns")

_HEURISTIC_EXACT = 1.0
_LLM = 0.6
_NONE = 0.0


@dataclass(frozen=True)
class ColumnProfile:
    """A compact profile of one layer column (no geometry, no full values)."""

    name: str
    dtype: str  # "str" | "int" | "float" | "bool" | "mixed" | "null"
    n_unique: int
    samples: list[str]


@dataclass(frozen=True)
class ColumnSuggestion:
    """A detected column for one target, with provenance for the narrative/UI."""

    value: str | None
    confidence: float
    source: str  # "heuristic" | "llm" | "none"
    reason: str
    candidates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionTarget:
    """A column role to detect, with heuristic hints and human-facing text."""

    key: str
    title_ru: str
    description_ru: str
    known_names: tuple[str, ...]


VRI_TARGET = DetectionTarget(
    key="cadastral_vri_col",
    title_ru="вид разрешённого использования (ВРИ) участка",
    description_ru=(
        "Текстовое название вида разрешённого использования (ВРИ) земельного "
        "участка — обычно длинные формулировки, напр. «Для индивидуального "
        "жилищного строительства», «Многоэтажная жилая застройка»."
    ),
    known_names=(
        "Вид_разрешенного_исп",
        "вид разрешенного использования",
        "vri_name",
        "vri",
        "ври",
    ),
)
ZONE_CODE_TARGET = DetectionTarget(
    key="pzz_zone_code_col",
    title_ru="код (индекс) территориальной зоны ПЗЗ",
    description_ru=(
        "Короткий индекс/шифр территориальной зоны ПЗЗ — напр. «Ж-1», «О-2», "
        "«П-3». Это НЕ длинное текстовое наименование зоны."
    ),
    known_names=("Индекс_зоны", "индекс зоны", "zone_code", "index", "индекс"),
)
ZONE_NAME_TARGET = DetectionTarget(
    key="pzz_zone_name_col",
    title_ru="наименование территориальной зоны ПЗЗ",
    description_ru=(
        "Человекочитаемое НАИМЕНОВАНИЕ территориальной зоны — длинный текст, "
        "напр. «Производственная зона», «Зона застройки малоэтажными жилыми "
        "домами». Это НЕ короткий индекс/шифр вида «Ж-1», «П-2» и НЕ числовой код."
    ),
    known_names=("Код_объекта", "zone_name", "name", "наименование"),
)

CADASTRAL_TARGETS: list[DetectionTarget] = [VRI_TARGET]
PZZ_ZONE_TARGETS: list[DetectionTarget] = [ZONE_CODE_TARGET, ZONE_NAME_TARGET]

BUILDING_TYPE_TARGET = DetectionTarget(
    key="building_type_col",
    title_ru="тип здания (жилое/нежилое)",
    description_ru=(
        "Тип здания. Обычно числовой physical_object_type_id из Urban API "
        "(напр. 4 — жилой дом), либо текстовое название типа объекта "
        "(«жилой дом», «склад»). Определяет, жилое ли здание, и участвует "
        "в подборе ВРИ."
    ),
    known_names=(
        "physical_object_type_id",
        "physical_object_type_name",
        "physical_object_type",
        "тип",
        "тип_здания",
        "название_типа",
        "building_type",
        "building_type_name",
        "po_type_id",
        "po_type_name",
    ),
)
BUILDING_SERVICE_TARGET = DetectionTarget(
    key="building_service_col",
    title_ru="сервис здания (service_type_id)",
    description_ru=(
        "Тип сервиса здания — числовой service_type_id из Urban API или "
        "текстовое название/код сервиса (напр. «школа», «детский сад», "
        "«поликлиника», school). Используется для подбора ВРИ нежилых "
        "зданий. НЕ этажность и НЕ тип здания."
    ),
    known_names=(
        "service_type_id",
        "service_type_name",
        "service_type",
        "service_name",
        "сервис",
        "название_сервиса",
        "наименование_сервиса",
        "service",
        "тип_сервиса",
    ),
)
BUILDING_FLOORS_TARGET = DetectionTarget(
    key="building_floors_col",
    title_ru="этажность здания",
    description_ru=(
        "Количество этажей здания — целое число (напр. 1, 5, 24). Для жилых "
        "зданий определяет ВРИ по этажности. НЕ тип и НЕ сервис."
    ),
    known_names=(
        "floors_count",
        "floors",
        "этажность",
        "количество этажей",
        "этажей",
        "number_of_floors",
    ),
)

BUILDING_TARGETS: list[DetectionTarget] = [
    BUILDING_TYPE_TARGET,
    BUILDING_SERVICE_TARGET,
    BUILDING_FLOORS_TARGET,
]


def _normalise(name: str) -> str:
    """Lowercase and strip separators so ``VRI_name`` ~= ``vri name``."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _dtype_of(values: list[Any]) -> str:
    types = {type(v) for v in values if v is not None}
    if not types:
        return "null"
    if types == {bool}:
        return "bool"
    if types <= {int, bool}:
        return "int"
    if types <= {int, float, bool}:
        return "float"
    if types == {str}:
        return "str"
    return "mixed"


def profile_columns(
    feature_collection: dict[str, Any],
    *,
    max_sample_values: int = 5,
    max_value_chars: int = 80,
    scan_features: int = 50,
) -> list[ColumnProfile]:
    """Profile a GeoJSON layer's columns from the ``properties`` of its features.

    Geometry lives outside ``properties`` in GeoJSON, so it is excluded for
    free. Column *discovery* scans EVERY feature, so a column that only appears
    in later features is not missed — e.g. a merged buildings+services layer
    where ``service_type_id`` first shows up mid-file (after all the physical
    objects). Per column it keeps up to ``scan_features`` example values (for
    dtype / n_unique) and up to ``max_sample_values`` distinct non-null samples
    (each truncated to ``max_value_chars``).
    """
    features = feature_collection.get("features") or []
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    values_by_col: dict[str, list[Any]] = {}
    samples_by_col: dict[str, list[str]] = {}
    seen_samples: dict[str, set[str]] = {}

    for feature in features:
        props = (feature or {}).get("properties") or {}
        for name, value in props.items():
            if name not in seen_names:
                seen_names.add(name)
                ordered_names.append(name)
                values_by_col[name] = []
                samples_by_col[name] = []
                seen_samples[name] = set()
            if len(values_by_col[name]) < scan_features:
                values_by_col[name].append(value)
            if value is None or value == "":
                continue
            text = str(value)
            if len(text) > max_value_chars:
                text = text[:max_value_chars] + "…"
            if text not in seen_samples[name] and len(samples_by_col[name]) < max_sample_values:
                seen_samples[name].add(text)
                samples_by_col[name].append(text)

    profiles: list[ColumnProfile] = []
    for name in ordered_names:
        non_null = [v for v in values_by_col[name] if v is not None and v != ""]
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=_dtype_of(values_by_col[name]),
                n_unique=len({str(v) for v in non_null}),
                samples=samples_by_col[name],
            )
        )
    return profiles


def _heuristic_match(
    target: DetectionTarget, profiles: list[ColumnProfile]
) -> ColumnSuggestion | None:
    """Resolve a target by an EXACT (normalised) column-name match, else None.

    Only exact hits are trusted; anything else defers to the LLM (see the module
    docstring on why fuzzy matching is unsafe here).
    """
    names = [p.name for p in profiles]
    known_norm = {_normalise(k) for k in target.known_names}
    for name in names:
        if _normalise(name) in known_norm:
            return ColumnSuggestion(
                value=name,
                confidence=_HEURISTIC_EXACT,
                source="heuristic",
                reason=f"имя колонки «{name}» совпадает с известным для роли",
                candidates=names,
            )
    return None


def _build_detection_messages(
    targets: list[DetectionTarget], profiles: list[ColumnProfile]
) -> list[dict[str, str]]:
    system = (
        "Ты — помощник по геоданным для проверки ПЗЗ (правила землепользования "
        "и застройки). Тебе дают колонки одного слоя (имя, тип, число уникальных "
        "значений, примеры) и роли, которые нужно сопоставить с колонками.\n"
        "Правила:\n"
        "- Для каждой роли выбери РОВНО ОДНУ колонку ИЗ ПРЕДЛОЖЕННОГО СПИСКА имён.\n"
        "- Если подходящей колонки нет — верни null. Никогда не придумывай имена.\n"
        "- Ориентируйся и на имя колонки, и на примеры значений.\n"
        "- Верни строго JSON по заданной схеме, без пояснений вне JSON."
    )
    roles_lines = "\n".join(f"- {t.key}: {t.description_ru}" for t in targets)
    columns_json = _profiles_as_json(profiles)
    user = (
        "Роли, которые нужно определить:\n"
        f"{roles_lines}\n\n"
        "Колонки слоя (JSON):\n"
        f"{columns_json}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _profiles_as_json(profiles: list[ColumnProfile]) -> str:
    import json

    return json.dumps(
        [
            {
                "name": p.name,
                "dtype": p.dtype,
                "n_unique": p.n_unique,
                "samples": p.samples,
            }
            for p in profiles
        ],
        ensure_ascii=False,
    )


def _detection_schema(
    targets: list[DetectionTarget], column_names: list[str]
) -> dict[str, Any]:
    """JSON schema whose per-target ``column`` enum is the real column names."""
    enum_values: list[Any] = [*column_names, None]
    properties = {
        t.key: {
            "type": "object",
            "properties": {
                "column": {"type": ["string", "null"], "enum": enum_values},
                "reason": {"type": "string"},
            },
            "required": ["column", "reason"],
        }
        for t in targets
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [t.key for t in targets],
    }


async def detect_columns_for_file(
    ollama_client: OllamaChatClient,
    feature_collection: dict[str, Any],
    targets: list[DetectionTarget],
    *,
    model: str | None = None,
    max_sample_values: int = 5,
) -> dict[str, ColumnSuggestion]:
    """Detect each target's column: heuristic first, then one LLM call for the rest."""
    profiles = profile_columns(feature_collection, max_sample_values=max_sample_values)
    names = [p.name for p in profiles]
    suggestions: dict[str, ColumnSuggestion] = {}

    unresolved: list[DetectionTarget] = []
    for target in targets:
        hit = _heuristic_match(target, profiles)
        if hit is not None:
            suggestions[target.key] = hit
        else:
            unresolved.append(target)

    if not unresolved or not names:
        for target in unresolved:
            suggestions[target.key] = ColumnSuggestion(
                value=None,
                confidence=_NONE,
                source="none",
                reason="подходящая колонка не найдена",
                candidates=names,
            )
        return suggestions

    messages = _build_detection_messages(unresolved, profiles)
    schema = _detection_schema(unresolved, names)
    try:
        result = await ollama_client.complete_json(messages, schema=schema, model=model)
    except (OllamaChatError, httpx.HTTPError) as exc:
        # LLM down/unreachable must not 500 the auto endpoint — the unresolved
        # targets simply fall back to "not found" (heuristic-only detection).
        logger.warning("column-detection LLM call failed: %s", exc)
        result = {}

    name_set = set(names)
    for target in unresolved:
        block = result.get(target.key) if isinstance(result, dict) else None
        column = (block or {}).get("column") if isinstance(block, dict) else None
        reason = (block or {}).get("reason") if isinstance(block, dict) else None
        if isinstance(column, str) and column in name_set:
            suggestions[target.key] = ColumnSuggestion(
                value=column,
                confidence=_LLM,
                source="llm",
                reason=(reason if isinstance(reason, str) and reason else "определено моделью"),
                candidates=names,
            )
        else:
            suggestions[target.key] = ColumnSuggestion(
                value=None,
                confidence=_NONE,
                source="none",
                reason="модель не смогла определить подходящую колонку",
                candidates=names,
            )
    return suggestions


def render_detection_narrative(
    suggestions: dict[str, ColumnSuggestion],
    targets: list[DetectionTarget],
) -> str:
    """Build the RU chat message announcing the detected columns."""
    title_by_key = {t.key: t.title_ru for t in targets}
    resolved: list[str] = []
    missing: list[str] = []
    for key, title in title_by_key.items():
        suggestion = suggestions.get(key)
        if suggestion is not None and suggestion.value:
            resolved.append(f"- поле «{suggestion.value}» определено как {title}")
        else:
            missing.append(title)

    lines: list[str] = []
    if resolved:
        lines.append(
            "Результат анализа содержания полей в загруженном файле "
            "для проверки по правилам землепользования и застройки (ПЗЗ):"
        )
        lines.extend(resolved)
    if missing:
        if resolved:
            lines.append("")
        lines.append(
            "Не удалось определить: "
            + ", ".join(missing)
            + ". Уточните нужные поля, и я пересчитаю."
        )
    return "\n".join(lines)


def required_columns_resolved(
    suggestions: dict[str, ColumnSuggestion],
    targets: list[DetectionTarget],
) -> bool:
    """True when every target has a concrete column value."""
    return all(
        (suggestions.get(t.key) is not None and bool(suggestions[t.key].value))
        for t in targets
    )
