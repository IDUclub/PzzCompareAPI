"""Convert an uploaded zone-description table (CSV/XLSX) into label-schema JSON.

Separate from the classifier router because this is a pure preview/conversion step
(no task, no pipeline): the frontend converts a spreadsheet, shows the user what
was parsed, and only then feeds the resulting JSON to the building flow as the
zone-descriptions file.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..application.use_cases.convert_zone_table import (
    ConversionError,
    convert_zone_table,
)
from ..application.use_cases.detect_columns import (
    ZONE_TABLE_TARGETS,
    detect_columns_for_file,
)
from ..dependencies import build_ollama_chat_client, get_app_settings
from ..infrastructure.table_reader import TableReadError, read_table
from ..settings import Settings

router = APIRouter(prefix="/pzz/zone-descriptions", tags=["zone-descriptions"])
logger = logging.getLogger("service.api.zone_descriptions")


def _parse_column_map(raw: str | None) -> dict[str, str | None]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"column_map is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="column_map must be a JSON object")
    return {str(k): (str(v) if v is not None else None) for k, v in parsed.items()}


@router.post("/convert")
async def convert_zone_descriptions(
    file: UploadFile = File(...),
    sheet: str | None = Form(default=None),
    column_map: str | None = Form(default=None),
    app_settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Read a CSV/XLSX zone table, detect its columns, and return label JSON + report.

    ``column_map`` (optional JSON object) overrides the auto-detected role→column
    mapping, e.g. ``{"permission": "Раздел", "zone_code": "Индекс"}``.
    """
    data = await file.read()
    if len(data) > app_settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds limit of {app_settings.max_upload_bytes} bytes",
        )
    try:
        rows, headers = read_table(data, file.filename or "", sheet=sheet)
    except TableReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    overrides = _parse_column_map(column_map)
    feature_collection = {"features": [{"properties": r} for r in rows]}
    async with build_ollama_chat_client(app_settings) as ollama_client:
        suggestions = await detect_columns_for_file(
            ollama_client, feature_collection, ZONE_TABLE_TARGETS, heuristic_first=False
        )

    resolved: dict[str, str | None] = {}
    columns_detected: dict[str, Any] = {}
    for target in ZONE_TABLE_TARGETS:
        suggestion = suggestions.get(target.key)
        override = overrides.get(target.key)
        value = override if override else (suggestion.value if suggestion else None)
        resolved[target.key] = value
        columns_detected[target.key] = {
            "column": value,
            "source": "override" if override else (suggestion.source if suggestion else "none"),
            "title": target.title_ru,
        }

    try:
        zones, report = convert_zone_table(rows, resolved)
    except ConversionError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{exc}. Detected columns: {resolved}. Available headers: {headers}. "
                "Pass column_map to set the missing role."
            ),
        ) from exc

    return {
        "zones": zones,
        "columns_detected": columns_detected,
        "report": {
            "zones_count": report.zones_count,
            "vri_count": report.vri_count,
            "rows_total": report.rows_total,
            "rows_used": report.rows_used,
            "warnings": report.warnings,
        },
    }
