from __future__ import annotations

from .common import *

import re

from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

import geopandas as gpd

def build_zone_allowed_vri_codes(zone_templates: List[dict[str, Any]], include_conditional: bool=False) -> Dict[str, Set[str]]:
    """
    Build mapping: zone_code -> allowed VRI codes.
    """
    zone_allowed: Dict[str, Set[str]] = {}
    for zone in zone_templates:
        zone_code = str(zone.get('zone_code') or '').strip()
        if not zone_code:
            continue
        allowed_codes: Set[str] = set()
        for section_name in ['main', 'auxiliary']:
            for item in zone.get(section_name, []) or []:
                code = str(item.get('vri_code') or '').strip()
                if code:
                    allowed_codes.add(code)
        if include_conditional:
            for item in zone.get('conditional', []) or []:
                code = str(item.get('vri_code') or '').strip()
                if code:
                    allowed_codes.add(code)
        zone_allowed[zone_code] = allowed_codes
        base_zone_code = str(zone.get('base_zone_code') or '').strip()
        if base_zone_code and base_zone_code not in zone_allowed:
            zone_allowed[base_zone_code] = set(allowed_codes)
    return zone_allowed

def extract_vri_codes_from_candidates(value: Any) -> List[str]:
    """
    Extract VRI codes from serialized candidate string.
    Example:
    '4.6 Общественное питание, 4.7 Гостиничное обслуживание'
    -> ['4.6', '4.7']
    """
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(',') if part.strip()]
    codes: List[str] = []
    for part in parts:
        match = re.match('^(\\d+(?:\\.\\d+)*)\\b', part)
        if match:
            codes.append(match.group(1))
    return codes

def find_allowed_candidate_codes(actual_zone_code: Any, candidate_codes: Iterable[str], zone_allowed_map: Dict[str, Set[str]]) -> List[str]:
    """
    Return candidate codes that are definitely allowed in the actual zone.
    """
    zone_code = str(actual_zone_code or '').strip()
    if not zone_code:
        return []
    allowed_codes = zone_allowed_map.get(zone_code, set())
    if not allowed_codes:
        parts = zone_code.split('.')
        for i in range(len(parts) - 1, 0, -1):
            fallback_code = '.'.join(parts[:i])
            if fallback_code in zone_allowed_map:
                allowed_codes = zone_allowed_map[fallback_code]
                break
    return [code for code in candidate_codes if code in allowed_codes]

def mark_manual_review_for_allowed_top_candidates(gdf: gpd.GeoDataFrame, zone_templates: List[dict[str, Any]], actual_zone_col: str, verdict_col: str, status_col: str, candidates_col: str, reason_col: str, include_conditional: bool=False) -> gpd.GeoDataFrame:
    """
    Mark row as manual review if any top candidate is allowed in actual zone.
    """
    result = gdf.copy()
    zone_allowed_map = build_zone_allowed_vri_codes(zone_templates=zone_templates, include_conditional=include_conditional)
    result['HAS_ALLOWED_CANDIDATE_IN_ACTUAL_ZONE'] = False
    result['ALLOWED_TOP_CANDIDATE_CODES'] = None
    for idx in result.index:
        verdict = str(result.at[idx, verdict_col] or '').strip().lower()
        if verdict != 'not_allowed':
            continue
        actual_zone_code = result.at[idx, actual_zone_col]
        candidate_codes = extract_vri_codes_from_candidates(result.at[idx, candidates_col])
        if not candidate_codes:
            continue
        allowed_candidate_codes = find_allowed_candidate_codes(actual_zone_code=actual_zone_code, candidate_codes=candidate_codes, zone_allowed_map=zone_allowed_map)
        if not allowed_candidate_codes:
            continue
        result.at[idx, 'HAS_ALLOWED_CANDIDATE_IN_ACTUAL_ZONE'] = True
        result.at[idx, 'ALLOWED_TOP_CANDIDATE_CODES'] = ', '.join(allowed_candidate_codes)
        result.at[idx, status_col] = 'Требуется ручная проверка'
        old_reason = str(result.at[idx, reason_col] or '').strip()
        extra_reason = 'Среди Top-кандидатов есть ВРИ, разрешённые в фактической зоне: ' + ', '.join(allowed_candidate_codes) + '.'
        result.at[idx, reason_col] = f'{old_reason} {extra_reason}'.strip() if old_reason else extra_reason
    return result

def build_zone_name_lookup(zone_templates: List[dict[str, Any]], prefer_heading: bool=True) -> Dict[str, str]:
    """
    Build mapping: zone_code -> zone display name.

    Parameters
    ----------
    zone_templates : List[dict[str, Any]]
        Loaded zone templates from pzz_zone_llm_labels_template.json.
    prefer_heading : bool
        If True, use zone_heading first, otherwise use zone_name first.

    Returns
    -------
    Dict[str, str]
        Mapping from zone code to zone name.
    """
    lookup: Dict[str, str] = {}
    for zone in zone_templates:
        zone_code = str(zone.get('zone_code') or '').strip()
        if not zone_code:
            continue
        zone_heading = str(zone.get('zone_heading') or '').strip()
        zone_name = str(zone.get('zone_name') or '').strip()
        if prefer_heading:
            display_name = zone_heading or zone_name
        else:
            display_name = zone_name or zone_heading
        if display_name:
            lookup[zone_code] = display_name
    return lookup

def resolve_zone_name_by_code(zone_code: Any, zone_name_lookup: Dict[str, str]) -> Optional[str]:
    """
    Resolve zone name by exact code, then by progressively shortened base code.
    """
    if pd.isna(zone_code):
        return None
    code = str(zone_code).strip()
    if not code or code.upper() == 'NULL':
        return None
    if code in zone_name_lookup:
        return zone_name_lookup[code]
    parts = code.split('.')
    for i in range(len(parts) - 1, 0, -1):
        fallback_code = '.'.join(parts[:i])
        if fallback_code in zone_name_lookup:
            return zone_name_lookup[fallback_code]
    return None

def attach_actual_zone_name_column(gdf: gpd.GeoDataFrame, zone_templates: List[dict[str, Any]], actual_zone_code_col: str='PZZ_ACTUAL_CODE_x', output_col: str='PZZ_ACTUAL_NAME_x', prefer_heading: bool=True, overwrite: bool=True) -> gpd.GeoDataFrame:
    """
    Attach actual zone name column by matching actual zone code to raw zone templates.
    """
    result = gdf.copy()
    zone_name_lookup = build_zone_name_lookup(zone_templates=zone_templates, prefer_heading=prefer_heading)
    if output_col not in result.columns:
        result[output_col] = None
    for idx in result.index:
        if not overwrite:
            existing_value = result.at[idx, output_col]
            if not pd.isna(existing_value):
                normalized_existing_value = str(existing_value).strip()
                if normalized_existing_value and normalized_existing_value.upper() != 'NULL':
                    continue
        zone_code = result.at[idx, actual_zone_code_col]
        zone_name = resolve_zone_name_by_code(zone_code, zone_name_lookup)
        result.at[idx, output_col] = zone_name
    return result

def select_and_rename_result_columns(gdf: gpd.GeoDataFrame, cadastral_vri_col: str) -> gpd.GeoDataFrame:
    """
    Keep only selected result columns and rename them to Russian names.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with pipeline results.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with selected and renamed columns.
    """
    column_mapping: Dict[str, str] = {cadastral_vri_col: 'ВРИ_ЕГРН', 'PZZ_ACTUAL_CODE_x': 'Код фактической зоны нахождения кадастра', 'PZZ_ACTUAL_NAME_x': 'Название фактической зоны нахождения кадастра', 'CHECK_SCOPE': 'Область_проверки', 'PZZ_VRI_VERDICT': 'Вердикт_ПЗЗ', 'Статус': 'Статус', 'PZZ_REASON': 'Причина', 'MATCH_METHOD': 'Метод_сопоставления', 'MATCHED_VRI_NAME': 'Подобранный_ВРИ', 'MATCHED_VRI_CODE': 'Код_подобранного_ВРИ', 'ALLOWED_TOP_CANDIDATE_CODES': 'Код_возможного_подобранного_ВРИ', 'PZZ_NOT_ALLOWED_TOP1_CANDIDATE': 'Топ1_возможный_ВРИ', 'PZZ_NOT_ALLOWED_TOP5_CANDIDATES': 'Топ5_возможных_ВРИ'}
    existing_columns: List[str] = [column_name for column_name in column_mapping.keys() if column_name in gdf.columns]
    result_gdf = gdf[existing_columns + ['geometry']].copy()
    result_gdf = result_gdf.rename(columns={column_name: column_mapping[column_name] for column_name in existing_columns})
    return gpd.GeoDataFrame(result_gdf, geometry='geometry', crs=gdf.crs)
