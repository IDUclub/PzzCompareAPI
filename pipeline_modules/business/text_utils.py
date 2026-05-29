from __future__ import annotations
import copy
import json
import os
import re
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from typing import Any, Optional

import pandas as pd
try:
    import pymorphy3
except Exception:  # noqa: BLE001
    pymorphy3 = None
from nltk.stem.snowball import RussianStemmer

try:
    from iduconfig import Config
except Exception:  # noqa: BLE001
    Config = None

config = Config() if Config is not None else None

FAST_DESC_MAX_TOKENS = int(os.getenv("FAST_DESC_MAX_TOKENS", "24"))
def normalize_text(value: Any) -> str:
    """Convert value to a safe stripped string."""
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        return ''
    return text

def is_placeholder_value(value: Any) -> bool:
    """Return True when the value should be treated as empty."""
    return normalize_text(value).lower() in {'', '-', '—', '–', 'nan', 'none', 'null'}

def resolve_first_existing_path(path_candidates: list[str]) -> str:
    """Return the first existing file path from candidates."""
    for path in path_candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f'No existing file found in candidates: {path_candidates}')

VRI_CODE_PATTERN = re.compile('^\\d+(?:\\.\\d+)*$')

TOKEN_PATTERN = re.compile('[0-9a-zа-яё]+', flags=re.IGNORECASE)

_TOKEN_CANONICAL_MAP = {'ижс': 'индивидуальный'}

_STEMMER = RussianStemmer()



def _build_morph():
    """Create pymorph analyzer when available."""
    if pymorphy3 is None:
        return None
    try:
        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


_MORPH = _build_morph()

def is_valid_vri_code(value: Any) -> bool:
    """Return True when a value looks like a classifier VRI code."""
    return bool(VRI_CODE_PATTERN.fullmatch(normalize_text(value)))

def normalize_russian_text(text: Any) -> str:
    """Normalize Russian text for matching."""
    value = normalize_text(text).lower().replace('ё', 'е')
    value = re.sub('["\\\'`«»]', ' ', value)
    value = re.sub('[\\(\\)\\[\\]\\{\\}:;,.!?]', ' ', value)
    value = re.sub('[\\\\/]', ' ', value)
    value = re.sub('[-_]+', ' ', value)
    value = re.sub('\\s+', ' ', value).strip()
    return value

def normalize_match_token(token: str) -> str:
    """Normalize one token with lemmatization when available and stemming fallback."""
    token_norm = normalize_russian_text(token)
    if not token_norm:
        return ''
    token_norm = _TOKEN_CANONICAL_MAP.get(token_norm, token_norm)
    if _MORPH is not None:
        try:
            parsed = _MORPH.parse(token_norm)
            if parsed:
                token_norm = parsed[0].normal_form
        except Exception:
            pass
    if not token_norm:
        return ''
    token_norm = _STEMMER.stem(token_norm)
    return token_norm.strip()

def normalize_match_tokens(text: Any) -> list[str]:
    """Tokenize text and normalize tokens for robust Russian matching."""
    value = normalize_russian_text(text)
    if not value:
        return []
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(value):
        token = normalize_match_token(raw_token)
        if token:
            tokens.append(token)
    return tokens

def canonicalize_vri_name(value: Any) -> str:
    """Canonicalize VRI text for robust matching."""
    text = normalize_russian_text(value)
    replacements = [('^для\\s+', ''), ('^ведение\\s+', 'ведение '), ('\\bижс\\b', 'индивидуальное жилищное строительство'), ('\\bдля индивидуального жилищного строительства\\b', 'индивидуальное жилищное строительство'), ('\\bиндивидуальной жилой застройки\\b', 'индивидуальное жилищное строительство'), ('\\bиндивидуальной жилищной застройки\\b', 'индивидуальное жилищное строительство'), ('\\bдля размещения индивидуального жилого дома\\b', 'размещение индивидуального жилого дома'), ('\\bдля размещения жилого дома\\b', 'размещение жилого дома'), ('\\bразмещения\\b', 'размещение'), ('\\bсадоводства\\b', 'садоводство'), ('\\bведения садоводства\\b', 'садоводство'), ('\\bдля ведения садоводства\\b', 'садоводство')]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text).strip()
    tokens = normalize_match_tokens(text)
    return ' '.join(tokens)

def tokenize_canonical(text: Any) -> list[str]:
    """Split canonicalized text into tokens."""
    return [token for token in canonicalize_vri_name(text).split() if token]

def is_exactish_vri_match(source_vri: Any, candidate_vri_name: Any) -> bool:
    """Return True when two VRI phrases match exactly or almost exactly."""
    left = canonicalize_vri_name(source_vri)
    right = canonicalize_vri_name(candidate_vri_name)
    if not left or not right:
        return False
    if left == right:
        return True
    if left.startswith(right) or right.startswith(left):
        shorter = min(len(left), len(right))
        longer = max(len(left), len(right))
        if shorter >= 10 and shorter / max(longer, 1) >= 0.7:
            return True
    left_tokens = set(tokenize_canonical(left))
    right_tokens = set(tokenize_canonical(right))
    if not left_tokens or not right_tokens:
        return False
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / max(union, 1)
    return intersection >= 2 and jaccard >= 0.55

def build_short_description_match_text(value: Any, max_tokens: int=FAST_DESC_MAX_TOKENS) -> str:
    """Build a compact normalized description text for cheap string matching."""
    tokens = normalize_match_tokens(value)
    if not tokens:
        return ''
    stop_tokens = {'земельн', 'участок', 'территор', 'объект', 'капитальн', 'строительств', 'предназнач', 'цел', 'такж', 'включа', 'содержан', 'дан', 'вид', 'разреш', 'использован', 'код', 'размещен', 'размеща', 'границ', 'населен', 'пункт', 'данн', 'связ', 'осуществлен', 'обеспечен', 'прием', 'граждан', 'вопрос', 'назначен', 'организац', 'обслуживан'}
    filtered: list[str] = []
    for token in tokens:
        if token in stop_tokens:
            continue
        filtered.append(token)
        if len(filtered) >= max_tokens:
            break
    return ' '.join(filtered if filtered else tokens[:max_tokens])

def compute_token_overlap_metrics(left: Any, right: Any) -> dict[str, float]:
    """Compute cheap token overlap metrics between two normalized texts."""
    left_tokens = set(tokenize_canonical(left))
    right_tokens = set(tokenize_canonical(right))
    if not left_tokens or not right_tokens:
        return {'overlap': 0.0, 'coverage': 0.0, 'jaccard': 0.0}
    overlap = float(len(left_tokens & right_tokens))
    coverage = overlap / max(len(left_tokens), 1)
    jaccard = overlap / max(len(left_tokens | right_tokens), 1)
    return {'overlap': overlap, 'coverage': coverage, 'jaccard': jaccard}

def compute_string_similarity(left: Any, right: Any) -> float:
    """Compute cheap normalized string similarity."""
    left_norm = canonicalize_vri_name(left)
    right_norm = canonicalize_vri_name(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(SequenceMatcher(None, left_norm, right_norm).ratio())

def compute_fast_match_score(query_text: Any, candidate_text: Any) -> dict[str, float]:
    """Blend sequence and token overlap metrics into one cheap score."""
    seq_ratio = compute_string_similarity(query_text, candidate_text)
    overlap_metrics = compute_token_overlap_metrics(query_text, candidate_text)
    score = max(seq_ratio, min(1.0, overlap_metrics['coverage'] * 0.97), min(1.0, overlap_metrics['jaccard'] * 1.08))
    return {'score': float(score), 'seq_ratio': float(seq_ratio), **overlap_metrics}

def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON safely and recover JSON object from text if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search('\\{.*\\}', text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))

def collect_unique_codes(values: list[Any]) -> list[str]:
    """Collect unique normalized codes preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = normalize_text(value)
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result

def split_codes(value: Any) -> list[str]:
    """Split joined code strings into a list of codes."""
    text = normalize_text(value)
    if not text:
        return []
    parts = re.split('\\s*\\|\\s*|;\\s*|,\\s*', text)
    return [part for part in [normalize_text(item) for item in parts] if part]

def build_actual_zone_key(vri_text: Any, actual_code: Any) -> str:
    """Build stable key for actual-zone checks only."""
    return ' || '.join([canonicalize_vri_name(vri_text), normalize_text(actual_code)])

def build_fallback_key(vri_text: Any, actual_code: Any, intersect_codes: Any) -> str:
    """Build stable key for fallback/global-search checks."""
    return ' || '.join([canonicalize_vri_name(vri_text), normalize_text(actual_code), normalize_text(intersect_codes)])

def build_comparison_key(vri_text: Any, actual_code: Any, intersect_codes: Any) -> str:
    """Backward-compatible alias for the final merge key."""
    return build_fallback_key(vri_text=vri_text, actual_code=actual_code, intersect_codes=intersect_codes)

CANONICAL_VERDICTS = {'allowed_main', 'allowed_conditional', 'allowed_auxiliary', 'not_allowed', 'unclear', 'no_actual_zone', 'no_zone_metadata'}

def normalize_llm_verdict(verdict: Any, matched_section: Any=None, reason: Any=None, matched_vri_name: Any=None) -> str:
    """Normalize raw LLM/fallback verdicts to canonical status enum."""
    verdict_norm = normalize_text(verdict).lower()
    section_norm = normalize_text(matched_section).lower()
    reason_norm = normalize_text(reason).lower()
    matched_name_norm = normalize_text(matched_vri_name).lower()
    if verdict_norm in CANONICAL_VERDICTS:
        return verdict_norm
    if verdict_norm == 'allowed':
        if section_norm in {'main', 'conditional', 'auxiliary'}:
            return f'allowed_{section_norm}'
        hint_text = f'{reason_norm} {matched_name_norm}'
        if any((token in hint_text for token in ['услов', 'conditional'])):
            return 'allowed_conditional'
        if any((token in hint_text for token in ['вспомог', 'auxiliary'])):
            return 'allowed_auxiliary'
        return 'allowed_main'
    if verdict_norm.startswith('allowed_'):
        if section_norm in {'main', 'conditional', 'auxiliary'}:
            return f'allowed_{section_norm}'
        hint_text = f'{verdict_norm} {reason_norm} {matched_name_norm}'
        if any((token in hint_text for token in ['услов', 'conditional'])):
            return 'allowed_conditional'
        if any((token in hint_text for token in ['вспомог', 'auxiliary'])):
            return 'allowed_auxiliary'
        return 'allowed_main'
    if verdict_norm in {'not_allowed', 'unclear', 'no_actual_zone', 'no_zone_metadata'}:
        return verdict_norm
    return 'unclear'

def status_to_russian_label(verdict: str) -> str:
    """Convert machine verdict to a human-readable Russian label."""
    mapping = {'allowed': 'Разрешен', 'allowed_main': 'Разрешен', 'allowed_conditional': 'Условно разрешен', 'allowed_auxiliary': 'Разрешен как вспомогательный', 'not_allowed': 'Не разрешен', 'unclear': 'Требуется ручная проверка', 'no_actual_zone': 'Нет пересечения с ПЗЗ', 'no_zone_metadata': 'Нет описания зоны в шаблоне'}
    return mapping.get(verdict, 'Требуется ручная проверка')

def build_short_retrieval_text_from_zone_dict(zone: dict[str, Any]) -> str:
    """Build compact searchable text for a zone."""
    zone_code = normalize_text(zone.get('zone_code'))
    zone_name = normalize_text(zone.get('zone_name'))
    summary = normalize_text(zone.get('zone_summary'))
    main_names = ', '.join([normalize_text(item.get('vri_name')) for item in zone.get('main', [])[:8] if normalize_text(item.get('vri_name'))])
    conditional_names = ', '.join([normalize_text(item.get('vri_name')) for item in zone.get('conditional', [])[:5] if normalize_text(item.get('vri_name'))])
    auxiliary_names = ', '.join([normalize_text(item.get('vri_name')) for item in zone.get('auxiliary', [])[:5] if normalize_text(item.get('vri_name'))])
    parts = [f'Код зоны: {zone_code}.', f'Наименование зоны: {zone_name}.', summary, f'Основные ВРИ: {main_names}.', f'Условно разрешенные ВРИ: {conditional_names}.', f'Вспомогательные ВРИ: {auxiliary_names}.']
    return ' '.join([normalize_text(part) for part in parts if normalize_text(part)])

def build_zone_section_text(zone: dict[str, Any], section_name: str) -> str:
    """Render one zone section as text for prompts."""
    lines: list[str] = []
    for item in zone.get(section_name, []) or []:
        vri_code = normalize_text(item.get('vri_code'))
        vri_name = normalize_text(item.get('vri_name'))
        vri_description = normalize_text(item.get('vri_description'))
        lines.append(f'- {vri_code} | {vri_name} | {vri_description}')
    return '\n'.join(lines) if lines else '- нет данных'

def render_zone_retrieval_text(zone: dict[str, Any]) -> str:
    """Rebuild retrieval_text from structured zone fields after sanitation."""
    parts: list[str] = []
    field_specs = [('zone_code', 'Код зоны'), ('base_zone_code', 'Базовый код зоны'), ('article_code', 'Раздел ПЗЗ'), ('zone_group_name', 'Группа зон'), ('zone_name', 'Наименование зоны')]
    for field_name, label in field_specs:
        value = normalize_text(zone.get(field_name))
        if value:
            parts.append(f'{label}: {value}.')
    zone_notes = [normalize_text(item) for item in zone.get('zone_notes', []) if normalize_text(item)]
    if zone_notes:
        parts.append(f"Описание и примечания зоны: {' '.join(zone_notes)}.")
    section_labels = {'main': 'Основные виды разрешенного использования', 'conditional': 'Условно разрешенные виды использования', 'auxiliary': 'Вспомогательные виды использования'}
    for section_name, label in section_labels.items():
        items = zone.get(section_name, []) or []
        section_chunks: list[str] = []
        for item in items:
            vri_code = normalize_text(item.get('vri_code'))
            vri_name = normalize_text(item.get('vri_name'))
            vri_description = normalize_text(item.get('vri_description'))
            if vri_code and vri_name and vri_description:
                section_chunks.append(f'код {vri_code} — {vri_name} — {vri_description}')
            elif vri_code and vri_name:
                section_chunks.append(f'код {vri_code} — {vri_name}')
            elif vri_name:
                section_chunks.append(vri_name)
        if section_chunks:
            parts.append(f"{label}: {' ; '.join(section_chunks)}.")
            continue
        section_note = normalize_text((zone.get('section_notes') or {}).get(section_name))
        if section_note:
            parts.append(f'{label}: {section_note}.')
    return ' '.join((part for part in parts if normalize_text(part)))

def render_zone_summary(zone: dict[str, Any]) -> str:
    """Rebuild zone summary from structured zone fields after sanitation."""
    zone_name = normalize_text(zone.get('zone_name'))
    zone_notes = ' '.join([normalize_text(item) for item in zone.get('zone_notes', []) if normalize_text(item)]).strip()
    main_names = ', '.join([normalize_text(item.get('vri_name')) for item in zone.get('main', [])[:5] if normalize_text(item.get('vri_name'))])
    conditional_names = ', '.join([normalize_text(item.get('vri_name')) for item in zone.get('conditional', [])[:5] if normalize_text(item.get('vri_name'))])
    auxiliary_names = ', '.join([normalize_text(item.get('vri_name')) for item in zone.get('auxiliary', [])[:5] if normalize_text(item.get('vri_name'))])
    parts = [zone_name]
    if zone_notes:
        parts.append(zone_notes)
    if main_names:
        parts.append(f'Основные ВРИ: {main_names}.')
    if conditional_names:
        parts.append(f'Условно разрешенные ВРИ: {conditional_names}.')
    if auxiliary_names:
        parts.append(f'Вспомогательные ВРИ: {auxiliary_names}.')
    return ' '.join((part for part in parts if normalize_text(part))).strip()

def flatten_zone_catalog(raw_catalog: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten structured zone catalog into zone and VRI dataframes with classifier metadata."""
    zone_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for zone in raw_catalog:
        zone_code = normalize_text(zone.get('zone_code'))
        base_zone_code = normalize_text(zone.get('base_zone_code'))
        zone_name = normalize_text(zone.get('zone_name'))
        zone_group_name = normalize_text(zone.get('zone_group_name'))
        zone_summary = normalize_text(zone.get('zone_summary'))
        retrieval_text = normalize_text(zone.get('retrieval_text'))
        retrieval_text_short = build_short_retrieval_text_from_zone_dict(zone)
        main_names: list[str] = []
        conditional_names: list[str] = []
        auxiliary_names: list[str] = []
        main_codes: list[str] = []
        conditional_codes: list[str] = []
        auxiliary_codes: list[str] = []
        main_chunks: list[str] = []
        conditional_chunks: list[str] = []
        auxiliary_chunks: list[str] = []
        for section_name, target_list, target_codes, target_chunks in [('main', main_names, main_codes, main_chunks), ('conditional', conditional_names, conditional_codes, conditional_chunks), ('auxiliary', auxiliary_names, auxiliary_codes, auxiliary_chunks)]:
            for item in zone.get(section_name, []) or []:
                vri_code = normalize_text(item.get('vri_code'))
                vri_name = normalize_text(item.get('vri_name'))
                vri_description = normalize_text(item.get('vri_description'))
                original_vri_name = normalize_text(item.get('original_vri_name'))
                original_vri_description = normalize_text(item.get('original_vri_description'))
                vri_parent_code = normalize_text(item.get('vri_parent_code'))
                vri_top_level_code = normalize_text(item.get('vri_top_level_code'))
                normalized_by_rosreestr_classifier = bool(item.get('normalized_by_rosreestr_classifier'))
                if vri_name:
                    target_list.append(vri_name)
                if vri_code:
                    target_codes.append(vri_code)
                chunk_parts = [part for part in [vri_code, vri_name, vri_description] if part]
                if chunk_parts:
                    target_chunks.append(' | '.join(chunk_parts))
                item_rows.append({'zone_code': zone_code, 'base_zone_code': base_zone_code, 'zone_name': zone_name, 'zone_group_name': zone_group_name, 'zone_summary': zone_summary, 'section_name': section_name, 'catalog_vri_code': vri_code, 'catalog_vri_name': vri_name, 'catalog_vri_description': vri_description, 'catalog_vri_name_norm': canonicalize_vri_name(vri_name), 'catalog_vri_name_plain': normalize_text(item.get('vri_name_plain')) or vri_name, 'catalog_original_vri_name': original_vri_name, 'catalog_original_vri_description': original_vri_description, 'catalog_vri_parent_code': vri_parent_code, 'catalog_vri_top_level_code': vri_top_level_code, 'catalog_normalized_by_rosreestr_classifier': normalized_by_rosreestr_classifier})
        zone_rows.append({'zone_code': zone_code, 'base_zone_code': base_zone_code, 'zone_name': zone_name, 'zone_group_name': zone_group_name, 'zone_summary': zone_summary, 'retrieval_text': retrieval_text, 'retrieval_text_short': retrieval_text_short, 'main_vri_names': ' | '.join(main_names), 'conditional_vri_names': ' | '.join(conditional_names), 'auxiliary_vri_names': ' | '.join(auxiliary_names), 'main_vri_codes': ' | '.join(main_codes), 'conditional_vri_codes': ' | '.join(conditional_codes), 'auxiliary_vri_codes': ' | '.join(auxiliary_codes), 'main_vri_full': ' ; '.join(main_chunks), 'conditional_vri_full': ' ; '.join(conditional_chunks), 'auxiliary_vri_full': ' ; '.join(auxiliary_chunks)})
    if not zone_rows:
        # Empty catalog (e.g. classification-only mode) — return typed empty DataFrames
        # so callers that check .empty or iterate columns don't crash.
        zone_df = pd.DataFrame(columns=['zone_code', 'base_zone_code', 'zone_name', 'zone_group_name', 'zone_summary', 'retrieval_text', 'retrieval_text_short', 'main_vri_names', 'conditional_vri_names', 'auxiliary_vri_names', 'main_vri_codes', 'conditional_vri_codes', 'auxiliary_vri_codes', 'main_vri_full', 'conditional_vri_full', 'auxiliary_vri_full', 'zone_search_text'])
        item_df = pd.DataFrame(columns=['zone_code', 'base_zone_code', 'zone_name', 'zone_group_name', 'zone_summary', 'section_name', 'catalog_vri_code', 'catalog_vri_name', 'catalog_vri_description', 'catalog_vri_name_norm', 'catalog_vri_name_plain', 'catalog_original_vri_name', 'catalog_original_vri_description', 'catalog_vri_parent_code', 'catalog_vri_top_level_code', 'catalog_normalized_by_rosreestr_classifier', 'item_search_text'])
        return (zone_df, item_df)

    zone_df = pd.DataFrame(zone_rows)
    item_df = pd.DataFrame(item_rows)
    zone_df['zone_search_text'] = (zone_df['zone_code'].fillna('') + ' ' + zone_df['zone_name'].fillna('') + ' ' + zone_df['zone_summary'].fillna('') + ' ' + zone_df['main_vri_names'].fillna('') + ' ' + zone_df['conditional_vri_names'].fillna('') + ' ' + zone_df['auxiliary_vri_names'].fillna('')).map(canonicalize_vri_name)
    item_df['item_search_text'] = (item_df['catalog_vri_name'].fillna('') + ' ' + item_df['catalog_vri_description'].fillna('') + ' ' + item_df['catalog_original_vri_name'].fillna('') + ' ' + item_df['catalog_original_vri_description'].fillna('') + ' ' + item_df['zone_name'].fillna('') + ' ' + item_df['zone_summary'].fillna('')).map(canonicalize_vri_name)
    return (zone_df, item_df)

def build_rosreestr_classifier_maps(raw_classifier: Any) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Build code and canonical-name lookup maps from the Rosreestr classifier JSON."""
    by_code: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if isinstance(raw_classifier, dict):
        entries = raw_classifier.get('entries') or list((raw_classifier.get('by_code') or {}).values())
    elif isinstance(raw_classifier, list):
        entries = raw_classifier
    else:
        entries = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        code = normalize_text(raw_entry.get('code'))
        name = normalize_text(raw_entry.get('name'))
        description = normalize_text(raw_entry.get('description'))
        parent_code = normalize_text(raw_entry.get('parent_code'))
        top_level_code = normalize_text(raw_entry.get('top_level_code'))
        name_plain = normalize_text(raw_entry.get('name_plain')) or name
        if not is_valid_vri_code(code) or not name:
            continue
        entry = {'code': code, 'name': name, 'description': description, 'parent_code': parent_code, 'top_level_code': top_level_code, 'name_plain': name_plain}
        by_code[code] = entry
        candidate_keys = {canonicalize_vri_name(name), canonicalize_vri_name(name_plain), canonicalize_vri_name(normalize_russian_text(name))}
        for key in candidate_keys:
            if key:
                by_name[key].append(entry)
    return (by_code, dict(by_name))

def lookup_rosreestr_classifier_entry(*, vri_code: Any, vri_name: Any, classifier_by_code: Optional[dict[str, dict[str, Any]]]=None, classifier_by_name: Optional[dict[str, list[dict[str, Any]]]]=None) -> Optional[dict[str, Any]]:
    """Find the most plausible classifier entry by jointly considering code and name."""
    classifier_by_code = classifier_by_code or {}
    classifier_by_name = classifier_by_name or {}
    code = normalize_text(vri_code)
    name = normalize_text(vri_name)
    code_entry = classifier_by_code.get(code) if is_valid_vri_code(code) else None
    name_entry: Optional[dict[str, Any]] = None
    name_keys = [canonicalize_vri_name(name), canonicalize_vri_name(normalize_russian_text(name))]
    seen_codes: set[str] = set()
    name_candidates: list[dict[str, Any]] = []
    for key in name_keys:
        for candidate in classifier_by_name.get(key, []) or []:
            candidate_code = normalize_text(candidate.get('code'))
            if candidate_code and candidate_code not in seen_codes:
                seen_codes.add(candidate_code)
                name_candidates.append(candidate)
    if code_entry is not None and name_candidates:
        for candidate in name_candidates:
            if normalize_text(candidate.get('code')) == normalize_text(code_entry.get('code')):
                return candidate
    if len(name_candidates) == 1:
        name_entry = name_candidates[0]
    elif len(name_candidates) > 1:
        sorted_candidates = sorted(name_candidates, key=lambda item: (len(normalize_text(item.get('code'))), normalize_text(item.get('code'))))
        name_entry = sorted_candidates[0]
    if name_entry is not None:
        if code_entry is None:
            return name_entry
        code_entry_name = canonicalize_vri_name(code_entry.get('name'))
        name_entry_name = canonicalize_vri_name(name_entry.get('name'))
        input_name = canonicalize_vri_name(name)
        if input_name and name_entry_name == input_name and (code_entry_name != input_name):
            return name_entry
        if normalize_text(name_entry.get('code')) != normalize_text(code_entry.get('code')):
            return name_entry
    return code_entry or name_entry

def build_vri_reference_map(raw_catalog: list[dict[str, Any]], classifier_by_code: Optional[dict[str, dict[str, Any]]]=None, classifier_by_name: Optional[dict[str, list[dict[str, Any]]]]=None) -> dict[str, dict[str, str]]:
    """Build a canonical VRI reference map, preferring the Rosreestr classifier when available."""
    classifier_by_code = classifier_by_code or {}
    classifier_by_name = classifier_by_name or {}
    reference: dict[str, dict[str, str]] = {}
    if classifier_by_code:
        for entry in classifier_by_code.values():
            for key in {canonicalize_vri_name(entry.get('name')), canonicalize_vri_name(entry.get('name_plain'))}:
                if not key:
                    continue
                reference[key] = {'code': normalize_text(entry.get('code')), 'name': normalize_text(entry.get('name')), 'description': normalize_text(entry.get('description')), 'parent_code': normalize_text(entry.get('parent_code')), 'top_level_code': normalize_text(entry.get('top_level_code')), 'name_plain': normalize_text(entry.get('name_plain')) or normalize_text(entry.get('name'))}
    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for zone in raw_catalog:
        for section_name in ('main', 'conditional', 'auxiliary'):
            for item in zone.get(section_name, []) or []:
                vri_name = normalize_text(item.get('vri_name'))
                original_vri_name = normalize_text(item.get('original_vri_name'))
                vri_code = normalize_text(item.get('vri_code'))
                vri_description = normalize_text(item.get('vri_description'))
                classifier_entry = lookup_rosreestr_classifier_entry(vri_code=vri_code, vri_name=vri_name or original_vri_name, classifier_by_code=classifier_by_code, classifier_by_name=classifier_by_name)
                row = {'code': normalize_text((classifier_entry or {}).get('code')) or vri_code, 'name': normalize_text((classifier_entry or {}).get('name')) or vri_name, 'description': normalize_text((classifier_entry or {}).get('description')) or vri_description, 'parent_code': normalize_text((classifier_entry or {}).get('parent_code')) or normalize_text(item.get('vri_parent_code')), 'top_level_code': normalize_text((classifier_entry or {}).get('top_level_code')) or normalize_text(item.get('vri_top_level_code')), 'name_plain': normalize_text((classifier_entry or {}).get('name_plain')) or normalize_text(item.get('vri_name_plain')) or vri_name}
                for alias in [vri_name, original_vri_name, row['name'], row['name_plain']]:
                    alias_key = canonicalize_vri_name(alias)
                    if alias_key and normalize_text(row['code']):
                        by_name[alias_key].append(row)
    for key, rows in by_name.items():
        code_counter = Counter((row['code'] for row in rows if normalize_text(row.get('code'))))
        name_counter = Counter((row['name'] for row in rows if normalize_text(row.get('name'))))
        desc_counter = Counter((row['description'] for row in rows if normalize_text(row.get('description'))))
        parent_counter = Counter((row['parent_code'] for row in rows if normalize_text(row.get('parent_code'))))
        top_counter = Counter((row['top_level_code'] for row in rows if normalize_text(row.get('top_level_code'))))
        plain_counter = Counter((row['name_plain'] for row in rows if normalize_text(row.get('name_plain'))))
        reference[key] = {'code': code_counter.most_common(1)[0][0] if code_counter else '', 'name': name_counter.most_common(1)[0][0] if name_counter else '', 'description': desc_counter.most_common(1)[0][0] if desc_counter else '', 'parent_code': parent_counter.most_common(1)[0][0] if parent_counter else '', 'top_level_code': top_counter.most_common(1)[0][0] if top_counter else '', 'name_plain': plain_counter.most_common(1)[0][0] if plain_counter else ''}
    return reference

def sanitize_zone_catalog(raw_catalog: list[dict[str, Any]], classifier_by_code: Optional[dict[str, dict[str, Any]]]=None, classifier_by_name: Optional[dict[str, list[dict[str, Any]]]]=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Repair malformed VRI items and align them against the Rosreestr classifier when available."""
    sanitized_catalog = copy.deepcopy(raw_catalog)
    reference_map = build_vri_reference_map(raw_catalog, classifier_by_code=classifier_by_code, classifier_by_name=classifier_by_name)
    stats = {'zones_total': len(sanitized_catalog), 'items_total': 0, 'items_repaired': 0, 'code_filled_from_reference': 0, 'name_aligned_to_classifier': 0, 'description_filled_from_reference': 0, 'description_moved_from_code': 0, 'parent_code_filled_from_classifier': 0, 'top_level_code_filled_from_classifier': 0, 'classifier_alignments': 0, 'retrieval_text_rebuilt': 0, 'zone_summary_rebuilt': 0, 'changed_zones': 0}
    extra_string_fields = ['original_vri_name', 'original_vri_description', 'vri_name_plain', 'vri_parent_code', 'vri_top_level_code']
    for zone in sanitized_catalog:
        zone_changed = False
        for section_name in ('main', 'conditional', 'auxiliary'):
            repaired_items: list[dict[str, Any]] = []
            for item in zone.get(section_name, []) or []:
                stats['items_total'] += 1
                repaired_item = dict(item)
                repaired_item['vri_code'] = normalize_text(repaired_item.get('vri_code'))
                repaired_item['vri_name'] = normalize_text(repaired_item.get('vri_name'))
                repaired_item['vri_description'] = normalize_text(repaired_item.get('vri_description'))
                for field_name in extra_string_fields:
                    repaired_item[field_name] = normalize_text(repaired_item.get(field_name))
                repaired_item['normalized_by_rosreestr_classifier'] = bool(repaired_item.get('normalized_by_rosreestr_classifier'))
                original_item = dict(repaired_item)
                if not repaired_item['original_vri_name']:
                    repaired_item['original_vri_name'] = repaired_item['vri_name']
                if not repaired_item['original_vri_description']:
                    repaired_item['original_vri_description'] = repaired_item['vri_description']
                classifier_entry = lookup_rosreestr_classifier_entry(vri_code=repaired_item['vri_code'], vri_name=repaired_item['vri_name'] or repaired_item['original_vri_name'], classifier_by_code=classifier_by_code, classifier_by_name=classifier_by_name)
                reference = reference_map.get(canonicalize_vri_name(repaired_item['vri_name'] or repaired_item['original_vri_name']), {})
                code_is_valid = is_valid_vri_code(repaired_item['vri_code'])
                code_looks_like_description = repaired_item['vri_code'] and (not code_is_valid) and (len(repaired_item['vri_code']) >= 32)
                if code_looks_like_description and (not repaired_item['vri_description'] or repaired_item['vri_description'] == repaired_item['vri_name'] or repaired_item['vri_description'] == repaired_item['vri_code']):
                    repaired_item['vri_description'] = repaired_item['vri_code']
                    stats['description_moved_from_code'] += 1
                if classifier_entry:
                    classifier_code = normalize_text(classifier_entry.get('code'))
                    classifier_name = normalize_text(classifier_entry.get('name'))
                    classifier_description = normalize_text(classifier_entry.get('description'))
                    classifier_parent_code = normalize_text(classifier_entry.get('parent_code'))
                    classifier_top_level_code = normalize_text(classifier_entry.get('top_level_code'))
                    classifier_name_plain = normalize_text(classifier_entry.get('name_plain')) or classifier_name
                    if classifier_code and repaired_item['vri_code'] != classifier_code:
                        repaired_item['vri_code'] = classifier_code
                        stats['code_filled_from_reference'] += 1
                    if classifier_name and canonicalize_vri_name(repaired_item['vri_name']) != canonicalize_vri_name(classifier_name):
                        repaired_item['vri_name'] = classifier_name
                        stats['name_aligned_to_classifier'] += 1
                    if classifier_description and repaired_item['vri_description'] != classifier_description:
                        repaired_item['vri_description'] = classifier_description
                        stats['description_filled_from_reference'] += 1
                    if classifier_parent_code and repaired_item['vri_parent_code'] != classifier_parent_code:
                        repaired_item['vri_parent_code'] = classifier_parent_code
                        stats['parent_code_filled_from_classifier'] += 1
                    if classifier_top_level_code and repaired_item['vri_top_level_code'] != classifier_top_level_code:
                        repaired_item['vri_top_level_code'] = classifier_top_level_code
                        stats['top_level_code_filled_from_classifier'] += 1
                    if classifier_name_plain:
                        repaired_item['vri_name_plain'] = classifier_name_plain
                    repaired_item['normalized_by_rosreestr_classifier'] = True
                    stats['classifier_alignments'] += 1
                if not is_valid_vri_code(repaired_item['vri_code']) and normalize_text(reference.get('code')):
                    repaired_item['vri_code'] = normalize_text(reference.get('code'))
                    stats['code_filled_from_reference'] += 1
                if not repaired_item['vri_name'] and normalize_text(reference.get('name')):
                    repaired_item['vri_name'] = normalize_text(reference.get('name'))
                    stats['name_aligned_to_classifier'] += 1
                if repaired_item['vri_name'] and (not repaired_item['vri_description'] or repaired_item['vri_description'] == repaired_item['vri_name']) and normalize_text(reference.get('description')):
                    repaired_item['vri_description'] = normalize_text(reference.get('description'))
                    stats['description_filled_from_reference'] += 1
                if not repaired_item['vri_parent_code'] and normalize_text(reference.get('parent_code')):
                    repaired_item['vri_parent_code'] = normalize_text(reference.get('parent_code'))
                    stats['parent_code_filled_from_classifier'] += 1
                if not repaired_item['vri_top_level_code'] and normalize_text(reference.get('top_level_code')):
                    repaired_item['vri_top_level_code'] = normalize_text(reference.get('top_level_code'))
                    stats['top_level_code_filled_from_classifier'] += 1
                if not repaired_item['vri_name_plain'] and normalize_text(reference.get('name_plain')):
                    repaired_item['vri_name_plain'] = normalize_text(reference.get('name_plain'))
                if repaired_item != original_item:
                    stats['items_repaired'] += 1
                    zone_changed = True
                repaired_items.append(repaired_item)
            zone[section_name] = repaired_items
        if zone_changed:
            stats['changed_zones'] += 1
        section_name_ru = {'main': 'Основные виды разрешенного использования', 'conditional': 'Условно разрешенные виды использования', 'auxiliary': 'Вспомогательные виды использования'}
        summary_prefix_ru = {'main': 'Основные ВРИ', 'conditional': 'Условно разрешенные ВРИ', 'auxiliary': 'Вспомогательные ВРИ'}
        retrieval_parts: list[str] = []
        zone_summary_parts: list[str] = []
        if normalize_text(zone.get('zone_name')):
            retrieval_parts.append(f"Наименование зоны: {normalize_text(zone.get('zone_name'))}.")
            zone_summary_parts.append(normalize_text(zone.get('zone_name')))
        if normalize_text(zone.get('zone_notes')):
            retrieval_parts.append(f"Примечания зоны: {normalize_text(zone.get('zone_notes'))}.")
        for section_name in ('main', 'conditional', 'auxiliary'):
            items = zone.get(section_name, []) or []
            if not items:
                continue
            retrieval_chunks: list[str] = []
            summary_names: list[str] = []
            for item in items:
                item_parts = [part for part in [f"код {normalize_text(item.get('vri_code'))}" if normalize_text(item.get('vri_code')) else '', normalize_text(item.get('vri_name')), normalize_text(item.get('vri_description'))] if part]
                if item_parts:
                    retrieval_chunks.append(' — '.join(item_parts))
                if normalize_text(item.get('vri_name')):
                    summary_names.append(normalize_text(item.get('vri_name')))
            if retrieval_chunks:
                retrieval_parts.append(f'{section_name_ru[section_name]}: ' + ' ; '.join(retrieval_chunks) + '.')
            if summary_names:
                zone_summary_parts.append(f'{summary_prefix_ru[section_name]}: ' + ', '.join(summary_names) + '.')
        rebuilt_retrieval_text = ' '.join((part for part in retrieval_parts if part)).strip()
        rebuilt_zone_summary = ' '.join((part for part in zone_summary_parts if part)).strip()
        if rebuilt_retrieval_text:
            zone['retrieval_text'] = rebuilt_retrieval_text
            stats['retrieval_text_rebuilt'] += 1
        if rebuilt_zone_summary:
            zone['zone_summary'] = rebuilt_zone_summary
            stats['zone_summary_rebuilt'] += 1
    return (sanitized_catalog, stats)
