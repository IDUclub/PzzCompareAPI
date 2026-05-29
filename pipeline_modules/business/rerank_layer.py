from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from .clients import not_allowed_rerank_ollama
from .common import *
from .matching_layer import build_catalog_embed_text
from .profiled_fast_match_layer import (
    PROFILE_TO_VRI_PATTERNS,
    build_zone_classifier_snapshot,
    classify_object_profile,
    expand_vri_codes_with_classifier_children,
    render_profile_hint,
    should_use_deeper_llm_reasoning,
)


def truncate_text(text: Any, max_chars: int) -> str:
    """
    Truncate text to a maximum number of characters.
    """
    value = normalize_text(text)
    if not value or len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return value[:max_chars - 1].rstrip() + '…'


ENABLE_EMBED_CANDIDATE_SHORTLIST = bool(ENABLE_EMBED_FAST_MATCH)


def build_classifier_embedding_items(classifier_by_code: Optional[dict[str, dict[str, Any]]]) -> pd.DataFrame:
    """Build one compact dataframe for Rosreestr classifier embedding search."""
    rows: list[dict[str, Any]] = []
    for code, entry in (classifier_by_code or {}).items():
        code_norm = normalize_text(code)
        name = normalize_text((entry or {}).get('name'))
        description = normalize_text((entry or {}).get('description'))
        parent_code = normalize_text((entry or {}).get('parent_code'))
        top_level_code = normalize_text((entry or {}).get('top_level_code'))
        name_plain = normalize_text((entry or {}).get('name_plain')) or name
        if not is_valid_vri_code(code_norm) or not name:
            continue
        if len(code_norm.split('.')) == 1 and name.isdigit():
            continue
        rows.append({'classifier_code': code_norm, 'classifier_name': name, 'classifier_name_plain': name_plain,
                     'classifier_description': description, 'classifier_parent_code': parent_code,
                     'classifier_top_level_code': top_level_code})
    df = pd.DataFrame(rows).drop_duplicates(subset=['classifier_code']).reset_index(drop=True)
    if df.empty:
        return df
    df['classifier_embed_text'] = (
                df['classifier_name'].fillna('') + ' | ' + df['classifier_description'].fillna('')).map(
        canonicalize_vri_name)
    df['classifier_match_text'] = (
                df['classifier_name'].fillna('') + ' | ' + df['classifier_description'].fillna('') + ' | ' + df[
            'classifier_name_plain'].fillna('')).map(canonicalize_vri_name)
    return df


def build_zone_section_code_cache(
    zone_items_lookup_map: dict | None = None,
    classifier_children_map: dict | None = None,
) -> dict[str, dict[str, set[str]]]:
    """Expand per-zone section codes with classifier children for robust in-zone presence checks."""
    cache: dict[str, dict[str, set[str]]] = {}
    children_map = classifier_children_map or {}
    for zone_code, zone_items in (zone_items_lookup_map or {}).items():
        section_cache: dict[str, set[str]] = {'main': set(), 'conditional': set(), 'auxiliary': set()}
        if zone_items is None or zone_items.empty:
            cache[zone_code] = section_cache
            continue
        for section_name in ('main', 'conditional', 'auxiliary'):
            raw_codes = {normalize_text(code) for code in zone_items.loc[
                zone_items['section_name'].map(normalize_text) == section_name, 'catalog_vri_code'].tolist() if
                         normalize_text(code)}
            section_cache[section_name] = expand_vri_codes_with_classifier_children(raw_codes,
                                                                                    classifier_children_map=children_map)
        cache[zone_code] = section_cache
    return cache


NOT_ALLOWED_RECALL_CACHE_TOP_N = max(int(NOT_ALLOWED_CANDIDATES_TOP_N), int(NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N), 20)

NOT_ALLOWED_QUERY_VECTOR_CACHE: dict[str, np.ndarray] = {}

NOT_ALLOWED_RECALL_CANDIDATES_CACHE: dict[str, list[dict[str, Any]]] = {}

NOT_ALLOWED_FAST_RERANK_CACHE: dict[str, list[dict[str, Any]]] = {}

NOT_ALLOWED_LLM_RERANK_CACHE: dict[str, list[dict[str, Any]]] = {}


def _cache_maps(context: Any=None) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if context is None:
        return (
            NOT_ALLOWED_QUERY_VECTOR_CACHE,
            NOT_ALLOWED_RECALL_CANDIDATES_CACHE,
            NOT_ALLOWED_FAST_RERANK_CACHE,
            NOT_ALLOWED_LLM_RERANK_CACHE,
        )
    return (
        context.not_allowed_query_vector_cache,
        context.not_allowed_recall_candidates_cache,
        context.not_allowed_fast_rerank_cache,
        context.not_allowed_llm_rerank_cache,
    )


def get_not_allowed_query_key(vri_text: Any) -> str:
    """
    Build stable cache key for not_allowed candidate search.
    """
    return canonicalize_vri_name(normalize_text(vri_text))


def build_not_allowed_embed_query_text(vri_text: Any) -> str:
    """
    Expand cadastral wording into a more classifier-friendly semantic query.
    """
    raw_text = normalize_text(vri_text)
    canon = canonicalize_vri_name(raw_text)
    if not canon:
        return ''
    hints: list[str] = []
    if re.search('административн\\s+здан|административн', canon):
        hints.extend(['общественное управление', 'деловое управление', 'административное здание', 'офис'])
    if re.search('воинск|военн|казарм|штаб|гарнизон|полигон|оборон|част[ьи]\\s*№', canon):
        hints.extend(['обеспечение обороны и безопасности', 'военные объекты', 'воинская часть'])
    # Многоквартирные дома — самый частый источник ошибок (попадают в 3.2.1 Дома соц. обслуживания).
    # Опечатки: «могоквартирн», «многквартирн» (часты в реальных кадастровых данных).
    # NB: «малоэтажная многоквартирная» НЕ добавляется в хинты — иначе embedding отдаёт
    # 2.1.1 предпочтение из-за лексического совпадения слова «многоквартирная»; для МКД
    # без явного «малоэтажный» мы хотим 2.5/2.6/2.0.
    if re.search('многоквартирн|многквартирн|могоквартирн|мкд\\b|многоэтажн|среднеэтажн', canon):
        hints.extend([
            'среднеэтажная жилая застройка',
            'многоэтажная жилая застройка',
            'размещение многоквартирного жилого дома',
            'жилая застройка',
        ])
    # Только при явном «малоэтажная многоквартирная» — добавляем 2.1.1
    if re.search('малоэтажн\\s+многоквартирн', canon):
        hints.append('малоэтажная многоквартирная жилая застройка')
    # ИЖС / индивидуальный дом — приоритет перед общим «жилой дом»
    if re.search('индивидуальн\\s+жил|ижс|индивидуальн\\s+застройк|одна\\s+семь|однокварт', canon):
        hints.append('для индивидуального жилищного строительства')
    # Жилой дом без уточнений — НЕ должен скатываться в блокированную (2.3)
    if re.search('жил[а-я\\s-]*дом|под\\s+жил', canon) and not re.search('многоквартирн|многоэтажн|среднеэтажн|блокирован', canon):
        hints.extend(['для индивидуального жилищного строительства', 'жилая застройка'])
    # Электрические объекты
    if re.search('подстанц|трансформатор|\\bтп\\s*№|\\bтп\\s*\\d|\\bтп\\b', canon):
        hints.extend([
            'трансформаторная подстанция', 'линии электропередач',
            'предоставление коммунальных услуг', 'коммунальное обслуживание',
            'электричество электроснабжение',
        ])
    # Тепловые и инженерные сети
    if re.search('теплотрасс|теплосет|теплоснабж|теплопровод|тепловая\\s+сеть|тепло[\\s-]?трасс', canon):
        hints.extend([
            'предоставление коммунальных услуг', 'коммунальное обслуживание',
            'теплоснабжение тепловые сети', 'инженерная инфраструктура',
        ])
    # Водоснабжение, канализация, газ
    if re.search('водопровод|водоснабж|канализац|водоотвед|газопровод|газоснабж|очистн\\s+сооруж|насосн\\s+станц|котельн', canon):
        hints.extend([
            'предоставление коммунальных услуг', 'коммунальное обслуживание',
            'инженерная коммунальная инфраструктура',
        ])
    # Улицы, проезды, переулки, дороги — частая ошибка → ЖД/заправка/промышленность
    if re.search('улиц|переулок|проезд|тротуар|пешеходн|набережн|бульвар|велодорож|автомобильн\\s+дорог|капитальн\\s+ремонт\\s+пер|капитальн\\s+ремонт\\s+ул', canon):
        hints.extend([
            'улично-дорожная сеть', 'размещение автомобильных дорог',
            'земельные участки территории общего пользования',
            'тротуары пешеходные переходы',
        ])
    # Внутримикрорайонные проезды
    if re.search('внутримикрорайон|внутриквартал|внутридворов', canon):
        hints.extend(['улично-дорожная сеть', 'земельные участки общего пользования', 'размещение автомобильных дорог'])
    # Временные гаражи / гаражи без признака «служебный» — не должны идти в 4.9
    if re.search('гараж', canon) and not re.search('служебн|ведомствен|корпоративн|такси', canon):
        hints.extend([
            'размещение гаражей для собственных нужд',
            'хранение автотранспорта', 'гараж индивидуальный',
        ])
    # Благоустройство — отдельная категория
    if re.search('благоустройств', canon):
        hints.extend(['благоустройство территории'])
    merged = ' | '.join([raw_text] + hints)
    return build_catalog_embed_text(merged)


def _simple_match_tokens(text: Any) -> list[str]:
    """
    Split text into normalized lexical tokens for lightweight rerank.
    """
    canon = canonicalize_vri_name(text)
    if not canon:
        return []
    return [token for token in re.findall('[a-zA-Zа-яА-ЯёЁ0-9]+', canon.lower()) if len(token) >= 3]


def _token_overlap_ratio(query_text: Any, candidate_text: Any) -> float:
    """
    Compute overlap ratio between query and candidate tokens.
    """
    query_tokens = set(_simple_match_tokens(query_text))
    candidate_tokens = set(_simple_match_tokens(candidate_text))
    if not query_tokens or not candidate_tokens:
        return 0.0
    return float(len(query_tokens & candidate_tokens)) / float(len(query_tokens))


def _specificity_bonus_from_code(code: Any) -> float:
    """
    Small bonus for more specific child VRI codes.
    """
    code_norm = normalize_text(code)
    if not code_norm:
        return 0.0
    return min(code_norm.count('.'), 3) * 0.03


def fast_rerank_not_allowed_candidates(vri_text: Any, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Fast deterministic rerank for global not_allowed candidates before LLM.
    """
    if not candidates:
        return []
    query_text = normalize_text(vri_text)
    profile = classify_object_profile(query_text)
    query_canon = canonicalize_vri_name(query_text)
    reranked: list[dict[str, Any]] = []
    for item in candidates:
        item_copy = dict(item)
        name = normalize_text(item_copy.get('name'))
        description = normalize_text(item_copy.get('description'))
        combined = ' | '.join([part for part in [name, description] if part])
        embed_score = float(item_copy.get('score') or 0.0)
        name_overlap = _token_overlap_ratio(query_text, name)
        text_overlap = _token_overlap_ratio(query_text, combined)
        profile_rank = int(item_copy.get('profile_rank') or 0)
        profile_score = min(profile_rank, 100) / 100.0
        exactish_bonus = 0.0
        name_canon = canonicalize_vri_name(name)
        if query_canon and name_canon:
            if query_canon == name_canon:
                exactish_bonus += 0.5
            elif query_canon in name_canon or name_canon in query_canon:
                exactish_bonus += 0.22
        rerank_score = embed_score * 1.0 + name_overlap * 0.9 + text_overlap * 0.55 + profile_score * 0.45 + exactish_bonus + _specificity_bonus_from_code(
            item_copy.get('code'))
        item_copy['rerank_score'] = float(rerank_score)
        reranked.append(item_copy)
    reranked.sort(key=lambda x: (
    float(x.get('rerank_score') or 0.0), float(x.get('score') or 0.0), int(x.get('profile_rank') or 0),
    normalize_text(x.get('code'))), reverse=True)
    return reranked


def should_run_not_allowed_llm_rerank(vri_text: Any, candidates: list[dict[str, Any]]) -> bool:
    """
    Run LLM rerank only for uncertain cases.
    """
    if not ENABLE_LLM or not NOT_ALLOWED_LLM_RERANK_ENABLED:
        return False
    if not candidates:
        return False
    top_candidates = candidates[:max(2, NOT_ALLOWED_CANDIDATES_TOP_N)]
    top1 = float(top_candidates[0].get('rerank_score') or top_candidates[0].get('score') or 0.0)
    top2 = float(top_candidates[1].get('rerank_score') or top_candidates[1].get('score') or 0.0) if len(
        top_candidates) > 1 else 0.0
    gap = top1 - top2
    profile = classify_object_profile(vri_text)
    confidence = normalize_text(profile.get('confidence'))
    profile_rank = int(top_candidates[0].get('profile_rank') or 0)
    query_key = get_not_allowed_query_key(vri_text)
    is_generic = len(query_key.split()) <= 4
    if gap >= 0.16 and profile_rank >= 85 and (confidence in {'strong', 'medium'}) and (not is_generic):
        return False
    return True


# Codes look like 2.1, 2.1.1, 4.9.1.1, 12.0.1 — 2 to 4 dotted numeric segments.
# Word boundaries prevent matching cadastral numbers (65:04:0000040:1661 etc.).
EXPLICIT_VRI_CODE_RE = re.compile(r'(?<![\d:])(\d{1,2}(?:\.\d{1,2}){1,3})(?![\d:])')


def extract_explicit_vri_code(vri_text: Any, context: Any = None) -> Optional[dict[str, Any]]:
    """If the cadastral text contains an explicit valid Rosreestr VRI code, return it as a candidate.

    Many cadastral entries contain the code directly (e.g. "2.7.2 Размещение гаражей").
    A regex pre-pass short-circuits embedding/LLM rerank for these cases.
    """
    if context is None:
        return None
    classifier_by_code = getattr(context, 'rosreestr_classifier_by_code', None) or {}
    if not classifier_by_code:
        return None

    text = normalize_text(vri_text)
    if not text:
        return None

    matches = EXPLICIT_VRI_CODE_RE.findall(text)
    if not matches:
        return None

    # Prefer the most specific (longest) code; for ties, fall back to first occurrence.
    # Python's sort is stable — sorting by -dot_count alone preserves input order for ties,
    # which is the same as the original-position tiebreak we wanted.
    unique_in_order: list[str] = []
    seen: set[str] = set()
    for code in matches:
        if code not in seen:
            seen.add(code)
            unique_in_order.append(code)
    unique_in_order.sort(key=lambda c: -c.count('.'))

    for code in unique_in_order:
        entry = classifier_by_code.get(code)
        if not entry:
            continue
        return {
            'score': 1.0,
            'section_name': 'explicit_code',
            'code': code,
            'name': normalize_text(entry.get('name')),
            'description': normalize_text(entry.get('description')),
            'name_plain': normalize_text(entry.get('name_plain') or entry.get('name')),
            'parent_code': normalize_text(entry.get('parent_code')),
            'top_level_code': normalize_text(entry.get('top_level_code')),
            'profile_rank': 100,
            'query_key': get_not_allowed_query_key(text),
            'explicit_code': True,
        }
    return None


def build_not_allowed_same_zone_candidates(vri_text: Any, actual_zone_code: Any,
                                           query_vector: Optional[np.ndarray] = None, top_n: Optional[int] = None,
                                           min_similarity: Optional[float] = None, context: Any=None) -> list[dict[str, Any]]:
    """
    Build cached global Rosreestr classifier candidates for not_allowed rows.
    """
    if not ENABLE_EMBED_CANDIDATE_SHORTLIST:
        return []

    # Pre-pass: if the cadastral text explicitly contains a valid Rosreestr code,
    # treat it as the single high-confidence candidate. This bypasses embedding
    # recall and LLM rerank entirely for cases like "Для размещения объекта по
    # коду 2.7.2".
    explicit_candidate = extract_explicit_vri_code(vri_text, context=context)
    if explicit_candidate is not None:
        query_vector_cache, recall_candidates_cache, _, _ = _cache_maps(context)
        query_key = explicit_candidate['query_key']
        recall_candidates_cache[query_key] = [dict(explicit_candidate)]
        return [dict(explicit_candidate)]

    classifier_items_df = context.classifier_embed_items_df if context is not None else None
    classifier_vectors = context.classifier_embed_vectors if context is not None else None
    local_vectorizer = context.vectorizer if context is not None else None
    if classifier_items_df is None or classifier_items_df.empty:
        return []
    if classifier_vectors is None or classifier_vectors.size == 0:
        return []
    query_text = normalize_text(vri_text)
    if not query_text:
        return []
    query_vector_cache, recall_candidates_cache, _, _ = _cache_maps(context)
    query_key = get_not_allowed_query_key(query_text)
    top_n = int(top_n or NOT_ALLOWED_CANDIDATES_TOP_N)
    min_similarity = float(NOT_ALLOWED_CANDIDATES_MIN_SIMILARITY if min_similarity is None else min_similarity)
    cached_candidates = recall_candidates_cache.get(query_key)
    if cached_candidates is not None and len(cached_candidates) >= top_n:
        return [dict(item) for item in cached_candidates[:top_n]]
    object_profile = classify_object_profile(query_text)
    if query_vector is None:
        query_vector = query_vector_cache.get(query_key)
    if query_vector is None:
        query_matrix = local_vectorizer.embed_many(texts=[build_not_allowed_embed_query_text(query_text)], batch_size=1)
        if query_matrix is None or query_matrix.size == 0:
            return []
        query_vector = query_matrix[0]
    query_vector_cache[query_key] = query_vector
    scores = (classifier_vectors @ query_vector).astype(float)
    if scores.size == 0:
        return []
    recall_top_n = max(top_n, NOT_ALLOWED_RECALL_CACHE_TOP_N)
    top_pool = min(max(recall_top_n * 8, recall_top_n), len(scores))
    candidate_ids = np.argsort(-scores)[:top_pool]
    candidates: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for global_idx in candidate_ids:
        score = float(scores[int(global_idx)])
        if score < min_similarity:
            continue
        row = classifier_items_df.iloc[int(global_idx)]
        code = normalize_text(row.get('classifier_code'))
        name = normalize_text(row.get('classifier_name'))
        if not code and (not name):
            continue
        if code in seen_codes:
            continue
        entry = row.to_dict()
        seen_codes.add(code)
        candidates.append({'score': score, 'section_name': 'classifier_global', 'code': code, 'name': name,
                           'description': normalize_text(entry.get('classifier_description')),
                           'name_plain': normalize_text(entry.get('classifier_name_plain')),
                           'parent_code': normalize_text(entry.get('classifier_parent_code')),
                           'top_level_code': normalize_text(entry.get('classifier_top_level_code')),
                           'profile_rank': int(rank_classifier_entry_for_profile(entry, object_profile)),
                           'query_key': query_key})
        if len(candidates) >= recall_top_n:
            break
    recall_candidates_cache[query_key] = [dict(item) for item in candidates]
    return [dict(item) for item in candidates[:top_n]]


def serialize_not_allowed_same_zone_candidates(candidates: list[dict[str, Any]]) -> Any:
    """Serialize global classifier embedding candidates into one compact attribute."""
    if not candidates:
        return pd.NA
    parts: list[str] = []
    for item in candidates:
        code = normalize_text(item.get('code'))
        name = normalize_text(item.get('name'))
        if code and name:
            parts.append(f'{code} {name}')
        elif name:
            parts.append(name)
        elif code:
            parts.append(code)
    if not parts:
        return pd.NA
    return ', '.join(parts)


NOT_ALLOWED_RERANK_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'ranked_codes': {
            'type': 'array',
            'items': {'type': 'string'},
            'minItems': NOT_ALLOWED_CANDIDATES_TOP_N,
            'maxItems': NOT_ALLOWED_CANDIDATES_TOP_N,
        }
    },
    'required': ['ranked_codes'],
}

NOT_ALLOWED_RERANK_SYSTEM_PROMPT = """Ты ранжируешь кандидатов ВРИ Росреестра для кадастровой записи.

Верни только компактный JSON-объект с полем ranked_codes.
Никаких пояснений, рассуждений, повторов списка, комментариев до или после JSON.
Нужно вернуть только 5 лучших кодов из переданного shortlist, а не полный порядок всех кандидатов.
Выбирай только из переданного списка кандидатов и не выдумывай новые коды.
Учитывай название и описание кандидатов. Предпочитай прямые функциональные совпадения.

Смысловые правила (важно):
1. Если в кадастровом ВРИ есть слова 'улица', 'переулок', 'проезд', 'тротуар', 'набережная', 'бульвар',
   'капитальный ремонт пер./ул.' - это улично-дорожная сеть.
   Топ должны занимать 12.0.1, 7.2.1, 12.0, а НЕ железнодорожные пути (7.1.1),
   заправки (4.9.1.x), строительная промышленность (6.6) или дорожный отдых (4.9.1.2).
   Слово 'строительная' в названии улицы - это название, а НЕ функция.
2. Многоквартирный дом без уточнения этажности - это среднеэтажная (2.5) или
   многоэтажная (2.6) жилая застройка, а НЕ малоэтажная многоквартирная (2.1.1) и
   тем более НЕ дома социального обслуживания (3.2.1).
   2.1.1 «Малоэтажная многоквартирная» выбирай ТОЛЬКО если в тексте явно есть слово
   «малоэтажный» / «малоэтажная». Само слово «многоквартирный» в названии 2.1.1
   не делает её предпочтительной для общего случая МКД.
   Также НЕ выбирай 4.6 (общепит), 6.x (производство), 3.7.1 (религиозные обряды).
3. Жилой дом без уточнений типа застройки (без слов «многоквартирный», «многоэтажный»,
   «среднеэтажный», «блокированный», «малоэтажный») - это 2.1 (ИЖС) или 2.0
   (Жилая застройка).
   2.3 «Блокированную» выбирай ТОЛЬКО при явных признаках блокированного дома
   (слова «блокированный», «секционный», «таунхаус»).
   2.1.1 «Малоэтажную многоквартирную» в этом случае НЕ выбирать.
4. Теплотрасса, теплосеть, ТП (трансформаторная подстанция), котельная, водопровод, газопровод,
   электросеть, ЛЭП - это коммунальное обслуживание (3.1, 3.1.1, 3.1.2), а НЕ железнодорожные
   пути (7.1.1) и не транспорт.
5. Гараж без признака 'служебный' / 'ведомственный' / 'корпоративный' / 'такси' -
   это 2.7.2 (гаражи для собственных нужд) или 2.7.1 (хранение автотранспорта),
   а НЕ 4.9 (служебные гаражи).
6. Если кадастровый ВРИ - это 'благоустройство территории' - основным кандидатом должен быть
   12.0.2, даже если рядом упомянут магазин или другой объект.
7. Не делай широких ассоциаций по одному совпавшему слову ('строительная' != строительная
   промышленность, 'дом' != дома социального обслуживания).
8. Если есть близкий прямой кандидат, ставь смежные сервисные объекты ниже."""
NOT_ALLOWED_LLM_RERANK_MAX_ATTEMPTS = 3
NOT_ALLOWED_LLM_RERANK_RETRY_DELAY_SEC = 0.35
RETRYABLE_VLLM_ERROR_MARKERS = (
    'empty assistant content',
    'content',
    'finish_reason',
    'length',
    'reasoning',
    'json',
)


def is_retryable_llm_rerank_error(exc: Exception) -> bool:
    """Return True when the LLM error looks transient or truncation-related."""
    text = normalize_text(str(exc)).lower()
    if not text:
        return False
    return any(marker in text for marker in RETRYABLE_VLLM_ERROR_MARKERS)


def build_not_allowed_rerank_prompt(vri_text: Any, candidates: list[dict[str, Any]]) -> str:
    """Build compact LLM prompt for reranking global classifier candidates."""
    query_text = normalize_text(vri_text)
    profile = classify_object_profile(query_text)
    profile_hint = render_profile_hint(profile)

    lines = [
        f'Кадастровый ВРИ: {query_text}\n',
        f'Профиль объекта: {profile_hint}\n',
        '\n',
        f'Выбери ровно {NOT_ALLOWED_CANDIDATES_TOP_N} лучших кодов из shortlist ниже.\n',
        'Верни только JSON вида {"ranked_codes":["2.1","2.3","2.1.1","4.4","3.1"]}.\n',
        'Не добавляй никаких пояснений и не возвращай полный порядок всех кандидатов.\n',
        'Если сильных прямых совпадений меньше пяти, добери оставшиеся места ближайшими по смыслу кандидатами из shortlist.\n',
        'Нельзя возвращать меньше пяти кодов.\n',
        '\n',
        'Shortlist кандидатов:\n',
    ]

    for idx, item in enumerate(candidates, start=1):
        code = normalize_text(item.get('code'))
        name = normalize_text(item.get('name'))
        description = truncate_text(
            item.get('description'),
            max_chars=min(int(NOT_ALLOWED_LLM_RERANK_DESC_MAX_CHARS), 180),
        )
        line = f'{idx}. {code} | {name}'
        if description:
            line += f' | {description}'
        lines.append(line + '\n')

    return ''.join(lines)


def run_not_allowed_rerank_with_llm(vri_text: Any, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select top codes with LLM and keep only returned known codes."""
    if not ENABLE_LLM or not NOT_ALLOWED_LLM_RERANK_ENABLED:
        return candidates
    if not candidates:
        return candidates

    last_exc: Exception | None = None

    for attempt in range(1, NOT_ALLOWED_LLM_RERANK_MAX_ATTEMPTS + 1):
        narrowed_candidates = (
            candidates[:NOT_ALLOWED_CANDIDATES_TOP_N]
            if attempt >= 2
            else candidates[:NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N]
        )

        prompt = build_not_allowed_rerank_prompt(
            vri_text=vri_text,
            candidates=narrowed_candidates,
        )

        # Rerank output is a tiny JSON (~50 tokens). Thinking wastes tokens and
        # causes VLLMTruncatedReasoningError when max_tokens is large.
        # Always disable thinking; no reasoning needed for a simple ranking task.
        think_override = False

        try:
            response = not_allowed_rerank_ollama.complete_json(
                user_prompt=prompt,
                system_prompt=NOT_ALLOWED_RERANK_SYSTEM_PROMPT,
                schema=NOT_ALLOWED_RERANK_SCHEMA,
                model=LLM_MODEL,
                think_override=think_override,
            )

            ranked_codes_raw = response.get('ranked_codes') or []
            ranked_codes = [
                               normalize_text(code)
                               for code in ranked_codes_raw
                               if normalize_text(code)
                           ][:NOT_ALLOWED_CANDIDATES_TOP_N]

            if len(ranked_codes) != NOT_ALLOWED_CANDIDATES_TOP_N:
                raise ValueError(
                    f"LLM returned unexpected ranked_codes length: {len(ranked_codes)}"
                )

            by_code: dict[str, dict[str, Any]] = {}
            for item in narrowed_candidates:
                code = normalize_text(item.get('code'))
                if code and code not in by_code:
                    by_code[code] = item

            reranked: list[dict[str, Any]] = []
            used_codes: set[str] = set()

            for code in ranked_codes:
                if code in by_code and code not in used_codes:
                    reranked.append(by_code[code])
                    used_codes.add(code)

            for item in narrowed_candidates:
                code = normalize_text(item.get('code'))
                if code and code not in used_codes:
                    reranked.append(item)
                    used_codes.add(code)

            return reranked[:NOT_ALLOWED_CANDIDATES_TOP_N]

        except Exception as exc:
            last_exc = exc
            if not is_retryable_llm_rerank_error(exc):
                raise

            logger.warning(
                "Retryable not-allowed LLM rerank failure for '%s' on attempt %s/%s: %s",
                normalize_text(vri_text),
                attempt,
                NOT_ALLOWED_LLM_RERANK_MAX_ATTEMPTS,
                exc,
            )

            if attempt < NOT_ALLOWED_LLM_RERANK_MAX_ATTEMPTS:
                time.sleep(NOT_ALLOWED_LLM_RERANK_RETRY_DELAY_SEC * attempt)
                continue

    logger.warning(
        "Falling back to deterministic not-allowed rerank for '%s' after retries: %s",
        normalize_text(vri_text),
        last_exc,
    )
    return candidates[:NOT_ALLOWED_CANDIDATES_TOP_N]






def get_zone_status_for_classifier_code(zone_code: Any, classifier_code: Any, context: Any=None) -> tuple[bool, Optional[str]]:
    """Return whether a classifier code is present in the zone after parent-child expansion."""
    zone_code_norm = normalize_text(zone_code)
    code_norm = normalize_text(classifier_code)
    if not zone_code_norm or not code_norm:
        return (False, None)
    zone_section_cache = context.zone_section_code_cache if context is not None else {}
    section_cache = zone_section_cache.get(zone_code_norm) or {}
    for section_name in ('main', 'conditional', 'auxiliary'):
        if code_norm in (section_cache.get(section_name) or set()):
            return (True, section_name)
    return (False, None)


def rank_classifier_entry_for_profile(entry: dict[str, Any], profile: dict[str, Any]) -> int:
    """Score one Rosreestr classifier entry against the profiled object family."""
    family = normalize_text(profile.get('family'))
    candidate_keys = profile.get('candidate_keys') or ([] if not family else [family])
    if not candidate_keys or family == 'unknown':
        return 0
    combined = canonicalize_vri_name(' | '.join(
        [normalize_text(entry.get('classifier_name')), normalize_text(entry.get('classifier_description')),
         normalize_text(entry.get('classifier_name_plain'))]))
    if not combined:
        return 0
    best_rank = 0
    for key in candidate_keys:
        for pattern, rank in PROFILE_TO_VRI_PATTERNS.get(key, []):
            if re.search(pattern, combined):
                best_rank = max(best_rank, int(rank))
    return best_rank


def build_embed_classifier_candidates(vri_text: Any, actual_zone_code: Any, query_vector: Optional[np.ndarray] = None,
                                      top_n: Optional[int] = None, min_similarity: Optional[float] = None,
                                      context: Any = None) -> list[dict[str, Any]]:
    """Build a Rosreestr-classifier shortlist for LLM reranking, not for direct verdicts."""
    if not ENABLE_EMBED_CANDIDATE_SHORTLIST:
        return []
    classifier_embed_vectors = context.classifier_embed_vectors if context is not None else None
    if classifier_embed_vectors is None or classifier_embed_vectors.size == 0:
        return []
    query_text = normalize_text(vri_text)
    if not query_text:
        return []
    zone_code = normalize_text(actual_zone_code)
    object_profile = classify_object_profile(query_text)
    top_n = int(top_n or TOP_N_EMBED_CANDIDATES)
    min_similarity = float(MIN_EMBED_CANDIDATE_SIMILARITY if min_similarity is None else min_similarity)
    try:
        if query_vector is None:
            local_vectorizer = context.vectorizer if context is not None else None
            if local_vectorizer is None:
                return []
            query_matrix = local_vectorizer.embed_many(texts=[build_catalog_embed_text(query_text)], batch_size=1)
            if query_matrix is None or query_matrix.size == 0:
                return []
            query_vector = query_matrix[0]
        scores = (classifier_embed_vectors @ query_vector).astype(float)
        if scores.size == 0:
            return []
        top_pool = min(max(top_n * 4, top_n), len(scores))
        classifier_embed_items = context.classifier_embed_items_df if context is not None else None
        if classifier_embed_items is None:
            return []
        candidate_ids = np.argsort(-scores)[:top_pool]
        candidates: list[dict[str, Any]] = []
        for idx in candidate_ids:
            score = float(scores[int(idx)])
            if score < min_similarity:
                continue
            row = classifier_embed_items.iloc[int(idx)]
            entry = row.to_dict()
            profile_rank = rank_classifier_entry_for_profile(entry, object_profile)
            present_in_zone, zone_status = get_zone_status_for_classifier_code(zone_code, entry.get('classifier_code'), context=context)
            candidates.append({'score': score, 'code': normalize_text(entry.get('classifier_code')),
                               'name': normalize_text(entry.get('classifier_name')),
                               'description': normalize_text(entry.get('classifier_description')),
                               'parent_code': normalize_text(entry.get('classifier_parent_code')),
                               'top_level_code': normalize_text(entry.get('classifier_top_level_code')),
                               'present_in_zone': bool(present_in_zone),
                               'zone_status': normalize_text(zone_status) or None, 'profile_rank': int(profile_rank)})
        if not candidates:
            return []
        deduped: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for item in sorted(candidates, key=lambda x: (x['score'], x['present_in_zone'], x['profile_rank'], x['code']),
                           reverse=True):
            code_norm = normalize_text(item.get('code'))
            if not code_norm or code_norm in seen_codes:
                continue
            seen_codes.add(code_norm)
            deduped.append(item)
            if len(deduped) >= top_n:
                break
        return deduped
    except Exception as exc:
        logger.warning("Embedding classifier shortlist failed for '%s' in zone '%s': %s", query_text, zone_code, exc)
        return []


def format_embed_candidates_for_prompt(candidates: list[dict[str, Any]]) -> str:
    """Render embedding shortlist in a compact deterministic prompt-friendly form."""
    if not candidates:
        return '- нет\n'
    lines: list[str] = []
    for item in candidates[:MAX_EMBED_CANDIDATES_IN_PROMPT]:
        meta_parts = [f"score={float(item.get('score', 0.0)):.4f}", f"code={normalize_text(item.get('code'))}",
                      f"name={normalize_text(item.get('name'))}"]
        if normalize_text(item.get('parent_code')):
            meta_parts.append(f"parent={normalize_text(item.get('parent_code'))}")
        if normalize_text(item.get('top_level_code')):
            meta_parts.append(f"top={normalize_text(item.get('top_level_code'))}")
        meta_parts.append(f"present_in_zone={('yes' if item.get('present_in_zone') else 'no')}")
        if normalize_text(item.get('zone_status')):
            meta_parts.append(f"zone_status={normalize_text(item.get('zone_status'))}")
        if int(item.get('profile_rank', 0)) > 0:
            meta_parts.append(f"profile_rank={int(item.get('profile_rank', 0))}")
        lines.append('- ' + '; '.join(meta_parts))
        description = normalize_text(item.get('description'))
        if description:
            lines.append(f'  desc={description[:360]}')
    return '\n'.join(lines) + '\n'


def build_zone_check_prompt(vri_text: str, zone_ref: dict[str, Any], exact_matches: list[dict[str, Any]],
                            actual_zone_code: str, actual_zone_name: Any, actual_share: Any,
                            intersect_codes: Any, context: Any=None) -> str:
    """Build a strict actual-zone prompt using retrieval_text plus embedding shortlist hints."""
    raw_zone_lookup_map = context.raw_zone_lookup if context is not None else {}
    zone_template = raw_zone_lookup_map[zone_ref['zone_code']]
    retrieval_text = normalize_text(zone_template.get('retrieval_text'))
    zone_heading = normalize_text(
        zone_template.get('zone_heading') or zone_template.get('zone_name') or actual_zone_name)
    base_zone_code = normalize_text(zone_template.get('base_zone_code'))
    zone_summary = normalize_text(zone_template.get('zone_summary'))
    object_profile = classify_object_profile(vri_text)
    classifier_snapshot = build_zone_classifier_snapshot(zone_template, context=context)
    shortlist_top_n = TOP_N_EMBED_CANDIDATES_HARD if should_use_deeper_llm_reasoning(vri_text,
                                                                                     zone_template) else TOP_N_EMBED_CANDIDATES
    embed_candidates = build_embed_classifier_candidates(vri_text=vri_text, actual_zone_code=actual_zone_code,
                                                         query_vector=None, top_n=shortlist_top_n,
                                                         min_similarity=MIN_EMBED_CANDIDATE_SIMILARITY,
                                                         context=context)
    exact_lines: list[str] = []
    for match in exact_matches[:5]:
        exact_lines.append(
            f"- section={normalize_text(match.get('section_name'))};"
            f" code={normalize_text(match.get('matched_vri_code'))};"
            f" name={normalize_text(match.get('matched_vri_name'))}\n")
    if not exact_lines:
        exact_lines = ['- нет\n']
    lines = [f'Кадастровый ВРИ: {normalize_text(vri_text)}\n',
             f'Код фактической зоны ПЗЗ: {normalize_text(actual_zone_code)}\n', f'Базовый код зоны: {base_zone_code}\n',
             f'Наименование фактической зоны: {zone_heading}\n',
             f'Профиль объекта: {render_profile_hint(object_profile)}\n', '\n', 'Инструкция по принятию решения:\n',
             '- Сначала определи функциональную категорию кадастрового ВРИ.\n',
             '- Затем ищи только прямое совпадение, прямое покрытие подтипа через описание разрешенного ВРИ\n',
             '  или действительно близкую более широкую категорию в retrieval_text.\n',
             '- Если кадастровая формулировка прямо перечислена в описании разрешенного VRI,\n',
             '  это считается надежным прямым покрытием, даже если название VRI шире.\n',
             '- Ниже дан embedding-shortlist канонических ВРИ классификатора Росреестра. Это только гипотезы,\n',
             '  а не источник истины. Используй shortlist как ограниченный набор кандидатов для сопоставления смысла,\n',
             '  но итоговое решение принимай только по retrieval_text фактической зоны и официальным ВРИ этой зоны.\n',
             '- Кандидаты с present_in_zone=yes особенно важны, но если самый близкий по смыслу кандидат имеет present_in_zone=no,\n',
             '  это не основание автоматически разрешать использование.\n',
             '- Если надежного текстового покрытия нет, верни not_allowed.\n',
             '- Не делай широких аналогий между разными функциональными категориями.\n',
             '- Не предлагай альтернативные зоны.\n', '\n', 'Универсальные ограничения:\n',
             '- Производство / промышленность / цех / завод / склад / логистика не равны торговле,\n',
             '  магазинам, общепиту, бытовому обслуживанию, деловому управлению,\n',
             '  социальной или жилой функции, если это прямо не указано.\n',
             '- Пожарная охрана / спасательные службы / МЧС / ГО и ЧС не равны торговле или жилью;\n',
             '  их можно разрешать только если retrieval_text реально покрывает публичные / общественные /\n',
             '  управленческие / социальные объекты такого типа.\n',
             '- Не путай ИЖС, малоэтажную, среднеэтажную и многоэтажную жилую застройку.\n',
             '- Для verdict=allowed_* в reason обязательно укажи,\n',
             '  какая категория, описание VRI или формулировка зоны из retrieval_text покрывает кадастровый ВРИ.\n',
             '\n', 'Точные / почти точные совпадения в этой зоне:\n', *exact_lines, '\n',
             'Classifier-aligned snapshot фактической зоны:\n', (classifier_snapshot or '- нет данных') + '\n', '\n',
             'Embedding-shortlist канонических ВРИ классификатора:\n',
             format_embed_candidates_for_prompt(embed_candidates), '\n', 'Короткое summary зоны:\n',
             (zone_summary or '- нет данных') + '\n', '\n', 'Полное описание фактической зоны (retrieval_text):\n',
             (retrieval_text or '- нет данных') + '\n', '\n',
             'Верни строго JSON вида: {"verdict":"...","matched_vri_name":"...","matched_vri_code":"...","reason":"..."}\n']
    return ''.join(lines)


def attach_not_allowed_llm_rerank_column(df: pd.DataFrame, cadastral_vri_col: str, context: Any=None) -> pd.DataFrame:
    """Rerank cached global classifier candidates for not_allowed rows in a separate post-processing step.

    LLM calls for unique VRI texts are dispatched concurrently via ThreadPoolExecutor
    so that the GPU pipeline stays saturated instead of waiting for one call at a time.
    The number of workers is read from the ``PIPELINE_LLM_WORKERS`` env var (default 4).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    n_workers = max(1, int(os.getenv("PIPELINE_LLM_WORKERS", "4")))

    work_df = df.copy()
    if 'PZZ_NOT_ALLOWED_TOP5_CANDIDATES' not in work_df.columns:
        work_df['PZZ_NOT_ALLOWED_TOP5_CANDIDATES'] = pd.NA
    mask = (work_df['PZZ_VRI_VERDICT'].map(normalize_text) == 'not_allowed')
    mask_series = pd.Series(mask, index=work_df.index)
    if not bool(mask_series.any()):
        logger.info('No not_allowed rows for post-pipeline LLM rerank.')
        return work_df

    unique_queries: dict[str, dict[str, Any]] = {}
    for idx in work_df.index[mask_series]:
        row = work_df.loc[idx]
        query_text = row.get(cadastral_vri_col)
        query_key = get_not_allowed_query_key(query_text)
        if not query_key:
            continue
        if query_key not in unique_queries:
            unique_queries[query_key] = {'query_text': query_text, 'indexes': []}
        unique_queries[query_key]['indexes'].append(idx)

    logger.info('Post-pipeline not_allowed LLM rerank | rows=%s | unique queries=%s | workers=%s',
                int(mask_series.astype(int).sum()), len(unique_queries), n_workers)

    query_vector_cache, recall_candidates_cache, fast_rerank_cache, llm_rerank_cache = _cache_maps(context)
    _write_lock = threading.Lock()

    def _rerank_one(query_key: str, query_text: Any) -> tuple[str, list[dict[str, Any]]]:
        """Compute fast+LLM candidates for a single unique VRI text."""
        # Fast path: already cached from the main classification loop
        cached = llm_rerank_cache.get(query_key)
        if cached is not None:
            return query_key, cached[:NOT_ALLOWED_CANDIDATES_TOP_N]

        recall_candidates = recall_candidates_cache.get(query_key)
        if recall_candidates is None:
            recall_candidates = build_not_allowed_same_zone_candidates(
                vri_text=query_text,
                actual_zone_code=None,
                query_vector=query_vector_cache.get(query_key),
                top_n=max(NOT_ALLOWED_RECALL_CACHE_TOP_N, NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N),
                min_similarity=NOT_ALLOWED_CANDIDATES_MIN_SIMILARITY,
                context=context,
            )

        fast_candidates = fast_rerank_cache.get(query_key)
        if fast_candidates is None:
            fast_candidates = fast_rerank_not_allowed_candidates(
                vri_text=query_text, candidates=recall_candidates or []
            )
            with _write_lock:
                fast_rerank_cache.setdefault(query_key, [dict(i) for i in fast_candidates])
            fast_candidates = fast_rerank_cache[query_key]

        llm_input_candidates = (fast_candidates or [])[:NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N]

        if not should_run_not_allowed_llm_rerank(query_text, llm_input_candidates):
            result = llm_input_candidates[:NOT_ALLOWED_CANDIDATES_TOP_N]
        else:
            try:
                result = run_not_allowed_rerank_with_llm(
                    vri_text=query_text, candidates=llm_input_candidates
                )
            except Exception as exc:
                logger.warning("Not-allowed LLM rerank failed for '%s': %s", normalize_text(query_text), exc)
                result = llm_input_candidates[:NOT_ALLOWED_CANDIDATES_TOP_N]

        result = result[:NOT_ALLOWED_CANDIDATES_TOP_N]
        with _write_lock:
            llm_rerank_cache.setdefault(query_key, [dict(i) for i in result])
        return query_key, llm_rerank_cache[query_key]

    # Dispatch all unique queries concurrently
    results: dict[str, list[dict[str, Any]]] = {}
    query_items = list(unique_queries.items())
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_map = {
            pool.submit(_rerank_one, qk, payload['query_text']): qk
            for qk, payload in query_items
        }
        for future in tqdm(_as_completed(future_map),
                           total=len(future_map),
                           desc='Not-allowed LLM rerank',
                           **progress_kwargs(leave=False)):
            qk = future_map[future]
            try:
                _, final_candidates = future.result()
            except Exception as exc:
                logger.warning("Rerank future failed for key '%s': %s", qk, exc)
                final_candidates = []
            results[qk] = final_candidates

    # Write serialised results back to the DataFrame
    for query_key, payload in query_items:
        final_candidates = results.get(query_key, [])
        serialized = serialize_not_allowed_same_zone_candidates(final_candidates[:NOT_ALLOWED_CANDIDATES_TOP_N])
        for idx in payload['indexes']:
            work_df.at[idx, 'PZZ_NOT_ALLOWED_TOP5_CANDIDATES'] = serialized

    return work_df
