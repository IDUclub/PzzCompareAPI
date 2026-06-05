from __future__ import annotations

import numpy as np

from .common import *

SECTION_PRIORITY = {'main': 3, 'conditional': 2, 'auxiliary': 1}

SECTION_TO_VERDICT = {'main': 'allowed_main', 'conditional': 'allowed_conditional', 'auxiliary': 'allowed_auxiliary'}

def resolve_zone_reference(actual_zone_code: Any, context: Any=None) -> tuple[Optional[dict[str, Any]], str]:
    """Resolve zone metadata by exact code first, then by base code."""
    zone_code = normalize_text(actual_zone_code)
    if not zone_code:
        return (None, 'missing')
    zone_lookup_map = context.zone_lookup if context is not None else {}
    base_zone_lookup_map = context.base_zone_lookup if context is not None else {}
    if zone_code in zone_lookup_map:
        return (zone_lookup_map[zone_code], 'exact')
    fallback_rows = base_zone_lookup_map.get(zone_code, [])
    if len(fallback_rows) == 1:
        return (fallback_rows[0], 'base_unique')
    return (None, 'not_found')

def find_exact_zone_candidates(vri_text: str, zone_code: Optional[str]=None, context: Any=None) -> list[dict[str, Any]]:
    """Return exact or almost exact catalog matches for a VRI."""
    canon = canonicalize_vri_name(vri_text)
    if not canon:
        return []
    work_df = context.pzz_vri_items_df if context is not None else pd.DataFrame()
    if zone_code:
        work_df = work_df.loc[work_df['zone_code'] == normalize_text(zone_code)]

    # Vectorised boolean mask — avoids row-by-row Python loop
    mask = work_df['catalog_vri_name_norm'].map(lambda x: is_exactish_vri_match(canon, x))
    matched = work_df[mask]
    if matched.empty:
        return []
    candidates = matched[['zone_code', 'base_zone_code', 'zone_name', 'zone_group_name',
                           'zone_summary', 'section_name', 'catalog_vri_name',
                           'catalog_vri_code', 'catalog_vri_description']].rename(columns={
        'catalog_vri_name': 'matched_vri_name',
        'catalog_vri_code': 'matched_vri_code',
        'catalog_vri_description': 'matched_vri_description',
    }).drop_duplicates().sort_values(['zone_code', 'section_name', 'matched_vri_name'])
    return candidates.to_dict(orient='records')

def choose_best_exact_match(matches: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Choose the strongest exact match using section priority."""
    if not matches:
        return None
    ranked = sorted(matches, key=lambda item: (SECTION_PRIORITY.get(normalize_text(item.get('section_name')), 0), normalize_text(item.get('matched_vri_name'))), reverse=True)
    return ranked[0]

def build_catalog_embed_text(vri_name: Any, vri_description: Any='') -> str:
    """Build compact semantic text for embedding comparison."""
    return canonicalize_vri_name(f'{normalize_text(vri_name)} | {normalize_text(vri_description)}')

def detect_direct_description_coverage(vri_text: str, candidate_name: Any, candidate_description: Any) -> dict[str, Any]:
    """Detect when a cadastral phrase is directly covered by a broader VRI description."""
    query_norm = canonicalize_vri_name(vri_text)
    desc_full_norm = canonicalize_vri_name(candidate_description)
    desc_short_norm = build_short_description_match_text(candidate_description, max_tokens=DIRECT_DESC_MAX_TOKENS)
    if not query_norm or not desc_full_norm:
        return {'is_direct': False, 'score': 0.0, 'coverage': 0.0, 'overlap': 0.0, 'seq_ratio': 0.0, 'phrase_hit': False}
    full_metrics = compute_fast_match_score(query_norm, desc_full_norm)
    short_metrics = compute_fast_match_score(query_norm, desc_short_norm)
    best_metrics = full_metrics if full_metrics['score'] >= short_metrics['score'] else short_metrics
    phrase_hit = len(query_norm) >= 10 and query_norm in desc_full_norm
    is_direct = best_metrics['coverage'] >= DIRECT_DESC_COVERAGE_THRESHOLD and best_metrics['overlap'] >= max(DIRECT_DESC_MIN_OVERLAP, min(4, len(tokenize_canonical(query_norm)))) and (phrase_hit or best_metrics['seq_ratio'] >= DIRECT_DESC_SEQ_THRESHOLD)
    return {'is_direct': bool(is_direct), 'score': float(best_metrics['score']), 'coverage': float(best_metrics['coverage']), 'overlap': float(best_metrics['overlap']), 'seq_ratio': float(best_metrics['seq_ratio']), 'phrase_hit': bool(phrase_hit)}

# Жилой дом / жилая застройка / многоквартирный дом (включая частые опечатки и аббревиатуру МКД).
RESIDENTIAL_DWELLING_INDICATOR = re.compile('многоквартирн|многквартирн|могоквартирн|\\bмкд\\b|\\bжил[а-я]*\\s+дом|\\bжил[а-я]*\\s+застройк|\\bжил[а-я]*\\s+здан')
# Явный тип / этажность застройки — при наличии оставляем обычную логику (ИЖС, 2.1.1, 2.3, 2.5, 2.6 и т.п.).
RESIDENTIAL_TYPE_QUALIFIER = re.compile('малоэтажн|среднеэтажн|многоэтажн|высотн|блокиров|секционн|таунхаус|индивидуальн|одноэтажн|двухэтажн|трехэтажн')
# Не жилая застройка в узком смысле (обслуживание жилой застройки 2.7, ЛПХ, соц. обслуживание, коммуналка).
RESIDENTIAL_NON_DWELLING_GUARD = re.compile('обслуживан|коммунальн|социальн|подсобн\\s+хозяйств')

def is_residential_unspecified_vri(vri_text: Any) -> bool:
    """Detect a residential dwelling VRI without an explicit storey / building-type qualifier."""
    canon = canonicalize_vri_name(vri_text)
    if not canon:
        return False
    if RESIDENTIAL_NON_DWELLING_GUARD.search(canon):
        return False
    if not RESIDENTIAL_DWELLING_INDICATOR.search(canon):
        return False
    if RESIDENTIAL_TYPE_QUALIFIER.search(canon):
        return False
    return True

def resolve_residential_unspecified_in_zone(vri_text: str, actual_zone_code: Optional[str], context: Any=None) -> Optional[dict[str, Any]]:
    """Map storey-unspecified residential VRIs to the generic code 2.0 «Жилая застройка».

    Если обобщенный ВРИ прямо разрешен в фактической зоне — отдаем его как allowed_*.
    Если нет — все равно присваиваем 2.0, но помечаем как требующий ручной проверки,
    так как зона может допускать только конкретный класс жилой застройки.
    """
    if not RESIDENTIAL_UNSPECIFIED_TO_GENERIC:
        return None
    if not is_residential_unspecified_vri(vri_text):
        return None
    generic_code = normalize_text(RESIDENTIAL_GENERIC_VRI_CODE)
    if not generic_code:
        return None
    classifier_by_code = context.rosreestr_classifier_by_code if context is not None else {}
    generic_name = normalize_text((classifier_by_code.get(generic_code) or {}).get('name')) or 'Жилая застройка'
    zone_code = normalize_text(actual_zone_code)
    zone_items_map = context.zone_items_lookup if context is not None else {}
    zone_items = zone_items_map.get(zone_code)
    if zone_items is not None and not zone_items.empty:
        matches = [item for _, item in zone_items.iterrows() if normalize_text(item.get('catalog_vri_code')) == generic_code]
        if matches:
            best = max(matches, key=lambda item: SECTION_PRIORITY.get(normalize_text(item.get('section_name')), 0))
            section_name = normalize_text(best.get('section_name'))
            verdict = SECTION_TO_VERDICT.get(section_name, 'allowed_main')
            return {'verdict': verdict, 'matched_vri_name': normalize_text(best.get('catalog_vri_name')) or generic_name, 'matched_vri_code': generic_code, 'match_method': 'residential_unspecified_generic_in_zone', 'reason': f'Жилой объект без указания этажности/типа застройки отнесен к обобщенному ВРИ «{generic_name}» (код {generic_code}), который прямо разрешен в фактической зоне.'}
    return {'verdict': 'unclear', 'matched_vri_name': generic_name, 'matched_vri_code': generic_code, 'match_method': 'residential_unspecified_generic_manual', 'reason': f'Жилой объект без указания этажности/типа застройки отнесен к обобщенному ВРИ «{generic_name}» (код {generic_code}). В фактической зоне обобщенный ВРИ напрямую не указан и допустим только конкретный класс жилой застройки — требуется ручная проверка.'}

def fast_string_match_in_zone(vri_text: str, actual_zone_code: Optional[str], context: Any=None) -> Optional[dict[str, Any]]:
    """Resolve actual-zone VRI using cheap exact/fuzzy checks before embeddings or LLM."""
    zone_code = normalize_text(actual_zone_code)
    if not ENABLE_FAST_STRING_MATCH or not zone_code:
        return None
    zone_items_lookup_map = context.zone_items_lookup if context is not None else {}
    zone_fast_text_lookup_map = context.zone_fast_text_lookup if context is not None else {}
    zone_items = zone_items_lookup_map.get(zone_code)
    if zone_items is None or zone_items.empty:
        return None
    zone_texts = zone_fast_text_lookup_map.get(zone_code, {})
    zone_name_match = normalize_text(zone_texts.get('zone_name_match'))
    zone_summary_match = normalize_text(zone_texts.get('zone_summary_match'))
    retrieval_text_short_match = normalize_text(zone_texts.get('retrieval_text_short_match'))
    query_norm = canonicalize_vri_name(vri_text)
    query_desc_norm = build_short_description_match_text(vri_text)
    if not query_norm:
        return None
    direct_description_candidates: list[dict[str, Any]] = []
    best_candidate: Optional[dict[str, Any]] = None
    second_best_score = -1.0
    for _, item in zone_items.iterrows():
        name_text = normalize_text(item.get('catalog_vri_name_match'))
        desc_text = normalize_text(item.get('catalog_vri_description_match'))
        section_name = normalize_text(item.get('section_name'))
        verdict = SECTION_TO_VERDICT.get(section_name, 'unclear')
        if name_text and is_exactish_vri_match(query_norm, name_text):
            return {'confidence': 'exact_name', 'use_direct': True, 'verdict': verdict, 'matched_vri_name': normalize_text(item.get('catalog_vri_name')), 'matched_vri_code': normalize_text(item.get('catalog_vri_code')), 'reason': 'Точное / почти точное совпадение по названию ВРИ внутри фактической зоны.', 'best_score': 1.0, 'top_matches': [], 'zone_name_score': 0.0, 'zone_summary_score': 0.0}
        direct_desc_metrics = detect_direct_description_coverage(vri_text=vri_text, candidate_name=item.get('catalog_vri_name'), candidate_description=item.get('catalog_vri_description'))
        if direct_desc_metrics['is_direct']:
            direct_description_candidates.append({'score': float(direct_desc_metrics['score']), 'section_name': section_name, 'verdict': verdict, 'matched_vri_name': normalize_text(item.get('catalog_vri_name')), 'matched_vri_code': normalize_text(item.get('catalog_vri_code')), 'matched_vri_description': normalize_text(item.get('catalog_vri_description')), 'metrics': direct_desc_metrics})
        name_metrics = compute_fast_match_score(query_norm, name_text)
        desc_metrics = compute_fast_match_score(query_desc_norm or query_norm, desc_text)
        candidate_score = max(name_metrics['score'], min(1.0, desc_metrics['score'] * 0.98))
        candidate = {'score': float(candidate_score), 'name_metrics': name_metrics, 'desc_metrics': desc_metrics, 'section_name': section_name, 'verdict': verdict, 'matched_vri_name': normalize_text(item.get('catalog_vri_name')), 'matched_vri_code': normalize_text(item.get('catalog_vri_code')), 'matched_vri_description': normalize_text(item.get('catalog_vri_description'))}
        if best_candidate is None or candidate_score > float(best_candidate['score']):
            second_best_score = float(best_candidate['score']) if best_candidate is not None else second_best_score
            best_candidate = candidate
        elif candidate_score > second_best_score:
            second_best_score = candidate_score
    if direct_description_candidates:
        direct_best = sorted(direct_description_candidates, key=lambda item: (SECTION_PRIORITY.get(item['section_name'], 0), item['score'], item['matched_vri_name']), reverse=True)[0]
        return {'confidence': 'direct_description', 'use_direct': True, 'verdict': direct_best['verdict'], 'matched_vri_name': direct_best['matched_vri_name'], 'matched_vri_code': direct_best['matched_vri_code'], 'reason': f"Кадастровая формулировка прямо покрывается описанием разрешенного ВРИ внутри фактической зоны. score={direct_best['metrics']['score']:.4f}; coverage={direct_best['metrics']['coverage']:.4f}; overlap={direct_best['metrics']['overlap']:.0f}; phrase_hit={direct_best['metrics']['phrase_hit']}", 'best_score': float(direct_best['score']), 'top_matches': [{'score': float(item['score']), 'section_name': item['section_name'], 'matched_vri_name': item['matched_vri_name'], 'matched_vri_code': item['matched_vri_code']} for item in direct_description_candidates[:3]], 'zone_name_score': 0.0, 'zone_summary_score': 0.0}
    if best_candidate is None:
        return None
    margin = float(best_candidate['score']) - float(second_best_score)
    name_metrics = best_candidate['name_metrics']
    desc_metrics = best_candidate['desc_metrics']
    zone_name_metrics = compute_fast_match_score(query_norm, zone_name_match)
    zone_summary_metrics = compute_fast_match_score(query_desc_norm or query_norm, zone_summary_match)
    retrieval_short_metrics = compute_fast_match_score(query_desc_norm or query_norm, retrieval_text_short_match)
    strong_by_name = name_metrics['score'] >= FAST_NAME_STRONG_THRESHOLD and name_metrics['coverage'] >= FAST_TOKEN_COVERAGE_THRESHOLD and (name_metrics['overlap'] >= FAST_MIN_TOKEN_OVERLAP) and (margin >= FAST_SCORE_MARGIN_THRESHOLD)
    strong_by_desc = desc_metrics['score'] >= FAST_DESC_STRONG_THRESHOLD and desc_metrics['coverage'] >= FAST_DESC_TOKEN_COVERAGE_THRESHOLD and (desc_metrics['overlap'] >= FAST_MIN_TOKEN_OVERLAP) and (margin >= FAST_SCORE_MARGIN_THRESHOLD)
    strong_by_zone = best_candidate['score'] >= FAST_ZONE_CANDIDATE_MIN_THRESHOLD and (zone_name_metrics['score'] >= FAST_ZONE_NAME_STRONG_THRESHOLD or zone_summary_metrics['score'] >= FAST_ZONE_SUMMARY_STRONG_THRESHOLD or retrieval_short_metrics['score'] >= FAST_ZONE_SUMMARY_STRONG_THRESHOLD)
    borderline = name_metrics['score'] >= FAST_NAME_BORDERLINE_THRESHOLD and name_metrics['coverage'] >= 0.45 or (desc_metrics['score'] >= FAST_DESC_BORDERLINE_THRESHOLD and desc_metrics['coverage'] >= 0.4) or (best_candidate['score'] >= FAST_ZONE_CANDIDATE_MIN_THRESHOLD and (zone_name_metrics['score'] >= FAST_ZONE_NAME_BORDERLINE_THRESHOLD or zone_summary_metrics['score'] >= FAST_ZONE_SUMMARY_BORDERLINE_THRESHOLD or retrieval_short_metrics['score'] >= FAST_ZONE_SUMMARY_BORDERLINE_THRESHOLD))
    top_matches = [{'score': float(best_candidate['score']), 'section_name': best_candidate['section_name'], 'matched_vri_name': best_candidate['matched_vri_name'], 'matched_vri_code': best_candidate['matched_vri_code']}]
    if strong_by_name or strong_by_desc:
        return {'confidence': 'high', 'use_direct': True, 'verdict': best_candidate['verdict'], 'matched_vri_name': best_candidate['matched_vri_name'], 'matched_vri_code': best_candidate['matched_vri_code'], 'reason': f"Быстрое строковое сопоставление внутри фактической зоны. name_score={name_metrics['score']:.4f}; desc_score={desc_metrics['score']:.4f}; margin={margin:.4f}", 'best_score': float(best_candidate['score']), 'top_matches': top_matches, 'zone_name_score': float(zone_name_metrics['score']), 'zone_summary_score': float(zone_summary_metrics['score'])}
    if strong_by_zone:
        return {'confidence': 'zone_semantic_high', 'use_direct': True, 'verdict': best_candidate['verdict'], 'matched_vri_name': best_candidate['matched_vri_name'], 'matched_vri_code': best_candidate['matched_vri_code'], 'reason': f"Быстрое сопоставление по смыслу зоны: кадастровый ВРИ хорошо согласуется с названием/summary фактической зоны и наиболее близким разрешенным ВРИ. candidate_score={float(best_candidate['score']):.4f}; zone_name_score={float(zone_name_metrics['score']):.4f}; zone_summary_score={float(zone_summary_metrics['score']):.4f}; retrieval_short_score={float(retrieval_short_metrics['score']):.4f}", 'best_score': float(best_candidate['score']), 'top_matches': top_matches, 'zone_name_score': float(zone_name_metrics['score']), 'zone_summary_score': float(zone_summary_metrics['score'])}
    if borderline:
        return {'confidence': 'borderline', 'use_direct': False, 'verdict': best_candidate['verdict'], 'matched_vri_name': best_candidate['matched_vri_name'], 'matched_vri_code': best_candidate['matched_vri_code'], 'reason': f"Пограничное быстрое строковое сопоставление внутри фактической зоны. name_score={name_metrics['score']:.4f}; desc_score={desc_metrics['score']:.4f}; zone_name_score={float(zone_name_metrics['score']):.4f}; zone_summary_score={float(zone_summary_metrics['score']):.4f}; retrieval_short_score={float(retrieval_short_metrics['score']):.4f}; margin={margin:.4f}", 'best_score': float(best_candidate['score']), 'top_matches': top_matches, 'zone_name_score': float(zone_name_metrics['score']), 'zone_summary_score': float(zone_summary_metrics['score'])}
    return {'confidence': 'weak', 'use_direct': False, 'verdict': best_candidate['verdict'], 'matched_vri_name': best_candidate['matched_vri_name'], 'matched_vri_code': best_candidate['matched_vri_code'], 'reason': 'Быстрое строковое сопоставление не дало уверенного совпадения ни по ВРИ, ни по смыслу названия/summary фактической зоны.', 'best_score': float(best_candidate['score']), 'top_matches': top_matches, 'zone_name_score': float(zone_name_metrics['score']), 'zone_summary_score': float(zone_summary_metrics['score'])}

def fast_embed_match_in_zone(vri_text: str, actual_zone_code: Optional[str], query_vector: Optional[np.ndarray]=None, context: Any=None) -> Optional[dict[str, Any]]:
    """Run a fast local semantic comparison only inside the actual zone."""
    zone_code = normalize_text(actual_zone_code)
    if not ENABLE_EMBED_FAST_MATCH or not zone_code:
        return None
    zone_items_lookup_map = context.zone_items_lookup if context is not None else {}
    zone_item_embeddings_map = context.zone_item_embeddings if context is not None else {}
    local_vectorizer = context.vectorizer if context is not None else None
    zone_items = zone_items_lookup_map.get(zone_code)
    zone_vectors = zone_item_embeddings_map.get(zone_code)
    if zone_items is None or zone_items.empty or zone_vectors is None or (zone_vectors.size == 0):
        return None
    if query_vector is None:
        query_matrix = local_vectorizer.embed_many(texts=[build_catalog_embed_text(vri_text)], batch_size=1)
        if query_matrix.size == 0:
            return None
        query_vector = query_matrix[0]
    scores = (zone_vectors @ query_vector).astype(float)
    if scores.size == 0:
        return None
    top_ids = np.argsort(-scores)[:min(3, len(scores))]
    best_idx = int(top_ids[0])
    best_score = float(scores[best_idx])
    second_score = float(scores[int(top_ids[1])]) if len(top_ids) > 1 else -1.0
    margin = best_score - second_score
    best_row = zone_items.iloc[best_idx]
    section_name = normalize_text(best_row['section_name'])
    verdict = SECTION_TO_VERDICT.get(section_name, 'unclear')
    if best_score >= LOCAL_EMBED_STRONG_THRESHOLD and margin >= LOCAL_EMBED_MARGIN_THRESHOLD:
        confidence = 'strong'
        use_direct = True
    elif best_score >= LOCAL_EMBED_BORDERLINE_THRESHOLD and margin >= LOCAL_EMBED_MARGIN_THRESHOLD:
        confidence = 'borderline'
        use_direct = False
    else:
        confidence = 'weak'
        use_direct = False
    top_matches = []
    for idx in top_ids:
        row = zone_items.iloc[int(idx)]
        top_matches.append({'score': float(scores[int(idx)]), 'section_name': normalize_text(row['section_name']), 'matched_vri_name': normalize_text(row['catalog_vri_name']), 'matched_vri_code': normalize_text(row['catalog_vri_code'])})
    return {'confidence': confidence, 'use_direct': use_direct, 'verdict': verdict, 'matched_vri_name': normalize_text(best_row['catalog_vri_name']), 'matched_vri_code': normalize_text(best_row['catalog_vri_code']), 'reason': f'Быстрое локальное semantic-сопоставление внутри фактической зоны. best_score={best_score:.4f}; margin={margin:.4f}', 'best_score': best_score, 'margin': margin, 'top_matches': top_matches}

ZONE_CHECK_SYSTEM_PROMPT = (
    'Ты проверяешь соответствие кадастрового ВРИ фактической зоне ПЗЗ, где реально расположен участок.\n\n'
    'Тебе передают только одну фактическую зону ПЗЗ. Нельзя предлагать другие зоны и нельзя сравнивать с другими зонами.\n\n'
    'Важно:\nretrieval_text этой зоны содержит не только перечень ВРИ, но и:\n'
    '- наименование зоны,\n- описание / целевое назначение зоны,\n- примечания,\n'
    '- основные, условно разрешенные и вспомогательные виды использования.\n\n'
    'Опирайся только на retrieval_text этой зоны, на официальный список ВРИ этой зоны и на переданный нормализованный профиль объекта как на подсказку.\n\n'
    'Рабочий порядок:\n'
    '1. Сначала определи функциональную категорию кадастрового ВРИ.\n'
    '2. Затем проверь, есть ли для него основание в retrieval_text одним из двух способов:\n'
    '   A) прямое или действительно близкое смысловое покрытие через перечисленные ВРИ и их описания;\n'
    '   B) прямое покрытие через наименование зоны или описание / целевое назначение зоны,\n'
    '      если из retrieval_text явно следует, что зона предназначена именно для таких объектов или территорий.\n'
    '3. Подтип считается прямым покрытием, если кадастровая формулировка прямо перечислена\n'
    '   или надежно охватывается описанием разрешенного ВРИ внутри retrieval_text,\n'
    '   даже когда название самого VRI шире, чем кадастровая формулировка.\n'
    '4. Если найдено основание по A или по B и оно не противоречит ограничениям зоны, разрешай использование.\n'
    '5. Если такого основания нет, возвращай not_allowed.\n\n'
    'Строгие правила:\n'
    '1. Main важнее conditional, conditional важнее auxiliary.\n'
    '2. Официальные коды и наименования ВРИ считай каноническими.\n'
    '3. Не делай широких аналогий между разными функциональными категориями.\n'
    '4. Поверхностное сходство слов не является основанием для allowed_*.\n'
    '5. Если у тебя нет явного основания из retrieval_text, выбирай not_allowed.\n'
    '6. Нельзя отказывать только потому, что фраза отсутствует как буквальное название ВРИ,\n'
    '   если она прямо покрывается описанием разрешенного VRI, назначением самой зоны\n'
    '   или ее наименованием / описанием.\n'
    '7. Если использование следует только из названия/описания зоны, а не из буквального VRI,\n'
    '   это допустимо только когда такое толкование прямо и естественно следует из retrieval_text.\n'
    '8. Не путай типы жилой застройки:\n'
    '   - индивидуальный / одна семья / до 3 этажей -> ИЖС / Ж-1 тип\n'
    '   - малоэтажный многоквартирный -> 2.1.1 / Ж-2 тип\n'
    '   - среднеэтажный многоквартирный -> 2.5 / Ж-3 тип\n'
    '   - многоэтажный многоквартирный -> 2.6 / Ж-4 тип\n'
    '9. Не путай личное подсобное хозяйство с подсобными / вспомогательными сооружениями:\n'
    '   - подсобные / вспомогательные сооружения могут покрываться описанием жилого VRI;\n'
    '   - личное подсобное хозяйство разрешай только при прямом наличии соответствующего VRI или действительно близкой канонической категории.\n\n'
    'Универсальные смысловые ограничения:\n'
    '- промышленное производство / производственные объекты / завод / фабрика / цех / склад / логистика\n'
    '  не равны магазинам, торговым объектам, общественному питанию, бытовому обслуживанию,\n'
    '  деловому управлению, социальному обслуживанию, культурному развитию и жилой застройке,\n'
    '  если это прямо не указано в retrieval_text.\n'
    '- жилой объект не равен автоматически ИЖС: учитывай слова про одну семью, квартиры, этажность и тип застройки.\n'
    '- пожарная охрана / пожарное депо / спасательные службы / МЧС / ГО и ЧС\n'
    '  не равны торговле или жилью, но могут относиться к широким публичным / общественным /\n'
    '  общественно-деловым / управленческим / социальным объектам только если retrieval_text\n'
    '  действительно содержит такую широкую категорию, которая по смыслу покрывает эти объекты.\n'
    '- коммунальная инфраструктура (газопровод, водопровод, ЛЭП, подстанция, котельная, очистные)\n'
    '  может относиться к коммунальному обслуживанию, только если retrieval_text это поддерживает.\n'
    '- улично-дорожная сеть, проезды, тротуары, пешеходные переходы, набережные и аналогичные элементы\n'
    '  могут покрываться ВРИ общего пользования или транспортной инфраструктуры, если это прямо следует из описания VRI.\n'
    '- религиозные, историко-культурные, образовательные, медицинские, мемориальные,\n'
    '  музейные, парковые и социальные объекты разрешай только при наличии соответствующей\n'
    '  или действительно более широкой категории в retrieval_text, либо при прямом указании\n'
    '  в названии/описании зоны, что зона предназначена для таких объектов или территорий.\n\n'
    'Как трактовать историко-культурные и специальные зоны:\n'
    '- Если в retrieval_text зона прямо названа как зона парков, исторических парков, мемориальных территорий,\n'
    '  музейных, дворцово-парковых, историко-культурных, религиозных или иных специальных объектов,\n'
    '  такое наименование и описание зоны считается сильным основанием для allowed_*,\n'
    '  даже если кадастровая формулировка не совпадает буквально с названием одного VRI.\n'
    '- Но такое основание нельзя переносить на чужие функциональные категории\n'
    '  (например производство, торговля, жилье, склад, промышленность), если они не названы в retrieval_text.\n\n'
    'Требования к ответу:\n'
    '- Для allowed_* в reason обязательно назови:\n'
    '  1) либо конкретную категорию / формулировку ВРИ из retrieval_text,\n'
    '  2) либо конкретную фразу из описания разрешенного VRI,\n'
    '  3) либо конкретную фразу из наименования / описания зоны в retrieval_text,\n'
    '     которая прямо покрывает кадастровый ВРИ.\n'
    '- Если такой опорной формулировки назвать нельзя, верни not_allowed и укажи причину, по которой определил в эту категорию.\n'
    '- Не предлагай альтернативные зоны.\n'
    '- Если основание двусмысленное и недостаточно надежное, верни unclear.\n\n'
    'Разрешенные verdict:\n'
    '- allowed_main\n- allowed_conditional\n- allowed_auxiliary\n- not_allowed\n- unclear\n\n'
    'Верни строго JSON по схеме.\n'
)

ZONE_CHECK_SCHEMA = {'type': 'object', 'properties': {'verdict': {'type': 'string'}, 'matched_vri_name': {'type': ['string', 'null']}, 'matched_vri_code': {'type': ['string', 'null']}, 'reason': {'type': 'string'}}, 'required': ['verdict', 'matched_vri_name', 'matched_vri_code', 'reason']}

FALLBACK_SCHEMA = {'type': 'object', 'properties': {'suggested_code': {'type': ['string', 'null']}, 'suggested_description': {'type': ['string', 'null']}, 'verdict': {'type': 'string'}, 'matched_vri_name': {'type': ['string', 'null']}, 'matched_vri_code': {'type': ['string', 'null']}, 'reason': {'type': 'string'}}, 'required': ['suggested_code', 'suggested_description', 'verdict', 'matched_vri_name', 'matched_vri_code', 'reason']}

FALLBACK_SYSTEM_PROMPT = 'Ты подбираешь альтернативную зону ПЗЗ только если кадастровый ВРИ не подходит фактической зоне.\n\nПравила:\n1. Выбирай только из переданного списка кандидатов.\n2. Предпочитай явные совпадения ВРИ.\n3. Если в списке нет надежного кандидата, верни verdict=not_found и suggested_code=null.\n4. Если кандидат подходит как условно разрешенный или вспомогательный вид, это нужно указать соответствующим verdict.\n\nРазрешенные verdict:\n- allowed_main\n- allowed_conditional\n- allowed_auxiliary\n- not_found\n\nВерни строго JSON по схеме.\n'.strip()

def build_zone_check_prompt(vri_text: str, zone_ref: dict[str, Any], exact_matches: list[dict[str, Any]], actual_zone_code: str, actual_zone_name: Any, actual_share: Any, intersect_codes: Any, context: Any=None) -> str:
    """Build a strict actual-zone prompt using retrieval_text of the factual zone."""
    raw_zone_lookup_map = context.raw_zone_lookup if context is not None else {}
    zone_template = raw_zone_lookup_map[zone_ref['zone_code']]
    retrieval_text = normalize_text(zone_template.get('retrieval_text'))
    zone_heading = normalize_text(zone_template.get('zone_heading') or zone_template.get('zone_name') or actual_zone_name)
    base_zone_code = normalize_text(zone_template.get('base_zone_code'))
    zone_summary = normalize_text(zone_template.get('zone_summary'))
    exact_lines: list[str] = []
    for match in exact_matches[:5]:
        exact_lines.append(f"- section={normalize_text(match.get('section_name'))}; code={normalize_text(match.get('matched_vri_code'))}; name={normalize_text(match.get('matched_vri_name'))}\n")
    if not exact_lines:
        exact_lines = ['- нет\n']
    lines = [f'Кадастровый ВРИ: {normalize_text(vri_text)}\n', f'Код фактической зоны ПЗЗ: {normalize_text(actual_zone_code)}\n', f'Базовый код зоны: {base_zone_code}\n', f'Наименование фактической зоны: {zone_heading}\n', '\n', 'Инструкция по принятию решения:\n', '- Сначала определи функциональную категорию кадастрового ВРИ.\n', '- Затем ищи только прямое совпадение, прямое покрытие подтипа через описание разрешенного ВРИ\n', '  или действительно близкую более широкую категорию в retrieval_text.\n', '- Если кадастровая формулировка прямо перечислена в описании разрешенного VRI,\n', '  это считается надежным прямым покрытием, даже если название VRI шире.\n', '- Если надежного текстового покрытия нет, верни not_allowed.\n', '- Не делай широких аналогий между разными функциональными категориями.\n', '- Не предлагай альтернативные зоны.\n', '\n', 'Универсальные ограничения:\n', '- Производство / промышленность / цех / завод / склад / логистика не равны торговле,\n', '  магазинам, общепиту, бытовому обслуживанию, деловому управлению,\n', '  социальной или жилой функции, если это прямо не указано.\n', '- Пожарная охрана / спасательные службы / МЧС / ГО и ЧС не равны торговле или жилью;\n', '  их можно разрешать только если retrieval_text реально покрывает публичные / общественные /\n', '  управленческие / социальные объекты такого типа.\n', '- Не путай ИЖС, малоэтажную, среднеэтажную и многоэтажную жилую застройку.\n', '- Для verdict=allowed_* в reason обязательно укажи,\n', '  какая категория, описание VRI или формулировка зоны из retrieval_text покрывает кадастровый ВРИ.\n', '\n', 'Точные / почти точные совпадения в этой зоне:\n', *exact_lines, '\n', 'Короткое summary зоны:\n', (zone_summary or '- нет данных') + '\n', '\n', 'Полное описание фактической зоны (retrieval_text):\n', (retrieval_text or '- нет данных') + '\n', '\n', 'Верни строго JSON вида: {"verdict":"...","matched_vri_name":"...","matched_vri_code":"...","reason":"..."}\n']
    return ''.join(lines)

def run_zone_check_with_llm(prompt: str, think_override: Any=None, context: Any=None) -> dict[str, Any]:
    """Run actual-zone validation with LLM."""
    llm = context.llm_client if context is not None else None
    if llm is None:
        raise RuntimeError("LLM client not available: context.llm_client is None")
    return llm.complete_json(user_prompt=prompt, system_prompt=ZONE_CHECK_SYSTEM_PROMPT, schema=ZONE_CHECK_SCHEMA, model=LLM_MODEL, think_override=think_override)

def heuristic_zone_decision(zone_ref: Optional[dict[str, Any]], exact_matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback heuristic for actual zone decision when LLM is unavailable."""
    best_match = choose_best_exact_match(exact_matches)
    if zone_ref is None:
        return {'verdict': 'no_zone_metadata', 'matched_vri_name': None, 'matched_vri_code': None, 'reason': 'Для фактической зоны не найдено описание в шаблоне ПЗЗ.'}
    if best_match is not None:
        return {'verdict': SECTION_TO_VERDICT.get(best_match['section_name'], 'unclear'), 'matched_vri_name': best_match['matched_vri_name'], 'matched_vri_code': best_match['matched_vri_code'], 'reason': 'Решение принято по точному / почти точному совпадению внутри фактической зоны.'}
    return {'verdict': 'unclear', 'matched_vri_name': None, 'matched_vri_code': None, 'reason': 'Точного совпадения внутри фактической зоны нет; без LLM требуется ручная проверка.'}

def heuristic_fallback_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback heuristic for alternative zone search when LLM is unavailable."""
    if not candidates:
        return {'suggested_code': None, 'suggested_description': None, 'verdict': 'not_found', 'matched_vri_name': None, 'matched_vri_code': None, 'reason': 'Кандидаты для альтернативной зоны не найдены.'}
    best_candidate = candidates[0]
    best_exact = None
    for item in best_candidate.get('matched_items', []):
        if item.get('source') == 'exact_match':
            best_exact = item
            break
    if best_exact is not None:
        section_name = normalize_text(best_exact.get('section_name'))
        verdict = SECTION_TO_VERDICT.get(section_name, 'not_found')
        return {'suggested_code': best_candidate['code'], 'suggested_description': best_candidate['description'], 'verdict': verdict, 'matched_vri_name': best_exact.get('matched_vri_name'), 'matched_vri_code': best_exact.get('matched_vri_code'), 'reason': 'Альтернативная зона выбрана по точному совпадению в глобальном каталоге.'}
    return {'suggested_code': None, 'suggested_description': None, 'verdict': 'not_found', 'matched_vri_name': None, 'matched_vri_code': None, 'reason': 'Надежная альтернативная зона не найдена без LLM.'}

def build_fallback_prompt(vri_text: str, actual_zone_code: Any, actual_zone_name: Any, candidates: list[dict[str, Any]]) -> str:
    """Build prompt for alternative zone suggestion."""
    lines = [f'Кадастровый ВРИ: {normalize_text(vri_text)}', f'Фактическая зона, где участок расположен: {normalize_text(actual_zone_code)} | {normalize_text(actual_zone_name)}', '', 'Кандидаты для альтернативного поиска:']
    for candidate in candidates:
        lines.append(f"- code={candidate['code']}; base_code={candidate['base_code']}; name={candidate['description']}; group={candidate['group']}; score={float(candidate['score']):.4f}; main_vri_names={candidate['main_vri_names']}; conditional_vri_names={candidate['conditional_vri_names']}; auxiliary_vri_names={candidate['auxiliary_vri_names']}; summary={candidate['summary']}")
        for item in candidate.get('matched_items', [])[:6]:
            lines.append(f"  evidence: source={item['source']}; section={item['section_name']}; matched_vri_name={item['matched_vri_name']}; matched_vri_code={item['matched_vri_code']}; contribution={float(item['contribution']):.4f}; note={item['matched_vri_description']}")
    lines += ['', 'Верни JSON: suggested_code, suggested_description, verdict, matched_vri_name, matched_vri_code, reason.']
    return '\n'.join(lines)
