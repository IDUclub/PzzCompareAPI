from __future__ import annotations

from .common import *

SECTION_PRIORITY = {'main': 3, 'conditional': 2, 'auxiliary': 1}
SECTION_TO_VERDICT = {'main': 'allowed_main', 'conditional': 'allowed_conditional', 'auxiliary': 'allowed_auxiliary'}


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

SERVICE_HIERARCHY_PATH_CANDIDATES = ['data/services_hierarchy.json', '/mnt/data/services_hierarchy.json', 'services_hierarchy.json']

PHYSICAL_HIERARCHY_PATH_CANDIDATES = ['data/physical_objects_hierarchy.json', '/mnt/data/physical_objects_hierarchy.json', 'physical_objects_hierarchy.json']

def resolve_optional_existing_path(path_candidates: list[str]) -> Optional[str]:
    """Return the first existing path or None when all candidates are missing."""
    for path in path_candidates:
        if path and os.path.exists(path):
            return path
    return None


def build_zone_classifier_snapshot(zone_ref: Optional[dict[str, Any]], context: Any=None) -> str:
    """Render classifier-aligned VRI lists for one zone in a compact prompt-friendly format."""
    if not zone_ref:
        return '- нет данных'
    section_labels = {'main': 'MAIN', 'conditional': 'CONDITIONAL', 'auxiliary': 'AUXILIARY'}
    lines: list[str] = []
    for section_name in ('main', 'conditional', 'auxiliary'):
        items = zone_ref.get(section_name) or []
        if not items:
            continue
        lines.append(f'[{section_labels[section_name]}]')
        for item in items:
            vri_code = normalize_text(item.get('vri_code') or item.get('catalog_vri_code'))
            classifier_item = (context.rosreestr_classifier_by_code if context is not None else {}).get(vri_code, {})
            vri_name = normalize_text(item.get('vri_name') or item.get('catalog_vri_name') or classifier_item.get('name'))
            vri_description = normalize_text(item.get('vri_description') or item.get('catalog_vri_description') or classifier_item.get('description'))
            vri_parent_code = normalize_text(item.get('vri_parent_code') or classifier_item.get('parent_code'))
            vri_top_level_code = normalize_text(item.get('vri_top_level_code') or classifier_item.get('top_level_code'))
            original_vri_name = normalize_text(item.get('original_vri_name'))
            normalized_by_classifier = bool(item.get('normalized_by_rosreestr_classifier'))
            meta_parts = [part for part in [f'code={vri_code}', f'name={vri_name}'] if part]
            if vri_parent_code:
                meta_parts.append(f'parent={vri_parent_code}')
            if vri_top_level_code:
                meta_parts.append(f'top={vri_top_level_code}')
            if normalized_by_classifier:
                meta_parts.append('classifier_aligned=yes')
            if original_vri_name and original_vri_name != vri_name:
                meta_parts.append(f'source_name={original_vri_name}')
            lines.append('- ' + '; '.join(meta_parts))
            if vri_description:
                lines.append(f'  desc={vri_description[:320]}')
    return '\n'.join(lines) if lines else '- нет данных'

LLM_DEEP_REASONING_PATTERNS = ['музей[-\\s]?заповед', 'историко[-\\s]?художествен', 'дворцово[-\\s]?парков', 'объект(?:ы|ов)?\\s+культурн(?:ого|ых)?\\s+наслед', 'историческ', 'культурн', 'ансамбл', 'заповедник', 'мемориальн']

def should_use_deeper_llm_reasoning(vri_text: Any, zone_ref: Optional[dict[str, Any]], context: Any=None) -> bool:
    """Return True when the case should be sent to the LLM with deeper reasoning enabled."""
    raw_text = normalize_text(vri_text)
    if not raw_text:
        return False
    canon = canonicalize_vri_name(raw_text)
    zone_text = canonicalize_vri_name(' '.join([normalize_text((zone_ref or {}).get('zone_name')), normalize_text((zone_ref or {}).get('zone_heading')), normalize_text((zone_ref or {}).get('zone_summary')), normalize_text((zone_ref or {}).get('retrieval_text')), normalize_text((zone_ref or {}).get('main_vri_full')), normalize_text((zone_ref or {}).get('conditional_vri_full')), normalize_text((zone_ref or {}).get('auxiliary_vri_full'))]))
    text_has_hard_pattern = any((re.search(pattern, canon) for pattern in LLM_DEEP_REASONING_PATTERNS))
    zone_has_historical_context = any((re.search(pattern, zone_text) for pattern in ['историческ', 'культурн', 'парк', 'заповед', 'музе', 'рекреаци']))
    has_complex_surface_form = len(raw_text) >= max(LLM_LONG_TEXT_HARD_CASE_MIN_LEN, 80) or any((marker in raw_text for marker in ['"', '«', '»']))
    return bool(text_has_hard_pattern and (zone_has_historical_context or has_complex_surface_form))

def load_optional_json(path_candidates: list[str]) -> Any:
    """Load JSON from the first existing candidate path or return None."""
    path = resolve_optional_existing_path(path_candidates)
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

SERVICE_HIERARCHY_RAW = load_optional_json(SERVICE_HIERARCHY_PATH_CANDIDATES)

PHYSICAL_HIERARCHY_RAW = load_optional_json(PHYSICAL_HIERARCHY_PATH_CANDIDATES)

SAFE_VRI_REPLACEMENTS = [('^для\\s+', ''), ('^под\\s+', ''), ('^по\\s+', ''), ('\\bижс\\b', 'индивидуальное жилищное строительство'), ('\\bдля индивидуального жилищного строительства\\b', 'индивидуальное жилищное строительство'), ('\\bпод строительство индивидуального жилого дома\\b', 'индивидуальное жилищное строительство'), ('\\bдля строительства индивидуального жилого дома\\b', 'индивидуальное жилищное строительство'), ('\\bпод жилую застройку\\s*индивидуальную\\b', 'индивидуальное жилищное строительство'), ('\\bдома индивидуальной жилой застройки\\b', 'индивидуальное жилищное строительство'), ('\\bразмещение домов индивидуальной жилой застройки\\b', 'индивидуальное жилищное строительство'), ('\\bиндивидуальные жилые дома\\b', 'индивидуальное жилищное строительство'), ('\\bземельный участок под жилой дом\\b', 'размещение жилого дома'), ('\\bземельные участки под жилой дом\\b', 'размещение жилого дома'), ('\\bдля размещения существующего жилого дома\\b', 'размещение жилого дома'), ('\\bдля размещения индивидуального жилого дома\\b', 'индивидуальное жилищное строительство'), ('\\bдля размещения жилого дома\\b', 'размещение жилого дома'), ('\\bпод размещение жилого дома\\b', 'размещение жилого дома'), ('\\bпод жилой дом\\b', 'размещение жилого дома'), ('\\bпод размещение одноэтажного жилого дома с надворными постройками\\b', 'индивидуальное жилищное строительство'), ('\\bдля размещения одноэтажного жилого дома с надворными постройками\\b', 'индивидуальное жилищное строительство'), ('\\bпод размещение жилого дома одноэтажного с надворными постройками\\b', 'индивидуальное жилищное строительство'), ('\\bмногоквартирные многоэтажные дома\\b', 'многоэтажная жилая застройка'), ('\\bразмещения\\b', 'размещение'), ('\\bведения садоводства\\b', 'садоводство'), ('\\bдля ведения садоводства\\b', 'садоводство'), ('\\bдля\\s+ведения\\s+садоводства\\b', 'садоводство'), ('\\bсадоводства\\b', 'садоводство'), ('\\bдля\\s+ведения\\s+огородничества\\b', 'огородничество'), ('\\bведения\\s+огородничества\\b', 'огородничество'), ('\\bдля\\s+ведения\\s+личного\\s+подсобного\\s+хозяйства\\b', 'личное подсобное хозяйство'), ('\\bразмещение\\s+гаражей\\s+для\\s+собственных\\s+нужд\\b', 'гараж'), ('\\bдля\\s+размещения\\s+гаражей\\b', 'гараж'), ('\\bдля\\s+размещения\\s+гаража\\b', 'гараж'), ('\\bдля\\s+размещения\\s+кирпичного\\s+гаража\\b', 'гараж'), ('\\bдля\\s+размещения\\s+трансформаторной\\s+подстанции\\b', 'трансформаторная подстанция'), ('\\bразмещение\\s+трансформаторной\\s+подстанции\\b', 'трансформаторная подстанция'), ('\\bтп\\b', 'трансформаторная подстанция'), ('\\bрп\\b', 'трансформаторная подстанция'), ('\\bкнс\\b', 'канализационная насосная станция'), ('\\bдля\\s+размещения\\s+существующего\\s+газорегуляторного\\s+пункта\\b', 'газорегуляторный пункт'), ('\\bдля\\s+размещения\\s+газорегуляторного\\s+пункта\\b', 'газорегуляторный пункт'), ('\\bгазорегуляторного\\s+пункта\\b', 'газорегуляторный пункт'), ('\\bгазорегуляторных\\s+пунктов\\b', 'газорегуляторный пункт'), ('\\bгрп\\b', 'газорегуляторный пункт'), ('\\bгру\\b', 'газорегуляторный пункт'), ('\\bдля\\s+размещения\\s+объектов\\s+электроснабжения\\b', 'электроснабжение'), ('\\bдля\\s+размещения\\s+объекта\\s+электроснабжения\\b', 'электроснабжение'), ('\\bобъектов\\s+электроснабжения\\b', 'электроснабжение'), ('\\bобъекта\\s+электроснабжения\\b', 'электроснабжение'), ('\\bдля\\s+размещения\\s+подстанций\\b', 'подстанция'), ('\\bдля\\s+размещения\\s+существующих\\s+подстанций\\b', 'подстанция'), ('\\bподстанций\\b', 'подстанция'), ('\\bстанци[яи]\\s+подкачк(?:и)?\\s+воды\\b', 'насосная станция'), ('\\bстанци[яи]\\s+подкачк(?:и)?\\b', 'насосная станция'), ('\\bподкачк(?:и)?\\s+воды\\b', 'насосная станция'), ('\\bстанция\\s+подкачки\\b', 'насосная станция'), ('\\bнасосной\\s+подкачки\\b', 'насосная станция'), ('\\bуличн\\s+дорожн(?:ая|ой)?\\s+сет(?:ь|и)\\b', 'улично дорожная сеть'), ('\\bподсобн(?:ых|ые|ое|ого|ому|ым|ыми)?\\s+сооружен(?:ий|ия|ие|ием|иями)?\\b', 'подсобные сооружения'), ('\\bвспомогательн(?:ых|ые|ое|ого|ому|ым|ыми)?\\s+сооружен(?:ий|ия|ие|ием|иями)?\\b', 'вспомогательные сооружения'), ('\\bхозяйственн(?:ых|ые|ое|ого|ому|ым|ыми)?\\s+постро(?:ек|йк(?:а|и|е|ой|ами)?)\\b', 'хозяйственные постройки'),
    ('\\bзру\\b', 'электрическая подстанция'),
    ('\\bзтп\\b', 'трансформаторная подстанция'),
    ('\\bктп\\b', 'трансформаторная подстанция'),
    ('\\bтэц\\b', 'котельная теплоснабжение'),
    ('\\bгск\\b', 'гаражный кооператив')]

SHORT_DESC_STOP_TOKENS = {'земельн', 'участок', 'территор', 'объект', 'капитальн', 'строительств', 'предназнач', 'цел', 'такж', 'включа', 'содержан', 'дан', 'вид', 'разреш', 'использован', 'код', 'размещен', 'размеща', 'границ', 'населен', 'пункт', 'данн', 'связ', 'осуществлен', 'обеспечен', 'прием', 'граждан', 'вопрос', 'назначен', 'организац'}

def canonicalize_vri_name(value: Any) -> str:
    """Canonicalize VRI text with conservative domain normalization."""
    text = normalize_russian_text(value)
    if not text:
        return ''
    text = re.sub('\\b\\d+/\\d+\\b', ' ', text)
    text = re.sub('\\b\\d+\\s*доли\\b', ' ', text)
    text = re.sub('\\bсуществующ(?:его|ий|ая|ие)?\\b', ' ', text)
    text = re.sub('\\bнадворн(?:ыми|ые|ая|ого)?\\s+постройк(?:ами|и|а|ой)?\\b', ' ', text)
    for pattern, replacement in SAFE_VRI_REPLACEMENTS:
        text = re.sub(pattern, replacement, text).strip()
    text = re.sub('\\s+', ' ', text).strip()
    tokens = normalize_match_tokens(text)
    return ' '.join(tokens)

def build_short_description_match_text(value: Any, max_tokens: int=FAST_DESC_MAX_TOKENS) -> str:
    """Build compact searchable text for short descriptions without dropping key functional terms."""
    tokens = normalize_match_tokens(value)
    if not tokens:
        return ''
    filtered: list[str] = []
    for token in tokens:
        if token in SHORT_DESC_STOP_TOKENS:
            continue
        filtered.append(token)
        if len(filtered) >= max_tokens:
            break
    return ' '.join(filtered if filtered else tokens[:max_tokens])

def _iter_service_leaf_paths(nodes: list[dict[str, Any]], path: tuple[str, ...]=()):
    """Yield service leaf names together with their hierarchy path."""
    for node in nodes or []:
        node_name = normalize_text(node.get('name'))
        next_path = path + ((node_name,) if node_name else ())
        children = node.get('children') or []
        for child in children:
            if isinstance(child, dict) and 'service_type_id' in child:
                leaf_name = normalize_text(child.get('name'))
                full_path = next_path + ((leaf_name,) if leaf_name else ())
                if leaf_name:
                    yield full_path
            elif isinstance(child, dict):
                yield from _iter_service_leaf_paths([child], next_path)

def _iter_physical_leaf_paths(nodes: list[dict[str, Any]], path: tuple[str, ...]=()):
    """Yield physical-object leaf names together with their hierarchy path."""
    for node in nodes or []:
        node_name = normalize_text(node.get('name'))
        next_path = path + ((node_name,) if node_name else ())
        children = node.get('children') or []
        for child in children:
            if isinstance(child, dict) and 'physical_object_type_id' in child:
                leaf_name = normalize_text(child.get('name'))
                full_path = next_path + ((leaf_name,) if leaf_name else ())
                if leaf_name:
                    yield full_path
            elif isinstance(child, dict):
                yield from _iter_physical_leaf_paths([child], next_path)

def _family_from_service_path(path: tuple[str, ...]) -> Optional[str]:
    """Map service hierarchy path to a generic functional family."""
    text = canonicalize_vri_name(' | '.join(path))
    if not text:
        return None
    if 'образован' in text:
        if 'профессиональн' in text:
            return 'education_professional'
        if 'дошкольн' in text:
            return 'education_preschool'
        if 'общеобразовательн' in text or 'средн общ' in text or 'школ' in text:
            return 'education_school'
        return 'education'
    if 'здравоохран' in text or 'амбулаторн' in text or 'стационарн' in text or ('фармацевт' in text):
        return 'healthcare'
    if 'социальн защит' in text:
        return 'social_service'
    if 'государствен' in text or 'муниципальн' in text:
        return 'public_admin'
    if 'обществен питан' in text:
        return 'public_catering'
    if 'спортивн' in text:
        return 'sport'
    if 'культур' in text or 'досуг' in text or 'достопримечатель' in text:
        return 'culture'
    if 'туристическ' in text or 'гостинич' in text:
        return 'hotel'
    if 'магазин' in text or 'продовольств' in text:
        return 'retail'
    if 'финансов' in text:
        return 'business'
    if 'услуг уход' in text or 'парикмахер' in text or 'салон красот' in text or ('бан' in text):
        return 'consumer_service'
    if 'услуг для питомц' in text or 'ветеринар' in text:
        return 'veterinary'
    if 'транспорт' in text or 'вокзал' in text or 'остановк' in text:
        return 'transport'
    if 'безопасност' in text or 'пожар' in text or 'полиц' in text:
        return 'public_safety'
    return None

def _family_from_physical_path(path: tuple[str, ...]) -> Optional[str]:
    """Map physical-object hierarchy path to a generic functional family."""
    text = canonicalize_vri_name(' | '.join(path))
    if not text:
        return None
    if 'электроснабж' in text or 'электрическ подстанц' in text or 'электростанц' in text or ('газорегуляторн пункт' in text) or ('подстанц' in text):
        return 'communal_engineering'
    if 'теплоснабж' in text or 'котельн' in text:
        return 'communal_engineering'
    if 'водоснабж' in text or 'водозабор' in text or 'насосн станц' in text or ('станц подкачк' in text) or ('подкачк воды' in text) or ('водонапорн башн' in text):
        return 'communal_engineering'
    if 'водоотведен' in text or 'водоочистн' in text or 'канализац' in text:
        return 'communal_engineering'
    if 'линейн объект' in text or 'линия электропередач' in text or 'газопровод' in text or ('сеть теплоснабж' in text) or ('сеть водоотведен' in text) or ('сеть электросвяз' in text):
        return 'communal_engineering'
    if 'объект связ' in text:
        return 'communal_engineering'
    if 'газ' in text:
        return 'communal_engineering'
    if 'жилой дом' in text:
        return 'residential'
    if 'уличн дорожн' in text or 'тротуар' in text or 'пешеходн переход' in text or ('проезд' in text) or ('набережн' in text) or ('бульвар' in text) or ('велодорож' in text):
        return 'street_network'
    if 'дорог' in text or 'остановк' in text or 'вокзал' in text or ('станц' in text) or ('заправочн' in text):
        return 'transport'
    if 'досугов' in text or 'парк развлечен' in text or 'зоопарк' in text:
        return 'culture'
    if 'памятник' in text or 'достопримечатель' in text:
        return 'culture'
    if 'склад' in text or 'промышлен' in text:
        return 'industrial'
    return None

SERVICE_PROFILE_ALIASES: dict[str, set[str]] = defaultdict(set)

PHYSICAL_PROFILE_ALIASES: dict[str, set[str]] = defaultdict(set)

PROFILE_REGEX_RULES: list[tuple[str, str, str]] = [('communal_engineering', 'strong', 'трансформаторн\\s+подстанц|электрическ\\s+подстанц|\\bподстанц\\b|насосн\\s+станц|станц\\s+подкачк|подкачк\\s+воды|канализационн\\s+насосн\\s+станц|водозабор|водонапорн\\s+башн|котельн|лэп|линия\\s+электропередач|электроснабж|объект\\s+электроснабж|газопровод|газорегуляторн\\s+пункт|\\bгрп\\b|\\bгру\\b|водопровод|очистн\\s+сооружен|сеть\\s+теплоснабж|сеть\\s+водоотведен|сеть\\s+электросвяз|электроснабж|газоснабж|водоснабж|теплоснабж|\\bнасосн\\b|бойлерн|теплоэлектроцентрал'), ('street_network', 'strong', 'уличн\\s+дорожн|тротуар|пешеходн\\s+переход|проезд|набережн|бульвар|велодорож|автомобильн\\s+дорог|уличн\\s+сеть|\\bдорог\\b'), ('household_outbuilding', 'strong', 'подсобн\\s+сооруж|вспомогательн\\s+сооруж|хозяйственн\\s+постро|надворн\\s+постро'), ('education_school', 'strong', 'школ|лице|гимнази'), ('education_preschool', 'strong', 'детск\\s+сад|ясл'), ('education_professional', 'strong', 'техникум|колледж|училищ|вуз|университет|институт|высш\\s+профессиональн|средн\\s+профессиональн'), ('education', 'medium', 'образован|просвещен'), ('public_admin', 'strong', 'налогов|администрац|административн\\s+здан|административн(?![а-я]*\\s+правонаруш)|мфц|загс|суд|прокуратур|полиц|отдел\\s+полици|ведомств|инспекц|государственн|муниципальн'), ('defense_security', 'strong', 'воинск|военн|казарм|штаб|гарнизон|полигон|оборон|министерств\\s+обороны|част[ьи]\\s*№'), ('social_service', 'medium', 'социальн|дом\\s+престарел|центр\\s+занятост|детск\\s+дом'), ('healthcare', 'strong', 'поликлиник|больниц|роддом|фельдшер|здравпункт|стоматолог|аптек|женск\\s+консультац|скора[яй]\\s+медицинск|инфекцион|лечебн\\s+корпус'), ('sport', 'strong', 'спорт|стадион|бассейн|каток|ледов\\s+арен|скалодром'), ('culture', 'strong', 'музе|музей\\s*заповедник|заповедник|дворцово[-\\s]*парков|библиотек|театр|кинотеатр|дом\\s+культур|дворец\\s+культур|цирк|зоопарк|памятник|достопримечатель|объект\\s+культурн\\s+наслед'), ('public_catering', 'strong', 'кафе|ресторан|столов|бар|булочн'), ('retail', 'strong', 'магазин|супермаркет|рынок|торгов|\\bкиоск\\b'), ('business', 'medium', 'делов|офис|банк|страхов'), ('garage', 'strong', 'гараж|гаражи|автостоянк|стоянк\\b'), ('public_land', 'strong', 'общ\\s+пользован|земельн\\s+участк\\s+территор'), ('gardening', 'strong', 'садоводств|огородничеств|личн\\s+подсобн\\s+хозяйств|\\bогородн\\b'), ('residential_individual', 'strong', 'индивидуальн\\s+жилищн\\s+строительств|индивидуальн\\s+жил|индивидуальн\\s+жил\\s+дом|одноэтажн\\s+жил\\s+дом|жилой\\s+дом\\s+одноэтажн|индивидуальн\\s+застройк'), ('residential_blocked', 'strong', 'блокированн'), ('residential_low', 'strong', 'малоэтажн\\s+многоквартирн|малоэтажн\\s+жил'), ('residential_mid', 'strong', 'среднеэтажн\\s+жил|среднеэтажн\\s+застройк'), ('residential_high', 'strong', 'многоэтажн\\s+жил|многоквартирн\\s+многоэтажн|высотн\\s+жил'), ('residential_multifamily', 'strong', 'многоквартирн|многквартирн|могоквартирн|\\bмкд\\b'), ('residential_unspecified', 'weak', 'жил[а-я\\s-]*дом|жил[а-я\\s-]*застройк'), ('industrial', 'medium', 'производств|цех|завод|склад|логистик|промышлен|мастерск'), ('hotel', 'strong', 'гостиниц|гостев\\s+дом|хостел|мотел'),
    ('recreation', 'medium', 'рекреацион'),
    ('public_safety', 'strong', 'пожарн\\s+депо|пожарн\\s+часть|пожарн\\s+охран|\\bмчс\\b'),
    ('transport', 'medium', 'железнодорож|вокзал|автозаправочн|парковк|остановк|транспорт')]

PROFILE_TO_VRI_PATTERNS: dict[str, list[tuple[str, int]]] = {'communal_engineering': [('коммунальн\\s+обслуживан|предоставлен\\s+коммунальн\\s+услуг|водозабор|очистн|насосн|котельн|трансформаторн|подстанц|линия\\s+электропередач|газопровод|газорегуляторн\\s+пункт|\\bгрп\\b|\\bгру\\b|водопровод|электричеств|газа|воды|тепла|электроснабж|объект\\s+электроснабж|подкачк', 100)], 'street_network': [('общ\\s+пользован|уличн\\s+дорожн|автомобильн\\s+дорог|пешеходн\\s+тротуар|пешеходн\\s+переход|набережн|берегов[а-я]*\\s+полос|бульвар|площад|проезд|велодорож', 100)], 'household_outbuilding': [('подсобн\\s+сооруж|вспомогательн\\s+сооруж|хозяйственн\\s+постро|надворн\\s+постро|гараж', 100)], 'education_preschool': [('дошкольн|образован|просвещен', 95)], 'education_school': [('начальн|средн\\s+общ|образован|просвещен', 95)], 'education_professional': [('профессиональн|высш|образован|просвещен', 95)], 'education': [('образован|просвещен', 90)], 'public_admin': [('обществен\\s+управлен', 100), ('делов\\s+управлен', 80), ('административн\\s+здан|административн', 90)], 'defense_security': [('обеспечен\\s+обороны\\s+и\\s+безопасности|военн|воинск|казарм|штаб|гарнизон|полигон|оборон', 100)], 'social_service': [('социальн\\s+обслуживан', 100)], 'healthcare': [('амбулаторн|поликлинич|медицин|стационар|здравоохран', 95)], 'sport': [('спорт', 100)], 'culture': [('культурн\\s+развити|культур|музе|театр|библиотек|цирк|зоопарк', 95)], 'public_catering': [('обществен\\s+питан', 100)], 'retail': [('магазин|торгов', 95), ('объект\\s+торгов', 90)], 'business': [('делов\\s+управлен', 100), ('банк|страхов', 95)], 'garage': [('обслуживан\\s+автотранспорт', 90), ('гараж', 85)], 'public_land': [('общ\\s+пользован|уличн\\s+дорожн|пешеходн\\s+тротуар|пешеходн\\s+переход|проезд|бульвар|набережн|автомобильн\\s+дорог', 100)], 'gardening': [('садоводств|огородничеств|личн\\s+подсобн\\s+хозяйств', 100)], 'residential_individual': [('индивидуальн\\s+жилищн\\s+строительств', 100), ('индивидуальн\\s+жил\\s+дом', 95)], 'residential_blocked': [('блокированн', 100)], 'residential_low': [('малоэтажн\\s+многоквартирн', 100)], 'residential_mid': [('среднеэтажн\\s+жил', 100)], 'residential_high': [('многоэтажн\\s+жил', 100)], 'residential_multifamily': [('среднеэтажн\\s+жил', 100), ('многоэтажн\\s+жил', 100), ('многоквартирн(?!\\s*(?:жил|застройк)?\\s*$)', 95), ('жилая\\s+застройка', 80), ('малоэтажн\\s+многоквартирн', 60)], 'residential_unspecified': [('жил', 60)], 'industrial': [('производств|склад|промышлен', 95)], 'transport': [('железнодорожн|автомобильн\\s+транспорт|обслуживан\\s+автотранспорт|транспорт', 95)], 'hotel': [('гостиничн\\s+обслуживан|гостиниц|средств\\s+размещен|санаторн\\s+деятельност', 100)], 'recreation': [('рекреаци|отдых|природн|зеленн|парк', 95)], 'public_safety': [('пожарн|обеспечен\\s+безопасн|охран\\s+порядк', 90), ('коммунальн\\s+обслуживан', 60)]}

def classify_object_profile(vri_text: Any) -> dict[str, Any]:
    """Classify raw cadastral VRI into a transferable functional profile."""
    raw_text = normalize_text(vri_text)
    canon = canonicalize_vri_name(raw_text)
    if not canon:
        return {'family': 'unknown', 'confidence': 'none', 'source': 'empty', 'residential_subtype': None, 'is_residential': False, 'allow_residential_auto': False, 'candidate_keys': [], 'canon': ''}
    scores: Counter[str] = Counter()
    source = 'unknown'
    for family, weight_label, pattern in PROFILE_REGEX_RULES:
        if re.search(pattern, canon):
            weight = 100 if weight_label == 'strong' else 70 if weight_label == 'medium' else 35
            scores[family] += weight
            source = 'regex'
    for family, alias_set in SERVICE_PROFILE_ALIASES.items():
        for alias in alias_set:
            if alias and (alias in canon or is_exactish_vri_match(canon, alias)):
                scores[family] += 40
                source = 'hierarchy'
    for family, alias_set in PHYSICAL_PROFILE_ALIASES.items():
        for alias in alias_set:
            if alias and (alias in canon or is_exactish_vri_match(canon, alias)):
                scores[family] += 40
                source = 'hierarchy'
    if not scores:
        family = 'unknown'
        confidence = 'weak'
    else:
        family, best_score = scores.most_common(1)[0]
        confidence = 'strong' if best_score >= 100 else 'medium' if best_score >= 60 else 'weak'
    residential_subtype = None
    is_residential = family.startswith('residential_') or family == 'residential'
    if family.startswith('residential_'):
        residential_subtype = family.split('residential_', 1)[1]
    elif family == 'residential':
        residential_subtype = None
    candidate_keys: list[str] = []
    if family != 'unknown':
        candidate_keys.append(family)
        if family in {'education_preschool', 'education_school', 'education_professional'}:
            candidate_keys.append('education')
        if family.startswith('residential_'):
            candidate_keys.append('residential_unspecified')
        if family == 'residential_multifamily':
            candidate_keys.extend(['residential_mid', 'residential_high', 'residential_low'])
        if family == 'public_admin':
            candidate_keys.append('business')
        if family == 'defense_security':
            candidate_keys.append('public_admin')
        if family == 'street_network':
            candidate_keys.extend(['public_land', 'transport'])
        if family == 'household_outbuilding':
            candidate_keys.extend(['garage', 'residential_individual', 'residential_blocked', 'residential_low', 'residential_unspecified'])
        if family == 'public_safety':
            candidate_keys.extend(['public_admin', 'communal_engineering'])
        if family == 'hotel':
            candidate_keys.append('business')
        if family == 'recreation':
            candidate_keys.append('culture')
    return {'family': family, 'confidence': confidence, 'source': source, 'residential_subtype': residential_subtype, 'is_residential': is_residential, 'allow_residential_auto': bool(not RESIDENTIAL_AUTO_REQUIRES_SUBTYPE or residential_subtype not in {None, 'unspecified'}), 'candidate_keys': candidate_keys, 'canon': canon}

def rank_zone_item_for_profile(item: pd.Series, profile: dict[str, Any]) -> int:
    """Score how well one zone-item matches the profiled functional family."""
    family = normalize_text(profile.get('family'))
    candidate_keys = profile.get('candidate_keys') or ([] if not family else [family])
    if not candidate_keys or family == 'unknown':
        return 0
    text_parts = [normalize_text(item.get('catalog_vri_name')), normalize_text(item.get('catalog_vri_description')), normalize_text(item.get('catalog_original_vri_name')), normalize_text(item.get('catalog_original_vri_description'))]
    combined = canonicalize_vri_name(' | '.join([part for part in text_parts if part]))
    best_rank = 0
    for key in candidate_keys:
        for pattern, rank in PROFILE_TO_VRI_PATTERNS.get(key, []):
            if re.search(pattern, combined):
                best_rank = max(best_rank, rank)
    return best_rank

def filter_zone_items_for_profile(zone_items: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    """Restrict zone VRI candidates to the functional family when the profile is strong enough."""
    if zone_items is None or zone_items.empty:
        return zone_items
    if not ENABLE_PROFILED_FAST_MATCH:
        return zone_items
    family = normalize_text(profile.get('family'))
    confidence = normalize_text(profile.get('confidence'))
    if family in {'', 'unknown'} or confidence == 'weak':
        return zone_items
    work = zone_items.copy()
    work['__profile_rank__'] = work.apply(lambda row: rank_zone_item_for_profile(row, profile), axis=1)
    filtered = work.loc[work['__profile_rank__'] > 0].copy()
    if filtered.empty:
        return zone_items
    return filtered.sort_values(['__profile_rank__', 'section_name', 'catalog_vri_name'], ascending=[False, False, True]).reset_index(drop=True)

def profile_allows_auto_decision(profile: dict[str, Any], candidate_name: Any) -> bool:
    """Decide whether deterministic auto-assignment is safe for the current profile."""
    candidate_canon = canonicalize_vri_name(candidate_name)
    if not candidate_canon:
        return True
    if profile.get('is_residential') and (not profile.get('allow_residential_auto')):
        residential_markers = ['индивидуальн жилищн строительств', 'блокированн', 'малоэтажн', 'среднеэтажн', 'многоэтажн', 'жил']
        if any((marker in candidate_canon for marker in residential_markers)):
            return False
    return True

EXPLICIT_DESCRIPTION_COVERAGE_RULES: dict[str, dict[str, Any]] = {'communal_engineering': {'query_patterns': ['трансформаторн|подстанц|лэп|линия\\s+электропередач|газопровод|газорегуляторн\\s+пункт|\\bгрп\\b|\\bгру\\b|водопровод|котельн|очистн|насосн|станц\\s+подкачк|подкачк\\s+воды|электроснабж|объект\\s+электроснабж|газоснабж|водоснабж|теплоснабж'], 'candidate_patterns': ['коммунальн\\s+обслуживан|предоставлен\\s+коммунальн\\s+услуг|водозабор|очистн|насосн|котельн|трансформаторн|подстанц|линия\\s+электропередач|газопровод|газорегуляторн\\s+пункт|водопровод|электричеств|газа|воды|тепла|электроснабж|газоснабж|водоснабж|теплоснабж|подкачк'], 'preferred_codes': {'3.1', '3.1.1', '3.1.2'}}, 'street_network': {'query_patterns': ['уличн\\s+дорожн|тротуар|пешеходн\\s+переход|проезд|набережн|бульвар|велодорож|автомобильн\\s+дорог'], 'candidate_patterns': ['земельн\\s+участк.*общ\\s+пользован|общ\\s+пользован|уличн\\s+дорожн|автомобильн\\s+дорог|пешеходн\\s+тротуар|пешеходн\\s+переход|набережн|берегов[а-я]*\\s+полос|бульвар|площад|проезд|велодорож'], 'preferred_codes': {'12.0', '12.0.1', '7.2', '7.2.1'}}, 'household_outbuilding': {'query_patterns': ['подсобн\\s+сооруж|вспомогательн\\s+сооруж|хозяйственн\\s+постро|надворн\\s+постро'], 'candidate_patterns': ['подсобн\\s+сооруж|вспомогательн\\s+сооруж|хозяйственн\\s+постро|надворн\\s+постро|гараж'], 'preferred_codes': {'2.1', '2.2', '2.3', '2.1.1', '13.2', '2.7.2'}}}

def build_rosreestr_classifier_children_map(classifier_by_code: Optional[dict[str, dict[str, Any]]]) -> dict[str, set[str]]:
    """Build parent -> direct children map from the Rosreestr classifier."""
    children_map: dict[str, set[str]] = defaultdict(set)
    for code, entry in (classifier_by_code or {}).items():
        parent_code = normalize_text((entry or {}).get('parent_code'))
        if parent_code and is_valid_vri_code(code):
            children_map[parent_code].add(code)
    return dict(children_map)


def expand_vri_codes_with_classifier_children(codes: Any, *, classifier_children_map: Optional[dict[str, set[str]]]=None) -> set[str]:
    """Expand a set of VRI codes with their direct Rosreestr classifier children."""
    classifier_children_map = classifier_children_map or {}
    expanded: set[str] = set()
    queue: list[str] = [normalize_text(code) for code in codes or set() if normalize_text(code)]
    while queue:
        code = queue.pop()
        if code in expanded:
            continue
        expanded.add(code)
        for child_code in classifier_children_map.get(code, set()):
            child_norm = normalize_text(child_code)
            if child_norm and child_norm not in expanded:
                queue.append(child_norm)
    return expanded

def try_explicit_description_coverage_in_zone(vri_text: Any, actual_zone_code: Any, context: Any=None) -> Optional[dict[str, Any]]:
    """Apply narrow domain-specific description coverage rules before LLM."""
    zone_code = normalize_text(actual_zone_code)
    if not zone_code:
        return None
    zone_items_map = context.zone_items_lookup if context is not None else {}
    zone_items = zone_items_map.get(zone_code)
    if zone_items is None or zone_items.empty:
        return None
    profile = classify_object_profile(vri_text)
    family = normalize_text(profile.get('family'))
    rule = EXPLICIT_DESCRIPTION_COVERAGE_RULES.get(family)
    if not rule:
        return None
    query_norm = canonicalize_vri_name(vri_text)
    if not query_norm:
        return None
    if not any((re.search(pattern, query_norm) for pattern in rule.get('query_patterns', []))):
        return None
    preferred_codes = expand_vri_codes_with_classifier_children({normalize_text(code) for code in rule.get('preferred_codes', set())}, classifier_children_map=context.rosreestr_classifier_children_map if context is not None else {})
    candidates: list[dict[str, Any]] = []
    for _, item in zone_items.iterrows():
        section_name = normalize_text(item.get('section_name'))
        verdict = SECTION_TO_VERDICT.get(section_name, 'unclear')
        matched_vri_name = normalize_text(item.get('catalog_vri_name'))
        matched_vri_code = normalize_text(item.get('catalog_vri_code'))
        candidate_parts = [normalize_text(item.get('catalog_vri_name')), normalize_text(item.get('catalog_vri_description')), normalize_text(item.get('catalog_original_vri_name')), normalize_text(item.get('catalog_original_vri_description'))]
        candidate_text = canonicalize_vri_name(' '.join([part for part in candidate_parts if part]))
        if not candidate_text:
            continue
        if not any((re.search(pattern, candidate_text) for pattern in rule.get('candidate_patterns', []))):
            continue
        score = SECTION_PRIORITY.get(section_name, 0) * 100
        if matched_vri_code in preferred_codes:
            score += 25
        if any((re.search(pattern, candidate_text) for pattern in rule.get('query_patterns', []))):
            score += 10
        candidates.append({'score': score, 'section_name': section_name, 'verdict': verdict, 'matched_vri_name': matched_vri_name, 'matched_vri_code': matched_vri_code})
    if not candidates:
        return None
    best = sorted(candidates, key=lambda item: (item['score'], SECTION_PRIORITY.get(item['section_name'], 0), item['matched_vri_name']), reverse=True)[0]
    reasons = {'communal_engineering': 'Кадастровая формулировка покрывается описанием коммунального обслуживания в фактической зоне.', 'street_network': 'Кадастровая формулировка покрывается описанием ВРИ общего пользования / улично-дорожной сети в фактической зоне.', 'household_outbuilding': 'Кадастровая формулировка покрывается описанием вспомогательных / подсобных сооружений в фактической зоне.'}
    return {'verdict': best['verdict'], 'matched_vri_name': best['matched_vri_name'], 'matched_vri_code': best['matched_vri_code'], 'reason': f"{reasons.get(family, 'Кадастровая формулировка покрывается описанием разрешенного ВРИ в фактической зоне.')} Опорный ВРИ: «{best['matched_vri_name']}» (код {best['matched_vri_code']})."}

def render_profile_hint(profile: dict[str, Any]) -> str:
    """Render profile information for debugging and LLM prompts."""
    family = normalize_text(profile.get('family')) or 'unknown'
    confidence = normalize_text(profile.get('confidence')) or 'unknown'
    residential_subtype = normalize_text(profile.get('residential_subtype')) or '-'
    source = normalize_text(profile.get('source')) or 'unknown'
    return f'family={family}; confidence={confidence}; residential_subtype={residential_subtype}; source={source}'

INTENT_SYNONYMS: dict[str, list[str]] = {'historical_park': ['историческ(?:ий|ого|ая|ие|их)?\\s+парк(?:и|ов)?\\b', 'дворцово?[-\\s]?парков', 'музей[-\\s]?заповедник', 'историко[-\\s]?культурн(?:ая|ой)\\s+деятельност', 'объект(?:ы|ов)?\\s+культурн(?:ого|ых)?\\s+наслед'], 'recreation_public': ['территор(?:ия|ии)\\s+общ\\s+пользован', 'парк(?:и|ов)?\\b', 'сквер', 'рекреаци'], 'industrial': ['промышлен', 'производств', 'склад'], 'residential_individual': ['ижс', 'индивидуальн(?:ое|ой)\\s+жилищн', 'индивидуальн(?:ая|ой)\\s+жил'], 'residential_blocked': ['блокированн'], 'residential_low': ['малоэтажн'], 'residential_mid': ['среднеэтажн'], 'residential_high': ['многоэтажн', 'высотн'], 'residential_unspecified': ['жил(?:ая|ой|ого)?', 'жил(?:ая|ой)?\\s+застройк']}

PROFILE_TO_INTENT: dict[str, str] = {'industrial': 'industrial', 'public_land': 'recreation_public', 'culture': 'historical_park'}

BRIDGE_BLOCK_PATTERNS = ['без\\s+вновь\\s+возводим(?:ых|ого)?\\s+окс', 'без\\s+размещени[яй]\\s+нов(?:ых|ого)\\s+объект', 'запреща(?:ет|ется)\\s+нов(?:ое|ые)\\s+строительств']

BRIDGE_INTENT_EQUIVALENCE_GROUPS: list[set[str]] = [{'historical_park', 'recreation_public'}]

BRIDGE_EQUIVALENCE_HINT_PATTERNS = ['историческ', 'культурн', 'парк', 'сквер', 'рекреаци', 'общ\\s+пользован']

def detect_zone_intent(zone_name: Any, zone_heading: Any, zone_summary: Any, base_zone_code: Any) -> str:
    """Detect high-level zone functional intent from naming fields."""
    combined = canonicalize_vri_name(' '.join([normalize_text(zone_name), normalize_text(zone_heading), normalize_text(zone_summary)]))
    if not combined:
        return 'unknown'
    if re.search('историческ.*парк', combined):
        return 'historical_park'
    for intent, patterns in INTENT_SYNONYMS.items():
        if any((re.search(pattern, combined) for pattern in patterns)):
            return intent
    return 'unknown'

def detect_object_intent(vri_text: Any, profile: dict[str, Any]) -> str:
    """Detect high-level object intent from profile + synonym dictionary."""
    family = normalize_text(profile.get('family'))
    if family.startswith('residential_'):
        return family
    if family == 'residential':
        return 'residential_unspecified'
    if family in PROFILE_TO_INTENT:
        return PROFILE_TO_INTENT[family]
    canon = canonicalize_vri_name(vri_text)
    for intent, patterns in INTENT_SYNONYMS.items():
        if any((re.search(pattern, canon) for pattern in patterns)):
            return intent
    return 'unknown'

def find_bridge_evidence_phrase(intent: str, main_vri_full: Any, retrieval_text: Any) -> Optional[str]:
    """Return short supporting phrase from zone texts, otherwise None."""
    for part in [normalize_text(main_vri_full), normalize_text(retrieval_text)]:
        if not part:
            continue
        for raw_phrase in re.split('[\\n.;]', part):
            phrase = normalize_text(raw_phrase)
            if len(phrase) < 10:
                continue
            canon_phrase = canonicalize_vri_name(phrase)
            if any((re.search(pattern, canon_phrase) for pattern in INTENT_SYNONYMS.get(intent, []))):
                return phrase[:220]
    return None

def has_bridge_conflict(main_vri_full: Any, retrieval_text: Any) -> bool:
    """Check textual prohibitions that should block semantic-bridge auto decision."""
    combined = canonicalize_vri_name(' '.join([normalize_text(main_vri_full), normalize_text(retrieval_text)]))
    return any((re.search(pattern, combined) for pattern in BRIDGE_BLOCK_PATTERNS))

def intents_match_for_bridge(zone_intent: str, object_intent: str, *, main_vri_full: Any, retrieval_text: Any) -> bool:
    """Check strict or explicitly-equivalent intent match for bridge auto decision."""
    if zone_intent == object_intent:
        return True
    for group in BRIDGE_INTENT_EQUIVALENCE_GROUPS:
        if zone_intent in group and object_intent in group:
            combined = canonicalize_vri_name(' '.join([normalize_text(main_vri_full), normalize_text(retrieval_text)]))
            if any((re.search(pattern, combined) for pattern in BRIDGE_EQUIVALENCE_HINT_PATTERNS)):
                return True
    return False

def try_semantic_bridge_autodecision(*, vri_text: Any, profile: dict[str, Any], zone_ref: dict[str, Any], actual_zone_name: Any, context: Any=None) -> Optional[dict[str, Any]]:
    """Return semantic-bridge auto decision only under strict safety conditions."""
    raw_zone_lookup_map = context.raw_zone_lookup if context is not None else {}
    zone_template = raw_zone_lookup_map.get(zone_ref['zone_code'], {})
    zone_name = normalize_text(zone_template.get('zone_name') or actual_zone_name)
    zone_heading = normalize_text(zone_template.get('zone_heading') or zone_name)
    zone_summary = normalize_text(zone_template.get('zone_summary'))
    base_zone_code = normalize_text(zone_template.get('base_zone_code'))
    retrieval_text = normalize_text(zone_template.get('retrieval_text'))
    main_vri_full = normalize_text(zone_template.get('main_vri_full'))
    zone_intent = detect_zone_intent(zone_name, zone_heading, zone_summary, base_zone_code)
    object_intent = detect_object_intent(vri_text, profile)
    if zone_intent == 'unknown' or object_intent == 'unknown':
        return None
    if not intents_match_for_bridge(zone_intent, object_intent, main_vri_full=main_vri_full, retrieval_text=retrieval_text):
        return None
    evidence_phrase = find_bridge_evidence_phrase(object_intent, main_vri_full, retrieval_text)
    if not evidence_phrase or has_bridge_conflict(main_vri_full, retrieval_text):
        return None
    return {'verdict': 'allowed_main', 'matched_vri_name': evidence_phrase, 'matched_vri_code': pd.NA, 'reason': f'Решение принято по semantic bridge-правилу: zone_intent={zone_intent}; object_intent={object_intent}; подтверждение: «{evidence_phrase}».', 'zone_intent': zone_intent, 'object_intent': object_intent}

