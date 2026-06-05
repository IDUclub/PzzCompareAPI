from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .runtime_settings import ENABLE_EMBED_FAST_MATCH as _ENABLE_EMBED_FAST_MATCH, ENABLE_LLM as _ENABLE_LLM
from .classification_layer import ensure_classification_columns
from .clients import vectorizer
from .data_loading import InputDataLoader, ReferenceDataProvider
from .matching_layer import (
    choose_best_exact_match,
    fast_embed_match_in_zone,
    fast_string_match_in_zone,
    find_exact_zone_candidates,
    is_residential_unspecified_vri,
    resolve_residential_unspecified_in_zone,
    resolve_zone_reference,
    run_zone_check_with_llm,
)
from .postprocess_layer import (
    attach_actual_zone_name_column,
    mark_manual_review_for_allowed_top_candidates,
    select_and_rename_result_columns,
)
from .rerank_layer import (
    attach_not_allowed_llm_rerank_column,
    build_classifier_embedding_items,
    build_zone_check_prompt,
    build_zone_section_code_cache,
    build_not_allowed_embed_query_text,
    build_not_allowed_same_zone_candidates,
    fast_rerank_not_allowed_candidates,
    get_not_allowed_query_key,
    promote_generic_residential_first,
    run_not_allowed_rerank_with_llm,
    serialize_not_allowed_same_zone_candidates,
    should_run_not_allowed_llm_rerank,
    NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N,
)
from .spatial_layer import build_source_with_spatial_attributes
from .text_utils import (
    build_actual_zone_key,
    build_rosreestr_classifier_maps,
    flatten_zone_catalog,
    normalize_llm_verdict,
    normalize_text,
    sanitize_zone_catalog,
    status_to_russian_label,
)
from .profiled_fast_match_layer import should_use_deeper_llm_reasoning, build_rosreestr_classifier_children_map
from .clients import llm_client
from .runtime_context import PipelineRuntimeContext

logger = logging.getLogger("pipeline.runtime")

_PIPELINE_LLM_WORKERS = max(1, int(os.getenv("PIPELINE_LLM_WORKERS", "4")))


def _log_stage(stage: str, status: str, **extra: Any) -> None:
    payload = {"stage": stage, "status": status, **extra}
    details = " | ".join((f"{key}={value}" for key, value in payload.items()))
    logger.info("PIPELINE_STAGE | %s", details)


def _init_runtime_context(zone_templates: list[dict[str, Any]], rosreestr_classifier: Any) -> PipelineRuntimeContext:
    """Initialize global notebook-era runtime objects used across business layers."""
    classifier_by_code, _ = build_rosreestr_classifier_maps(rosreestr_classifier)
    sanitized_templates, _ = sanitize_zone_catalog(zone_templates, classifier_by_code=classifier_by_code)

    pzz_ref_df, pzz_vri_items_df = flatten_zone_catalog(sanitized_templates)
    pzz_vri_items_df = pzz_vri_items_df.copy()
    pzz_vri_items_df["catalog_vri_name_match"] = pzz_vri_items_df["catalog_vri_name_norm"].map(normalize_text)
    pzz_vri_items_df["catalog_vri_description_match"] = pzz_vri_items_df["catalog_vri_description"].map(normalize_text)

    zone_lookup = {normalize_text(item.get("zone_code")): item for item in sanitized_templates if normalize_text(item.get("zone_code"))}
    base_zone_lookup: dict[str, list[dict[str, Any]]] = {}
    for item in sanitized_templates:
        base_code = normalize_text(item.get("base_zone_code"))
        if base_code:
            base_zone_lookup.setdefault(base_code, []).append(item)

    zone_items_lookup: dict[str, pd.DataFrame] = {
        zone_code: pzz_vri_items_df.loc[pzz_vri_items_df["zone_code"] == zone_code].reset_index(drop=True)
        for zone_code in pzz_ref_df["zone_code"].dropna().unique().tolist()
    }
    zone_fast_text_lookup = {
        normalize_text(item.get("zone_code")): {
            "zone_name_match": normalize_text(item.get("zone_name")),
            "zone_summary_match": normalize_text(item.get("zone_summary")),
            "retrieval_text_short_match": normalize_text(item.get("retrieval_text_short")),
        }
        for item in sanitized_templates
    }

    if pzz_ref_df.empty:
        zone_vectorizer = None
        zone_matrix = None
    else:
        zone_vectorizer = TfidfVectorizer(min_df=1)
        zone_matrix = zone_vectorizer.fit_transform(pzz_ref_df["zone_search_text"].fillna("").tolist())

    zone_item_embeddings: dict[str, np.ndarray] = {}
    if _ENABLE_EMBED_FAST_MATCH:
        for zone_code, zone_items in zone_items_lookup.items():
            if zone_items.empty:
                zone_item_embeddings[zone_code] = np.zeros((0, 0), dtype=np.float32)
                continue
            embed_texts = zone_items["item_search_text"].fillna("").tolist()
            embeddings = vectorizer.embed_many(embed_texts, batch_size=64)
            zone_item_embeddings[zone_code] = embeddings

    classifier_embed_items = build_classifier_embedding_items(classifier_by_code)
    classifier_embed_vectors = np.zeros((0, 0), dtype=np.float32)
    if _ENABLE_EMBED_FAST_MATCH and (not classifier_embed_items.empty):
        classifier_embed_vectors = vectorizer.embed_many(
            classifier_embed_items["classifier_embed_text"].fillna("").tolist(),
            batch_size=64,
        )

    classifier_children_map = build_rosreestr_classifier_children_map(classifier_by_code)
    zone_section_cache = build_zone_section_code_cache(
        zone_items_lookup_map=zone_items_lookup,
        classifier_children_map=classifier_children_map,
    )

    return PipelineRuntimeContext(
        zone_lookup=zone_lookup,
        base_zone_lookup=base_zone_lookup,
        raw_zone_lookup=zone_lookup,
        zone_items_lookup=zone_items_lookup,
        zone_fast_text_lookup=zone_fast_text_lookup,
        pzz_ref_df=pzz_ref_df,
        pzz_vri_items_df=pzz_vri_items_df,
        zone_vectorizer=zone_vectorizer,
        zone_matrix=zone_matrix,
        zone_item_embeddings=zone_item_embeddings,
        rosreestr_classifier_by_code=classifier_by_code,
        rosreestr_classifier_children_map=classifier_children_map,
        zone_section_code_cache=zone_section_cache,
        classifier_embed_items_df=classifier_embed_items,
        classifier_embed_vectors=classifier_embed_vectors,
        vectorizer=vectorizer,
        llm_client=llm_client,
    )


def _prefill_query_vectors(
    unique_texts: list[str],
    context: PipelineRuntimeContext,
) -> None:
    """Embed all unique VRI texts in one batch and cache the vectors.

    This pre-populates ``context.not_allowed_query_vector_cache`` so that
    subsequent per-row calls to ``build_not_allowed_same_zone_candidates``
    and ``fast_embed_match_in_zone`` skip the embed call entirely.
    """
    if not unique_texts or context.vectorizer is None:
        return

    missing = [t for t in unique_texts if get_not_allowed_query_key(t) not in context.not_allowed_query_vector_cache]
    if not missing:
        return

    embed_queries = [build_not_allowed_embed_query_text(t) for t in missing]
    vecs = context.vectorizer.embed_many(embed_queries, batch_size=64)
    for text, vec in zip(missing, vecs):
        context.not_allowed_query_vector_cache[get_not_allowed_query_key(text)] = vec


def run_pipeline(
    pzz_codes_path: str,
    cadastral_geojson_path: str,
    pzz_zones_geojson_path: str,
    pzz_zone_vri_labels_path: str,
    vri_classifier_path: str,
    include_pzz_check: bool,
    cadastral_vri_col: str,
    pzz_zone_code_col: str,
    pzz_zone_name_col: str,
    output_geojson_path: str,
    unique_results_xlsx_path: str,
    unique_results_json_path: str,
    base_url: str,
    embed_model: str,
    generate_model: str,
    top_k: int,
    batch_size: int,
) -> None:
    """Service-safe orchestrator that mirrors spatial-first notebook logic."""
    _ = (pzz_codes_path, base_url, embed_model, generate_model, top_k, batch_size)
    total_started = perf_counter()
    _log_stage("pipeline", "start", include_pzz_check=include_pzz_check, batch_size=batch_size)

    references = ReferenceDataProvider().resolve_paths(
        pzz_zone_labels_override_path=pzz_zone_vri_labels_path,
        vri_classifier_override_path=vri_classifier_path,
    )

    load_started = perf_counter()
    source_gdf = InputDataLoader.load_geojson_to_gdf(cadastral_geojson_path)
    rosreestr_classifier = ReferenceDataProvider.load_json(references.vri_classifier_path)

    if include_pzz_check:
        pzz_zones_gdf = InputDataLoader.load_geojson_to_gdf(pzz_zones_geojson_path)
        zone_templates = ReferenceDataProvider.load_json(references.pzz_zone_labels_path)
        _log_stage("load_inputs", "finished", duration_ms=int((perf_counter() - load_started) * 1000), source_rows=len(source_gdf), pzz_rows=len(pzz_zones_gdf), zone_templates=len(zone_templates))
    else:
        zone_templates = []  # not needed for classification-only mode
        _log_stage("load_inputs", "finished", duration_ms=int((perf_counter() - load_started) * 1000), source_rows=len(source_gdf), pzz_rows=0, zone_templates=0)

    context_started = perf_counter()
    context = _init_runtime_context(zone_templates=zone_templates, rosreestr_classifier=rosreestr_classifier)
    _log_stage("init_runtime_context", "finished", duration_ms=int((perf_counter() - context_started) * 1000), zone_lookup=len(context.zone_lookup), zone_embeddings=len(context.zone_item_embeddings), classifier_embed_rows=len(context.classifier_embed_items_df))

    if include_pzz_check:
        spatial_started = perf_counter()
        source_with_spatial_gdf = build_source_with_spatial_attributes(
            source_gdf=source_gdf,
            pzz_zones_gdf=pzz_zones_gdf,
            vri_col=cadastral_vri_col,
            pzz_zone_code_col=pzz_zone_code_col,
            pzz_zone_name_col=pzz_zone_name_col,
        )
        _log_stage("spatial_join", "finished", duration_ms=int((perf_counter() - spatial_started) * 1000), rows=len(source_with_spatial_gdf))
    else:
        source_with_spatial_gdf = source_gdf.copy()
        source_with_spatial_gdf["__comparison_key__"] = (
            source_with_spatial_gdf[cadastral_vri_col].fillna("").astype(str)
        )
        source_with_spatial_gdf["PZZ_ACTUAL_CODE"] = ""
        _log_stage("spatial_join", "skipped", reason="include_pzz_check=False")

    unique_df = source_with_spatial_gdf[["__comparison_key__", cadastral_vri_col, "PZZ_ACTUAL_CODE"]].drop_duplicates(
        subset=["__comparison_key__"]
    )

    embed_prefill_started = perf_counter()
    unique_vri_texts = [
        normalize_text(t)
        for t in unique_df[cadastral_vri_col].dropna().unique().tolist()
        if normalize_text(t)
    ]
    _prefill_query_vectors(unique_vri_texts, context)
    _log_stage("embed_prefill", "finished",
               duration_ms=int((perf_counter() - embed_prefill_started) * 1000),
               unique_texts=len(unique_vri_texts))

    llm_zone_check_cache: dict[str, dict[str, Any]] = {}
    llm_rerank_cache: dict[str, list[dict[str, Any]]] = {}
    _llm_cache_lock = threading.Lock()

    def _process_row(row: pd.Series) -> dict[str, Any]:
        vri_text = normalize_text(row.get(cadastral_vri_col))
        actual_zone_code = normalize_text(row.get("PZZ_ACTUAL_CODE"))

        payload: dict[str, Any] = {
            "__comparison_key__": row.get("__comparison_key__"),
            cadastral_vri_col: row.get(cadastral_vri_col),
            "CHECK_SCOPE": "actual_zone",
            "MATCH_METHOD": "heuristic",
            "MATCHED_VRI_NAME": pd.NA,
            "MATCHED_VRI_CODE": pd.NA,
            "PZZ_VRI_VERDICT": "unclear",
            "Статус": status_to_russian_label("unclear"),
            "PZZ_REASON": "Требуется ручная проверка.",
            "PZZ_NOT_ALLOWED_TOP5_CANDIDATES": pd.NA,
        }

        if not include_pzz_check:
            query_norm = normalize_text(vri_text).lower()
            exact_candidates: list[dict[str, Any]] = []
            if query_norm and context.classifier_embed_items_df is not None and (not context.classifier_embed_items_df.empty):
                df = context.classifier_embed_items_df
                name_norm = df["classifier_name"].fillna("").map(lambda x: normalize_text(x).lower())
                plain_norm = df["classifier_name_plain"].fillna("").map(lambda x: normalize_text(x).lower())
                mask = (name_norm == query_norm) | (plain_norm == query_norm)
                for rec in df[mask].head(5).to_dict("records"):
                    exact_candidates.append({
                        "score": 1.0,
                        "code": normalize_text(rec.get("classifier_code")),
                        "name": normalize_text(rec.get("classifier_name")),
                        "description": normalize_text(rec.get("classifier_description")),
                    })

            recall_candidates = build_not_allowed_same_zone_candidates(
                vri_text=vri_text,
                actual_zone_code=actual_zone_code,
                top_n=max(20, NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N),
                min_similarity=0.0,
                context=context,
            )
            ranked_candidates = fast_rerank_not_allowed_candidates(
                vri_text=vri_text,
                candidates=(exact_candidates + (recall_candidates or [])),
            )
            llm_input = ranked_candidates[:NOT_ALLOWED_LLM_RERANK_RECALL_TOP_N]
            final_candidates = ranked_candidates[:5]

            # Жилой объект без указания этажности/типа застройки: детерминированно
            # ставим обобщенный ВРИ 2.0 «Жилая застройка» в Топ-1 и не тратим LLM-реранк.
            residential_generic_case = is_residential_unspecified_vri(vri_text)

            if not residential_generic_case and should_run_not_allowed_llm_rerank(vri_text, llm_input):
                rerank_key = normalize_text(vri_text).lower()
                cached = llm_rerank_cache.get(rerank_key)
                if cached is not None:
                    final_candidates = cached
                else:
                    try:
                        result = run_not_allowed_rerank_with_llm(vri_text=vri_text, candidates=llm_input)[:5]
                    except Exception:
                        result = ranked_candidates[:5]
                    with _llm_cache_lock:
                        if rerank_key not in llm_rerank_cache:
                            llm_rerank_cache[rerank_key] = result
                    final_candidates = llm_rerank_cache[rerank_key]

            if residential_generic_case:
                final_candidates = promote_generic_residential_first(
                    vri_text=vri_text, candidates=ranked_candidates, context=context,
                )[:5]

            payload["CHECK_SCOPE"] = "classifier_only"
            payload["MATCH_METHOD"] = "classifier_top5_llm_or_fast"
            payload["PZZ_VRI_VERDICT"] = "classifier_only"
            payload["Статус"] = "Только кандидаты классификатора"
            payload["PZZ_REASON"] = "Проверка по ПЗЗ отключена: кандидаты из классификатора отобраны string-match + embed, при необходимости LLM-rerank."
            payload["PZZ_NOT_ALLOWED_TOP1_CANDIDATE"] = serialize_not_allowed_same_zone_candidates(final_candidates[:1])
            payload["PZZ_NOT_ALLOWED_TOP5_CANDIDATES"] = serialize_not_allowed_same_zone_candidates(final_candidates)
            return payload

        if not actual_zone_code:
            payload["PZZ_VRI_VERDICT"] = "no_actual_zone"
            payload["Статус"] = status_to_russian_label("no_actual_zone")
            payload["MATCH_METHOD"] = "no_actual_zone"
            payload["PZZ_REASON"] = "Участок не пересекается со слоем ПЗЗ."
            return payload

        zone_ref, _ = resolve_zone_reference(actual_zone_code, context=context)
        if zone_ref is None:
            payload["PZZ_VRI_VERDICT"] = "no_zone_metadata"
            payload["Статус"] = status_to_russian_label("no_zone_metadata")
            payload["MATCH_METHOD"] = "no_zone_metadata"
            payload["PZZ_REASON"] = "Для фактической зоны не найдено описание в шаблоне ПЗЗ."
            return payload

        residential_generic = resolve_residential_unspecified_in_zone(
            vri_text=vri_text, actual_zone_code=actual_zone_code, context=context,
        )
        if residential_generic is not None:
            verdict = normalize_text(residential_generic.get("verdict")) or "unclear"
            payload["PZZ_VRI_VERDICT"] = verdict
            payload["Статус"] = status_to_russian_label(verdict)
            payload["MATCH_METHOD"] = residential_generic.get("match_method") or "residential_unspecified_generic"
            payload["MATCHED_VRI_NAME"] = normalize_text(residential_generic.get("matched_vri_name")) or pd.NA
            payload["MATCHED_VRI_CODE"] = normalize_text(residential_generic.get("matched_vri_code")) or pd.NA
            payload["PZZ_REASON"] = normalize_text(residential_generic.get("reason")) or payload["PZZ_REASON"]
            return payload

        exact_matches = find_exact_zone_candidates(vri_text=vri_text, zone_code=actual_zone_code, context=context)
        best_exact = choose_best_exact_match(exact_matches)
        if best_exact is not None:
            verdict = f"allowed_{normalize_text(best_exact.get('section_name'))}" if normalize_text(best_exact.get("section_name")) else "allowed_main"
            payload["PZZ_VRI_VERDICT"] = verdict
            payload["Статус"] = status_to_russian_label(verdict)
            payload["MATCH_METHOD"] = "actual_zone_exact"
            payload["MATCHED_VRI_NAME"] = best_exact.get("matched_vri_name") or pd.NA
            payload["MATCHED_VRI_CODE"] = best_exact.get("matched_vri_code") or pd.NA
            payload["PZZ_REASON"] = "Точное / почти точное совпадение в фактической зоне."
            return payload

        string_hit = fast_string_match_in_zone(vri_text=vri_text, actual_zone_code=actual_zone_code, context=context)
        if string_hit and string_hit.get("use_direct"):
            verdict = normalize_text(string_hit.get("verdict")) or "unclear"
            payload["PZZ_VRI_VERDICT"] = verdict
            payload["Статус"] = status_to_russian_label(verdict)
            payload["MATCH_METHOD"] = "actual_zone_fast_string"
            payload["MATCHED_VRI_NAME"] = normalize_text(string_hit.get("matched_vri_name")) or pd.NA
            payload["MATCHED_VRI_CODE"] = normalize_text(string_hit.get("matched_vri_code")) or pd.NA
            payload["PZZ_REASON"] = normalize_text(string_hit.get("reason")) or payload["PZZ_REASON"]
            return payload

        embed_hit = fast_embed_match_in_zone(vri_text=vri_text, actual_zone_code=actual_zone_code, context=context)
        if embed_hit and embed_hit.get("use_direct"):
            verdict = normalize_text(embed_hit.get("verdict")) or "unclear"
            payload["PZZ_VRI_VERDICT"] = verdict
            payload["Статус"] = status_to_russian_label(verdict)
            payload["MATCH_METHOD"] = "actual_zone_fast_embed"
            payload["MATCHED_VRI_NAME"] = normalize_text(embed_hit.get("matched_vri_name")) or pd.NA
            payload["MATCHED_VRI_CODE"] = normalize_text(embed_hit.get("matched_vri_code")) or pd.NA
            payload["PZZ_REASON"] = normalize_text(embed_hit.get("reason")) or payload["PZZ_REASON"]
            return payload

        llm_cache_key = build_actual_zone_key(vri_text=vri_text, actual_code=actual_zone_code)
        llm_decision = llm_zone_check_cache.get(llm_cache_key)

        if llm_decision is None and _ENABLE_LLM:
            llm_prompt = build_zone_check_prompt(
                vri_text=vri_text,
                zone_ref=zone_ref,
                exact_matches=exact_matches,
                actual_zone_code=actual_zone_code,
                actual_zone_name=zone_ref.get("zone_name"),
                actual_share=pd.NA,
                intersect_codes=pd.NA,
                context=context,
            )
            try:
                llm_response = run_zone_check_with_llm(
                    prompt=llm_prompt,
                    think_override=should_use_deeper_llm_reasoning(vri_text=vri_text, zone_ref=zone_ref),
                    context=context,
                )
                computed = {
                    "verdict": normalize_llm_verdict(
                        verdict=llm_response.get("verdict"),
                        reason=llm_response.get("reason"),
                        matched_vri_name=llm_response.get("matched_vri_name"),
                    ),
                    "matched_vri_name": llm_response.get("matched_vri_name"),
                    "matched_vri_code": llm_response.get("matched_vri_code"),
                    "reason": llm_response.get("reason"),
                }
            except Exception as exc:
                computed = {
                    "verdict": "not_allowed",
                    "matched_vri_name": None,
                    "matched_vri_code": None,
                    "reason": f"LLM-check завершился ошибкой: {exc}",
                }
            with _llm_cache_lock:
                if llm_cache_key not in llm_zone_check_cache:
                    llm_zone_check_cache[llm_cache_key] = computed
            llm_decision = llm_zone_check_cache[llm_cache_key]

        if llm_decision is not None:
            payload["PZZ_VRI_VERDICT"] = normalize_text(llm_decision.get("verdict")) or "unclear"
            payload["Статус"] = status_to_russian_label(payload["PZZ_VRI_VERDICT"])
            payload["MATCH_METHOD"] = "actual_zone_llm"
            payload["MATCHED_VRI_NAME"] = normalize_text(llm_decision.get("matched_vri_name")) or pd.NA
            payload["MATCHED_VRI_CODE"] = normalize_text(llm_decision.get("matched_vri_code")) or pd.NA
            payload["PZZ_REASON"] = normalize_text(llm_decision.get("reason")) or "Решение принято на основе LLM-проверки фактической зоны."
            return payload

        payload["PZZ_VRI_VERDICT"] = "not_allowed"
        payload["Статус"] = status_to_russian_label("not_allowed")
        payload["MATCH_METHOD"] = "actual_zone_not_allowed"
        payload["PZZ_REASON"] = "Совпадения с разрешенными ВРИ фактической зоны не найдено."
        return payload

    _log_stage("classification_loop", "start",
               unique_rows=len(unique_df),
               llm_workers=_PIPELINE_LLM_WORKERS)

    row_list = [row for _, row in unique_df.iterrows()]
    rows: list[dict[str, Any]] = [None] * len(row_list)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=_PIPELINE_LLM_WORKERS) as pool:
        future_to_idx = {pool.submit(_process_row, row): idx for idx, row in enumerate(row_list)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rows[idx] = future.result()
            except Exception as exc:
                logger.error("Classification row %d failed: %s", idx, exc, exc_info=True)
                row = row_list[idx]
                rows[idx] = {
                    "__comparison_key__": row.get("__comparison_key__"),
                    cadastral_vri_col: row.get(cadastral_vri_col),
                    "CHECK_SCOPE": "error",
                    "MATCH_METHOD": "error",
                    "MATCHED_VRI_NAME": pd.NA,
                    "MATCHED_VRI_CODE": pd.NA,
                    "PZZ_VRI_VERDICT": "not_allowed",
                    "Статус": status_to_russian_label("not_allowed"),
                    "PZZ_REASON": f"Внутренняя ошибка классификации: {exc}",
                    "PZZ_NOT_ALLOWED_TOP5_CANDIDATES": pd.NA,
                }

    _log_stage("classification_loop", "finished",
               rows=len(rows),
               llm_cache_size=len(llm_zone_check_cache),
               rerank_cache_size=len(llm_rerank_cache))

    rerank_started = perf_counter()
    classified_unique_df = pd.DataFrame(rows)
    classified_unique_df = attach_not_allowed_llm_rerank_column(
        classified_unique_df,
        cadastral_vri_col=cadastral_vri_col,
        context=context,
    )
    _log_stage("postprocess_rerank", "finished", duration_ms=int((perf_counter() - rerank_started) * 1000), rows=len(classified_unique_df))

    classified_gdf = source_with_spatial_gdf.merge(
        classified_unique_df,
        on="__comparison_key__",
        how="left",
    )
    source_vri_col_after_merge = f"{cadastral_vri_col}_x"
    if cadastral_vri_col not in classified_gdf.columns and source_vri_col_after_merge in classified_gdf.columns:
        classified_gdf[cadastral_vri_col] = classified_gdf[source_vri_col_after_merge]

    classified_gdf = ensure_classification_columns(classified_gdf)

    classified_gdf = mark_manual_review_for_allowed_top_candidates(
        gdf=classified_gdf,
        zone_templates=zone_templates,
        actual_zone_col="PZZ_ACTUAL_CODE",
        verdict_col="PZZ_VRI_VERDICT",
        status_col="Статус",
        candidates_col="PZZ_NOT_ALLOWED_TOP5_CANDIDATES",
        reason_col="PZZ_REASON",
        include_conditional=False,
    )
    classified_gdf = attach_actual_zone_name_column(
        gdf=classified_gdf,
        zone_templates=zone_templates,
        actual_zone_code_col="PZZ_ACTUAL_CODE",
        output_col="PZZ_ACTUAL_NAME",
        prefer_heading=True,
        overwrite=False,
    )

    classified_gdf["PZZ_ACTUAL_CODE_x"] = classified_gdf["PZZ_ACTUAL_CODE"]
    classified_gdf["PZZ_ACTUAL_NAME_x"] = classified_gdf["PZZ_ACTUAL_NAME"]
    output_gdf = select_and_rename_result_columns(
        classified_gdf,
        cadastral_vri_col=cadastral_vri_col,
    )

    write_started = perf_counter()
    output_path = Path(output_geojson_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_gdf.to_file(output_path, driver="GeoJSON")

    output_table = pd.DataFrame(output_gdf.drop(columns="geometry", errors="ignore"))
    output_table.to_excel(unique_results_xlsx_path, index=False)
    output_table.to_json(unique_results_json_path, orient="records", force_ascii=False, indent=2)
    _log_stage("write_outputs", "finished",
               duration_ms=int((perf_counter() - write_started) * 1000),
               output_geojson_path=Path(output_geojson_path).name,
               unique_results_xlsx_path=Path(unique_results_xlsx_path).name,
               unique_results_json_path=Path(unique_results_json_path).name)
    _log_stage("pipeline", "finished", duration_ms=int((perf_counter() - total_started) * 1000))
