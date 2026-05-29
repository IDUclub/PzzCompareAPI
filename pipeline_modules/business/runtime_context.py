from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PipelineRuntimeContext:
    """Runtime dependencies and lookup caches for one pipeline run."""

    zone_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_zone_lookup: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw_zone_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)
    zone_items_lookup: dict[str, pd.DataFrame] = field(default_factory=dict)
    zone_fast_text_lookup: dict[str, dict[str, str]] = field(default_factory=dict)

    pzz_ref_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    pzz_vri_items_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    zone_vectorizer: Any = None
    zone_matrix: Any = None
    zone_item_embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    rosreestr_classifier_by_code: dict[str, dict[str, Any]] = field(default_factory=dict)
    rosreestr_classifier_children_map: dict[str, set[str]] = field(default_factory=dict)
    zone_section_code_cache: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    classifier_embed_items_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    classifier_embed_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    not_allowed_query_vector_cache: dict[str, np.ndarray] = field(default_factory=dict)
    not_allowed_recall_candidates_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    not_allowed_fast_rerank_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    not_allowed_llm_rerank_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    vectorizer: Any = None
    llm_client: Any = None
