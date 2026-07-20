"""select_and_rename_result_columns drops PZZ-only columns in classifier-only mode."""

import geopandas as gpd
from shapely.geometry import Point

from pipeline_modules.business.postprocess_layer import (
    select_and_rename_result_columns,
)

_PZZ_ONLY_RENAMED = {
    "Код фактической зоны нахождения кадастра",
    "Название фактической зоны нахождения кадастра",
    "Подобранный_ВРИ",
    "Код_подобранного_ВРИ",
    "Код_возможного_подобранного_ВРИ",
}


def _sample_gdf():
    return gpd.GeoDataFrame(
        {
            "vri_col": ["ИЖС"],
            "PZZ_ACTUAL_CODE_x": [""],
            "PZZ_ACTUAL_NAME_x": [None],
            "CHECK_SCOPE": ["classifier_only"],
            "Статус": ["Только кандидаты классификатора"],
            "PZZ_REASON": ["classifier only"],
            "MATCH_METHOD": ["classifier_top5_llm_or_fast"],
            "MATCHED_VRI_NAME": [None],
            "MATCHED_VRI_CODE": [None],
            "ALLOWED_TOP_CANDIDATE_CODES": [None],
            "PZZ_NOT_ALLOWED_TOP1_CANDIDATE": ["12.0 ..."],
            "PZZ_NOT_ALLOWED_TOP5_CANDIDATES": ["12.0 ..., 13.0 ..."],
            "geometry": [Point(0, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_classify_only_drops_pzz_columns():
    out = select_and_rename_result_columns(
        _sample_gdf(), cadastral_vri_col="vri_col", include_pzz_check=False
    )
    cols = set(out.columns)
    assert _PZZ_ONLY_RENAMED.isdisjoint(cols), cols & _PZZ_ONLY_RENAMED
    # meaningful classify columns survive
    assert {"ВРИ_ЕГРН", "Вердикт_ПЗЗ", "Область_проверки", "Топ5_возможных_ВРИ"} <= cols


def test_pzz_check_keeps_pzz_columns():
    out = select_and_rename_result_columns(
        _sample_gdf(), cadastral_vri_col="vri_col", include_pzz_check=True
    )
    assert _PZZ_ONLY_RENAMED <= set(out.columns)
