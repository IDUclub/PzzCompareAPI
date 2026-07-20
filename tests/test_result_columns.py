"""Result-layer property hygiene: pandas NA must serialise as JSON null."""

import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from pipeline_modules.business.postprocess_layer import select_and_rename_result_columns


def test_missing_values_serialise_as_null_not_na_string(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "vri": ["Для ИЖС", None],
            "Статус": ["Разрешен", "Требуется ручная проверка"],
            "MATCHED_VRI_NAME": [pd.NA, pd.NA],  # empty on every row
            "PZZ_NOT_ALLOWED_TOP1_CANDIDATE": ["2.1 ИЖС", pd.NA],  # empty on one row
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )

    out = select_and_rename_result_columns(gdf, cadastral_vri_col="vri")
    path = tmp_path / "result.geojson"
    out.to_file(path, driver="GeoJSON")
    features = json.load(open(path, encoding="utf-8"))["features"]

    for feat in features:
        props = feat["properties"]
        # keys stay present on every feature (uniform schema)
        assert "Подобранный_ВРИ" in props
        assert "Топ1_возможный_ВРИ" in props
        # ...and empty values are JSON null, never the literal "<NA>"
        assert props["Подобранный_ВРИ"] is None
    assert "<NA>" not in json.dumps(features, ensure_ascii=False)
    # a present value survives; the empty one is null
    assert features[0]["properties"]["Топ1_возможный_ВРИ"] == "2.1 ИЖС"
    assert features[1]["properties"]["Топ1_возможный_ВРИ"] is None
