"""Overlap checks: parcels on parcels, zones on zones, parcels in several zones.

Coordinates are degrees near the equator, where 0.001° ≈ 111 m, so a 0.001°
square is ≈ 12 300 m² — big enough that the 1 m² / 0.1 % noise thresholds only
bite where a test means them to.
"""

import geopandas as gpd
import pytest
from shapely.geometry import box

from pipeline_modules.business.overlap_layer import (
    COL_ACTUAL_SHARE_PCT,
    COL_INTERSECT_ZONES,
    COL_MO,
    COL_OVERLAP_NOTES,
    KIND_MO_MO,
    KIND_PARCEL_MULTI_MO,
    KIND_PARCEL_MULTI_ZONE,
    KIND_PARCEL_OUTSIDE_MO,
    KIND_PARCEL_PARCEL,
    KIND_ZONE_MULTI_MO,
    KIND_ZONE_OUTSIDE_MO,
    KIND_ZONE_ZONE,
    detect_cadastral_number_column,
    detect_mo_name_column,
    run_overlap_checks,
)
from service.api.tasks import _overlap_summary_lines as overlap_summary_lines


def _zones(shapes):
    return gpd.GeoDataFrame(
        {
            "Индекс_зоны": [code for code, _ in shapes],
            "Наименование_зоны": [f"Зона {code}" for code, _ in shapes],
            "geometry": [geom for _, geom in shapes],
        },
        crs="EPSG:4326",
    )


def _parcels(geoms, numbers=None):
    data = {"geometry": geoms}
    if numbers is not None:
        data["Кадастровый_номер"] = numbers
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def _check(parcels, zones):
    return run_overlap_checks(
        parcels,
        zones,
        zone_code_col="Индекс_зоны",
        zone_name_col="Наименование_зоны",
    )


def _kinds(result):
    return [f["properties"]["kind"] for f in result.report["features"]]


BIG_ZONE = [("Ж-1", box(0.0, 0.0, 0.01, 0.01))]


def test_clean_layers_report_nothing():
    parcels = _parcels([box(0.001, 0.001, 0.002, 0.002), box(0.002, 0.001, 0.003, 0.002)])
    result = _check(parcels, _zones(BIG_ZONE))

    assert result.report["features"] == []
    assert result.summary["parcel_overlaps"] == 0
    assert result.summary["parcels_in_multiple_zones"] == 0
    assert result.summary["zone_overlaps"] == 0
    # Touching parcels are neighbours, not an overlap.
    assert result.parcel_columns[COL_OVERLAP_NOTES].isna().all()
    assert result.parcel_columns[COL_ACTUAL_SHARE_PCT].tolist() == [100.0, 100.0]
    assert overlap_summary_lines(result.summary) == []


def test_overlapping_parcels_are_reported_on_both_sides():
    parcels = _parcels(
        [box(0.001, 0.001, 0.002, 0.002), box(0.0015, 0.001, 0.0025, 0.002)],
        numbers=["47:01:0101001:11", "47:01:0101001:12"],
    )
    result = _check(parcels, _zones(BIG_ZONE))

    assert _kinds(result) == [KIND_PARCEL_PARCEL]
    props = result.report["features"][0]["properties"]
    assert props["feature_index_1"] == 0 and props["feature_index_2"] == 1
    assert props["Доля_наложения_%"] == pytest.approx(50, abs=0.5)
    assert "47:01:0101001:12" in props["Объект_2"]
    notes = result.parcel_columns[COL_OVERLAP_NOTES].tolist()
    assert "№ 1 (КН 47:01:0101001:12)" in notes[0]
    assert "№ 0 (КН 47:01:0101001:11)" in notes[1]
    assert result.summary["parcel_overlaps"] == 1
    assert result.summary["parcels_with_overlaps"] == 2


def test_sliver_overlap_is_digitising_noise():
    # 0.0000001° ≈ 1 cm wide strip along a 111 m edge: ≈ 1 m² and 0.01 %.
    parcels = _parcels(
        [box(0.001, 0.001, 0.002, 0.002), box(0.0019999, 0.001, 0.003, 0.002)]
    )
    result = _check(parcels, _zones(BIG_ZONE))

    assert result.summary["parcel_overlaps"] == 0


def test_overlapping_zones_are_reported():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.01, 0.01)), ("ОД-1", box(0.008, 0.0, 0.02, 0.01))])
    result = _check(_parcels([box(0.001, 0.001, 0.002, 0.002)]), zones)

    assert _kinds(result) == [KIND_ZONE_ZONE]
    props = result.report["features"][0]["properties"]
    assert props["Объект_1"] == "Зона Ж-1 «Зона Ж-1»"
    assert props["Объект_2"] == "Зона ОД-1 «Зона ОД-1»"
    assert result.summary["zone_overlaps"] == 1


def test_parcel_cut_by_zone_boundary():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.00175, 0.01)), ("ОД-1", box(0.00175, 0.0, 0.01, 0.01))])
    result = _check(_parcels([box(0.001, 0.001, 0.002, 0.002)]), zones)

    assert _kinds(result) == [KIND_PARCEL_MULTI_ZONE]
    props = result.report["features"][0]["properties"]
    assert props["Основная_зона"] == "Ж-1"
    assert props["Доля_в_основной_зоне_%"] == pytest.approx(75, abs=0.5)
    assert props["Объект_2"] == "Зоны ОД-1"
    row = result.parcel_columns.iloc[0]
    assert row[COL_INTERSECT_ZONES] == "Ж-1 (75 %), ОД-1 (25 %)"
    assert row[COL_OVERLAP_NOTES].startswith("Участок расположен в нескольких зонах")
    assert result.summary["parcels_in_multiple_zones"] == 1
    assert overlap_summary_lines(result.summary) == [
        "Участков, расположенных сразу в нескольких территориальных зонах: 1."
    ]


def test_parcel_grazing_a_neighbour_zone_stays_single_zone():
    # 0.05 % of the parcel lies across the boundary — below the 0.1 % threshold.
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.0019995, 0.01)), ("ОД-1", box(0.0019995, 0.0, 0.01, 0.01))])
    result = _check(_parcels([box(0.001, 0.001, 0.002, 0.002)]), zones)

    assert result.summary["parcels_in_multiple_zones"] == 0
    assert result.parcel_columns.iloc[0][COL_INTERSECT_ZONES] is None


def test_parcel_outside_zones_has_no_share():
    result = _check(_parcels([box(0.05, 0.05, 0.051, 0.051)]), _zones(BIG_ZONE))

    assert result.parcel_columns.iloc[0][COL_ACTUAL_SHARE_PCT] is None
    assert result.report["features"] == []


def test_cadastral_number_column_is_found_by_values():
    frame = _parcels(
        [box(0, 0, 1, 1)] * 3,
        numbers=["47:01:0101001:11", "47:01:0101001:12", "66:41:0000000:7"],
    )
    frame["Примечание"] = ["47:01", "x", None]
    assert detect_cadastral_number_column(frame) == "Кадастровый_номер"
    assert detect_cadastral_number_column(frame.drop(columns="Кадастровый_номер")) is None


def test_summary_lines_mention_every_kind():
    lines = overlap_summary_lines(
        {
            "parcel_overlaps": 2,
            "parcels_with_overlaps": 3,
            "parcels_in_multiple_zones": 4,
            "zone_overlaps": 1,
        }
    )
    assert lines == [
        "Наложения земельных участков друг на друга: 2 (затронуто участков: 3).",
        "Участков, расположенных сразу в нескольких территориальных зонах: 4.",
        "Наложения территориальных зон друг на друга: 1.",
    ]



# --- Municipal boundaries (МО) ---------------------------------------------

def _mo(shapes):
    return gpd.GeoDataFrame(
        {
            "Наименование_МО": [name for name, _ in shapes],
            "geometry": [geom for _, geom in shapes],
        },
        crs="EPSG:4326",
    )


WEST_EAST_MO = [
    ("Западный", box(0.0, 0.0, 0.00175, 0.01)),
    ("Восточный", box(0.00175, 0.0, 0.01, 0.01)),
]


def _check_mo(parcels, zones, mo):
    return run_overlap_checks(
        parcels,
        zones,
        zone_code_col="Индекс_зоны",
        zone_name_col="Наименование_зоны",
        mo_gdf=mo,
    )


def test_without_mo_layer_nothing_about_mo():
    result = _check(_parcels([box(0.001, 0.001, 0.002, 0.002)]), _zones(BIG_ZONE))

    assert result.summary["mo_checked"] is False
    assert "parcels_outside_mo" not in result.summary
    assert COL_MO not in result.parcel_columns.columns


def test_parcel_inside_one_mo_gets_its_name():
    result = _check_mo(
        _parcels([box(0.003, 0.001, 0.004, 0.002)]),
        _zones(BIG_ZONE),
        _mo(WEST_EAST_MO),
    )

    assert _kinds(result) == [KIND_ZONE_MULTI_MO]
    assert result.summary["mo_checked"] is True
    assert result.summary["parcels_in_multiple_mo"] == 0
    assert result.parcel_columns.iloc[0][COL_MO] == "МО «Восточный»"
    assert result.parcel_columns.iloc[0][COL_OVERLAP_NOTES] is None
    # The zone straddles both МО — that is reported, parcels are clean.
    assert result.summary["zones_in_multiple_mo"] == 1


def test_parcel_split_between_two_mo():
    result = _check_mo(
        _parcels([box(0.001, 0.001, 0.002, 0.002)]),
        _zones([("Ж-1", box(0.0, 0.0, 0.00175, 0.01))]),
        _mo(WEST_EAST_MO),
    )

    assert KIND_PARCEL_MULTI_MO in _kinds(result)
    props = next(
        f["properties"]
        for f in result.report["features"]
        if f["properties"]["kind"] == KIND_PARCEL_MULTI_MO
    )
    assert props["Основное_МО"] == "МО «Западный»"
    assert props["Доля_в_основном_МО_%"] == pytest.approx(75, abs=0.5)
    assert props["Объект_2"] == "МО «Восточный»"
    row = result.parcel_columns.iloc[0]
    assert row[COL_MO] == "МО «Западный» (75 %), МО «Восточный» (25 %)"
    assert "пересекает границу МО" in row[COL_OVERLAP_NOTES]
    assert result.summary["parcels_in_multiple_mo"] == 1
    assert result.summary["parcels_outside_mo"] == 0


def test_parcel_partly_and_wholly_outside_mo():
    mo = _mo([("Западный", box(0.0, 0.0, 0.00175, 0.01))])
    parcels = _parcels(
        [box(0.001, 0.001, 0.002, 0.002), box(0.005, 0.001, 0.006, 0.002)]
    )
    result = _check_mo(parcels, _zones([("Ж-1", box(0.0, 0.0, 0.00175, 0.01))]), mo)

    outside = [
        f["properties"]
        for f in result.report["features"]
        if f["properties"]["kind"] == KIND_PARCEL_OUTSIDE_MO
    ]
    assert [p["feature_index_1"] for p in outside] == [0, 1]
    assert outside[0]["Доля_наложения_%"] == pytest.approx(25, abs=0.5)
    assert outside[1]["Доля_наложения_%"] == pytest.approx(100, abs=0.1)
    notes = result.parcel_columns[COL_OVERLAP_NOTES].tolist()
    assert "частично вне границ МО" in notes[0]
    assert "расположен вне границ МО" in notes[1]
    assert result.parcel_columns.iloc[1][COL_MO] == "вне МО (100 %)"
    assert result.summary["parcels_outside_mo"] == 2
    assert result.summary["parcels_in_multiple_mo"] == 0


def test_zone_across_mo_boundary_and_outside_mo():
    mo = _mo([("Западный", box(0.0, 0.0, 0.005, 0.01)), ("Восточный", box(0.005, 0.0, 0.01, 0.01))])
    zones = _zones(
        [("Ж-1", box(0.004, 0.0, 0.006, 0.01)), ("ОД-1", box(0.009, 0.0, 0.012, 0.01))]
    )
    result = _check_mo(_parcels([box(0.001, 0.001, 0.002, 0.002)]), zones, mo)

    by_kind = {}
    for f in result.report["features"]:
        by_kind.setdefault(f["properties"]["kind"], []).append(f["properties"])
    assert [p["zone_index_1"] for p in by_kind[KIND_ZONE_MULTI_MO]] == [0]
    assert by_kind[KIND_ZONE_MULTI_MO][0]["Объект_1"] == "Зона Ж-1 «Зона Ж-1»"
    assert [p["zone_index_1"] for p in by_kind[KIND_ZONE_OUTSIDE_MO]] == [1]
    assert by_kind[KIND_ZONE_OUTSIDE_MO][0]["Доля_наложения_%"] == pytest.approx(
        66.67, abs=0.5
    )
    assert result.summary["zones_in_multiple_mo"] == 1
    assert result.summary["zones_outside_mo"] == 1


def test_overlapping_mo_are_reported():
    mo = _mo([("Западный", box(0.0, 0.0, 0.006, 0.01)), ("Восточный", box(0.005, 0.0, 0.01, 0.01))])
    result = _check_mo(_parcels([box(0.001, 0.001, 0.002, 0.002)]), _zones([]), mo)

    mo_features = [
        f["properties"] for f in result.report["features"] if f["properties"]["kind"] == KIND_MO_MO
    ]
    assert len(mo_features) == 1
    assert mo_features[0]["Объект_1"] == "МО «Западный»"
    assert mo_features[0]["mo_index_1"] == 0 and mo_features[0]["mo_index_2"] == 1
    assert result.summary["mo_overlaps"] == 1


def test_mo_name_column_detection():
    frame = _mo(WEST_EAST_MO)
    assert detect_mo_name_column(frame) == "Наименование_МО"
    renamed = frame.rename(columns={"Наименование_МО": "что-то"})
    assert detect_mo_name_column(renamed) == "что-то"  # unique text values
    assert detect_mo_name_column(frame[["geometry"]]) is None


def test_mo_without_names_gets_numbers():
    mo = _mo(WEST_EAST_MO)[["geometry"]]
    result = _check_mo(_parcels([box(0.001, 0.001, 0.002, 0.002)]), _zones(BIG_ZONE), mo)

    assert result.parcel_columns.iloc[0][COL_MO] == "МО № 0 (75 %), МО № 1 (25 %)"


def test_summary_lines_mention_mo_kinds():
    lines = overlap_summary_lines(
        {
            "parcel_overlaps": 0,
            "mo_overlaps": 1,
            "parcels_in_multiple_mo": 2,
            "parcels_outside_mo": 3,
            "zones_in_multiple_mo": 0,
            "zones_outside_mo": 4,
        }
    )
    assert lines == [
        "Наложения границ муниципальных образований друг на друга: 1.",
        "Участков, пересекающих границу муниципальных образований: 2.",
        "Участков, полностью или частично вне границ муниципальных образований: 3.",
        "Территориальных зон, полностью или частично вне границ муниципальных образований: 4.",
    ]
