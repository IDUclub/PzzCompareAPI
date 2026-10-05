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
    COL_OVERLAP_NOTES,
    KIND_PARCEL_MULTI_ZONE,
    KIND_PARCEL_PARCEL,
    KIND_ZONE_ZONE,
    detect_cadastral_number_column,
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
