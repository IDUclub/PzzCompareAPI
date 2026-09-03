"""attach_spatial_pzz_attributes handles non-polygonal parcels and sums zone overlap."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)

from pipeline_modules.business.spatial_layer import attach_spatial_pzz_attributes


def _zones(shapes):
    return gpd.GeoDataFrame(
        {
            "Индекс_зоны": [code for code, _ in shapes],
            "Наименование_зоны": [f"Зона {code}" for code, _ in shapes],
            "geometry": [geom for _, geom in shapes],
        },
        crs="EPSG:4326",
    )


def _parcels(geoms):
    return gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:4326")


def _attach(parcels, zones):
    return attach_spatial_pzz_attributes(
        parcels_gdf=parcels,
        pzz_gdf=zones,
        zone_code_col="Индекс_зоны",
        zone_name_col="Наименование_зоны",
    )


def test_linear_parcel_is_matched_to_its_zone():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.02, 0.02))])
    parcels = _parcels([LineString([(0.004, 0.01), (0.016, 0.01)])])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
    assert result.loc[0, "PZZ_ACTUAL_NAME"] == "Зона Ж-1"
    assert result.loc[0, "PZZ_INTERSECT_COUNT"] == 1
    assert result.loc[0, "PZZ_ACTUAL_SHARE"] == pytest.approx(1.0, rel=1e-3)


def test_linear_parcel_share_is_measured_by_length():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.01, 0.02))])
    parcels = _parcels([LineString([(0.005, 0.01), (0.015, 0.01)])])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
    assert result.loc[0, "PZZ_ACTUAL_SHARE"] == pytest.approx(0.5, rel=1e-2)


def test_multilinestring_parcel_is_matched():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.02, 0.02))])
    parcels = _parcels(
        [
            MultiLineString(
                [
                    [(0.004, 0.005), (0.016, 0.005)],
                    [(0.004, 0.015), (0.016, 0.015)],
                ]
            )
        ]
    )

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"


def test_point_parcel_is_matched_to_containing_zone():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.02, 0.02))])
    parcels = _parcels([Point(0.01, 0.01)])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
    assert result.loc[0, "PZZ_ACTUAL_SHARE"] == pytest.approx(1.0)


def test_multipoint_dominant_zone_is_selected_by_point_count():
    zones = _zones(
        [
            ("Б", box(0.01, 0.0, 0.02, 0.01)),
            ("А", box(0.0, 0.0, 0.01, 0.01)),
        ]
    )
    points = [(0.001 + idx * 0.0005, 0.001) for idx in range(9)]
    points.append((0.015, 0.001))
    parcels = _parcels([MultiPoint(points)])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "А"
    assert result.loc[0, "PZZ_ACTUAL_SHARE"] == pytest.approx(0.9)


def test_mixed_geometry_layer_keeps_every_parcel():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.02, 0.02))])
    parcels = _parcels(
        [
            box(0.004, 0.004, 0.008, 0.008),
            LineString([(0.004, 0.01), (0.016, 0.01)]),
            Point(0.012, 0.012),
        ]
    )

    result = _attach(parcels, zones)

    assert len(result) == 3
    assert result["PZZ_ACTUAL_CODE"].tolist() == ["Ж-1", "Ж-1", "Ж-1"]


def test_dominant_zone_sums_all_intersection_pieces():
    """A zone split into several polygons must not lose to a single compact rival."""
    zones = _zones(
        [
            ("Ж-1", box(0.000, 0.000, 0.010, 0.002)),
            ("Ж-1", box(0.000, 0.004, 0.010, 0.006)),
            ("Ж-1", box(0.000, 0.008, 0.010, 0.010)),
            ("ОД-1", box(0.000, 0.012, 0.010, 0.015)),
        ]
    )
    parcels = _parcels([box(0.000, 0.000, 0.010, 0.020)])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
    assert result.loc[0, "PZZ_INTERSECT_CODES"].split(" | ")[0] == "Ж-1"
    assert result.loc[0, "PZZ_INTERSECT_COUNT"] == 2


def test_overlapping_features_of_same_zone_are_not_double_counted():
    zones = _zones(
        [
            ("Ж-1", box(0.0, 0.0, 0.01, 0.01)),
            ("Ж-1", box(0.0, 0.0, 0.01, 0.01)),
        ]
    )
    parcels = _parcels([box(0.0, 0.0, 0.01, 0.01)])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
    assert result.loc[0, "PZZ_ACTUAL_SHARE"] == pytest.approx(1.0)


def test_boundary_only_contact_is_not_an_intersection():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.01, 0.01))])
    parcels = _parcels([box(0.01, 0.002, 0.02, 0.008)])

    result = _attach(parcels, zones)

    assert pd.isna(result.loc[0, "PZZ_ACTUAL_CODE"])
    assert result.loc[0, "PZZ_INTERSECT_COUNT"] == 0
    assert result.loc[0, "PZZ_SPATIAL_NOTE"] == "No intersection with PZZ"


def test_parcel_outside_every_zone_reports_no_intersection():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.01, 0.01))])
    parcels = _parcels([box(0.10, 0.10, 0.11, 0.11)])

    result = _attach(parcels, zones)

    assert pd.isna(result.loc[0, "PZZ_ACTUAL_CODE"])
    assert result.loc[0, "PZZ_INTERSECT_COUNT"] == 0


def test_polygon_share_still_uses_area():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.01, 0.02))])
    parcels = _parcels([box(0.005, 0.005, 0.015, 0.015)])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
    assert result.loc[0, "PZZ_ACTUAL_SHARE"] == pytest.approx(0.5, rel=1e-2)


def test_self_intersecting_polygon_survives_validation():
    zones = _zones([("Ж-1", box(0.0, 0.0, 0.02, 0.02))])
    bowtie = Polygon([(0.004, 0.004), (0.012, 0.012), (0.004, 0.012), (0.012, 0.004)])
    parcels = _parcels([bowtie])

    result = _attach(parcels, zones)

    assert result.loc[0, "PZZ_ACTUAL_CODE"] == "Ж-1"
