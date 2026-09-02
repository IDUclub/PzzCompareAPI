"""Tests for uploaded geo-file → GeoJSON conversion (phase 5)."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from service.infrastructure.geo_ingest import (
    GeoCrsError,
    GeoIngestError,
    ensure_wgs84,
    geo_file_to_geojson_dict,
    is_geojson_filename,
    supported_extensions,
)


def _sample_gdf_3857() -> gpd.GeoDataFrame:
    """Two points around St. Petersburg, stored in EPSG:3857 (not 4326)."""
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"], "geometry": [Point(30.3, 59.9), Point(30.4, 60.0)]},
        crs="EPSG:4326",
    )
    return gdf.to_crs("EPSG:3857")


@pytest.mark.parametrize("filename", ["t.gpkg", "t.geoparquet", "t.gml", "t.kml"])
def test_geo_formats_convert_and_reproject(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    gdf = _sample_gdf_3857()
    if filename.endswith((".parquet", ".geoparquet")):
        gdf.to_parquet(path)
    elif filename.endswith(".kml"):
        gdf.to_file(path, driver="KML")
    else:
        gdf.to_file(path)

    fc = geo_file_to_geojson_dict(path)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    # Reprojected back to lon/lat near the original 4326 coordinates.
    x, y = fc["features"][0]["geometry"]["coordinates"]
    assert abs(x - 30.3) < 0.1
    assert abs(y - 59.9) < 0.1


def test_unreadable_file_raises_geoingest_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.gpkg"
    bad.write_bytes(b"not a geopackage")
    with pytest.raises(GeoIngestError):
        geo_file_to_geojson_dict(bad)


def test_is_geojson_filename() -> None:
    assert is_geojson_filename("x.geojson")
    assert is_geojson_filename("x.json")
    assert is_geojson_filename("noext")  # unknown ext treated as GeoJSON
    assert is_geojson_filename(None)
    assert not is_geojson_filename("x.gpkg")
    assert not is_geojson_filename("x.kml")


def test_supported_extensions_cover_selected_formats() -> None:
    ext = supported_extensions()
    for e in (".geojson", ".json", ".gpkg", ".gml", ".kml", ".geoparquet", ".parquet"):
        assert e in ext


# --- WGS84 gate on upload ----------------------------------------------------
#
# A GeoJSON exported from a local/NonEarth projection keeps projected metres and
# declares no CRS. It used to pass ingestion and only blow up minutes later in
# the pipeline, on ``estimate_utm_crs``.


def _fc(*coordinates, crs_name: str | None = None) -> dict:
    fc: dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": list(xy)},
                "properties": {},
            }
            for xy in coordinates
        ],
    }
    if crs_name is not None:
        fc["crs"] = {"type": "name", "properties": {"name": crs_name}}
    return fc


def test_wgs84_layer_passes() -> None:
    ensure_wgs84(_fc([142.75, 46.96], [30.3, 59.9], [-179.9, -89.9]))


def test_projected_metres_are_rejected() -> None:
    # The Yuzhno-Sakhalinsk parcels layer: NonEarth metres, no declared CRS.
    with pytest.raises(GeoCrsError) as exc:
        ensure_wgs84(_fc([1284377.58, 709961.18]))
    assert "1284377.58" in str(exc.value)


def test_declared_non_wgs84_crs_is_rejected_even_when_values_are_in_range() -> None:
    with pytest.raises(GeoCrsError) as exc:
        ensure_wgs84(_fc([10.0, 20.0], crs_name="urn:ogc:def:crs:EPSG::3857"))
    assert "3857" in str(exc.value)


@pytest.mark.parametrize(
    "crs_name",
    ["EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84", "urn:ogc:def:crs:EPSG::4326"],
)
def test_declared_wgs84_crs_passes(crs_name: str) -> None:
    ensure_wgs84(_fc([142.75, 46.96], crs_name=crs_name))


def test_nested_and_missing_geometry_do_not_break_the_check() -> None:
    ensure_wgs84({"type": "FeatureCollection", "features": []})
    ensure_wgs84({"type": "FeatureCollection", "features": [{"geometry": None}]})
    polygon = {
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[142.7, 46.9], [142.8, 46.9], [142.7, 47.0]]]],
                }
            }
        ],
    }
    ensure_wgs84(polygon)


def test_geometry_collection_is_walked() -> None:
    with pytest.raises(GeoCrsError):
        ensure_wgs84(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "geometry": {
                            "type": "GeometryCollection",
                            "geometries": [
                                {
                                    "type": "Point",
                                    "coordinates": [1284377.58, 709961.18],
                                }
                            ],
                        }
                    }
                ],
            }
        )


def test_geo_crs_error_is_a_geo_ingest_error() -> None:
    # Callers that only know the base class must still catch it.
    assert issubclass(GeoCrsError, GeoIngestError)
