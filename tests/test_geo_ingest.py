"""Tests for uploaded geo-file → GeoJSON conversion (phase 5)."""
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from service.infrastructure.geo_ingest import (
    GeoIngestError,
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
