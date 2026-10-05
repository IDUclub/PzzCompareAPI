"""Shapefile and MapInfo layers arrive as a ZIP archive holding one layer.

Both formats are several files per layer (geometry, attributes, index,
projection), so a single multipart field can only carry them zipped. The
archive is unpacked, its one layer read with the right attribute encoding, and
converted to GeoJSON in EPSG:4326 like every other format.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from service.infrastructure.geo_ingest import (
    GeoCrsError,
    GeoIngestError,
    geo_file_to_geojson_dict,
    supported_extensions,
)

_ZONES = ["Ж-1", "ОД-2"]


def _layer(crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {"zone": _ZONES, "geometry": [Point(30.3, 59.9), Point(30.4, 60.0)]},
        crs="EPSG:4326",
    )
    return gdf.to_crs(crs)


def _zip_dir(src: Path, archive: Path, prefix: str = "") -> Path:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.iterdir()):
            zf.write(path, f"{prefix}{path.name}")
    return archive


def _assert_converted(fc: dict) -> None:
    assert fc["type"] == "FeatureCollection"
    assert [f["properties"]["zone"] for f in fc["features"]] == _ZONES
    x, y = fc["features"][0]["geometry"]["coordinates"]
    assert abs(x - 30.3) < 0.01 and abs(y - 59.9) < 0.01


def _write_shapefile(
    folder: Path, encoding: str, keep_cpg: bool, stem: str = "zones"
) -> None:
    folder.mkdir(exist_ok=True)
    _layer().to_file(folder / f"{stem}.shp", encoding=encoding)
    if not keep_cpg:
        (folder / f"{stem}.cpg").unlink(missing_ok=True)
        # Older writers leave the DBF language-driver byte at 0 ("unknown").
        dbf = folder / f"{stem}.dbf"
        data = bytearray(dbf.read_bytes())
        data[29] = 0
        dbf.write_bytes(bytes(data))


def test_zip_is_a_supported_upload_extension() -> None:
    assert ".zip" in supported_extensions()


@pytest.mark.parametrize("encoding", ["cp1251", "utf-8"])
def test_shapefile_without_cpg_keeps_cyrillic(tmp_path: Path, encoding: str) -> None:
    # Without a .cpg GDAL assumes ISO-8859-1 and «Ж-1» comes out as mojibake.
    _write_shapefile(tmp_path / "layer", encoding, keep_cpg=False)
    archive = _zip_dir(tmp_path / "layer", tmp_path / "zones.zip")

    _assert_converted(geo_file_to_geojson_dict(archive))


def test_shapefile_with_cpg_and_uppercase_names(tmp_path: Path) -> None:
    folder = tmp_path / "layer"
    _write_shapefile(folder, "cp1251", keep_cpg=True)
    for path in folder.iterdir():  # MapInfo/ArcGIS exports are often upper case
        path.rename(path.with_name(path.stem.upper() + path.suffix.upper()))
    archive = _zip_dir(folder, tmp_path / "zones.zip", prefix="Зоны/")

    _assert_converted(geo_file_to_geojson_dict(archive))


def test_mapinfo_tab_in_zip(tmp_path: Path) -> None:
    folder = tmp_path / "layer"
    folder.mkdir()
    _layer().to_file(folder / "zones.tab", driver="MapInfo File")
    archive = _zip_dir(folder, tmp_path / "zones.zip")

    _assert_converted(geo_file_to_geojson_dict(archive))


def test_same_layer_as_tab_and_mif_counts_once(tmp_path: Path) -> None:
    folder = tmp_path / "layer"
    folder.mkdir()
    _layer().to_file(folder / "zones.tab", driver="MapInfo File")
    _layer().to_file(folder / "zones.mif", driver="MapInfo File")
    archive = _zip_dir(folder, tmp_path / "zones.zip")

    _assert_converted(geo_file_to_geojson_dict(archive))


def test_unpacked_files_are_cleaned_up(tmp_path: Path) -> None:
    _write_shapefile(tmp_path / "layer", "utf-8", keep_cpg=True)
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    archive = _zip_dir(tmp_path / "layer", upload_dir / "zones.zip")

    geo_file_to_geojson_dict(archive)

    assert [p.name for p in upload_dir.iterdir()] == ["zones.zip"]


def test_several_layers_are_refused_by_name(tmp_path: Path) -> None:
    folder = tmp_path / "layer"
    _write_shapefile(folder, "utf-8", keep_cpg=True, stem="zones")
    _write_shapefile(folder, "utf-8", keep_cpg=True, stem="parcels")
    archive = _zip_dir(folder, tmp_path / "both.zip")

    with pytest.raises(GeoIngestError, match="несколько слоёв") as exc:
        geo_file_to_geojson_dict(archive)
    assert "zones.shp" in str(exc.value) and "parcels.shp" in str(exc.value)


def test_archive_without_a_layer_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("regulations.docx", b"not a layer")
        zf.writestr("zones.dbf", b"attributes without geometry")

    with pytest.raises(GeoIngestError, match="не найден слой"):
        geo_file_to_geojson_dict(archive)


def test_broken_zip_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"PK not really")

    with pytest.raises(GeoIngestError, match="ZIP"):
        geo_file_to_geojson_dict(archive)


def test_zip_slip_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.geojson", b"{}")

    with pytest.raises(GeoIngestError, match="недопустимый путь"):
        geo_file_to_geojson_dict(archive)
    assert not (tmp_path.parent / "escaped.geojson").exists()


_NONEARTH_MIF = """Version 300
Charset "WindowsCyrillic"
Delimiter ","
CoordSys NonEarth Units "m" Bounds (0, 0) (100000, 100000)
Columns 1
  zone Char(10)
Data

Point 1000 2000
"""


def test_mapinfo_nonearth_is_refused_with_instructions(tmp_path: Path) -> None:
    folder = tmp_path / "layer"
    folder.mkdir()
    (folder / "zones.mif").write_text(_NONEARTH_MIF, encoding="cp1251")
    (folder / "zones.mid").write_text('"Ж-1"\n', encoding="cp1251")
    archive = _zip_dir(folder, tmp_path / "zones.zip")

    with pytest.raises(GeoCrsError) as exc:
        geo_file_to_geojson_dict(archive)
    message = str(exc.value)
    assert "NonEarth" in message
    assert "EPSG:4326" in message and "QGIS" in message


def test_empty_mapinfo_layer_is_refused_not_crashed(tmp_path: Path) -> None:
    # MapInfo template sets ship hundreds of empty layers; GDAL reads them as a
    # plain DataFrame without a geometry column.
    folder = tmp_path / "layer"
    folder.mkdir()
    _layer().iloc[:0].to_file(folder / "zones.tab", driver="MapInfo File")
    archive = _zip_dir(folder, tmp_path / "zones.zip")

    with pytest.raises(GeoIngestError, match="пустой"):
        geo_file_to_geojson_dict(archive)
