"""Convert uploaded geo files to a GeoJSON FeatureCollection.

The pipeline consumes GeoJSON FeatureCollections in EPSG:4326. We accept any
geopandas/pyogrio-readable vector format on upload (GeoPackage, GML, KML,
GeoParquet, …) but always persist GeoJSON, so the worker path is unchanged.

Multi-file formats — ESRI Shapefile (.shp + .shx + .dbf + .prj + .cpg) and
MapInfo (.tab + .dat + .map + .id, or .mif + .mid) — cannot travel as one
multipart field, so they arrive as a ZIP archive holding a single layer.

``geopandas`` is imported lazily so the API process doesn't pay its import
cost unless a non-GeoJSON upload actually needs conversion.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

# Already GeoJSON — streamed + validated as JSON, no conversion.
GEOJSON_EXTENSIONS = {".geojson", ".json"}
# Converted to GeoJSON via geopandas.
GEO_VECTOR_EXTENSIONS = {".gpkg", ".gml", ".kml", ".parquet", ".geoparquet"}
# A ZIP holding one layer: Shapefile, MapInfo, or any single-file format above.
ARCHIVE_EXTENSIONS = {".zip"}
# The main file of each multi-file layer, in preference order: MapInfo users
# often keep the same layer as both .TAB and .MIF side by side, and the native
# .tab is the one to read.
_ARCHIVE_LAYER_EXTENSIONS = (".tab", ".shp", ".mif")
# Parts of multi-file layers. Uploaded on their own they are unreadable — the
# geometry and the attributes live in different files — so they get a hint to
# zip the whole set instead of a bare "not supported".
MULTIFILE_PART_EXTENSIONS = {
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx",
    ".tab", ".dat", ".map", ".id", ".ind", ".mif", ".mid",
}

# Zip-bomb guard: the upload size is capped before we get here, but a few MB of
# compressed zeros can expand to tens of GB. Real layers compress ~3-10x.
_MAX_UNZIPPED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000

_TARGET_CRS = "EPSG:4326"


class GeoIngestError(ValueError):
    """An uploaded geo file could not be read or converted."""


class GeoCrsError(GeoIngestError):
    """An uploaded layer is not in WGS84 (EPSG:4326) longitude/latitude."""


# GeoJSON is defined in WGS84 lon/lat degrees (RFC 7946). QGIS/MapInfo exports of
# a local or NonEarth projection keep the projected metres and write no ``crs``
# member, so such a file is indistinguishable from a valid one on arrival — the
# pipeline only fails on it minutes later, deep inside geopandas ("Unable to
# determine UTM CRS"). Longitude never exceeds ±180 and latitude ±90, so a single
# out-of-range coordinate is conclusive: reject at the door with a message that
# says what to do. The check never rejects a genuine WGS84 layer (all its
# coordinates are in range by definition); it cannot catch a projection whose
# values happen to stay small, which is why the declared ``crs`` is checked too.
_LON_LIMIT = 180.0
_LAT_LIMIT = 90.0
# Enough features to be certain without walking a 40k-feature layer: a projected
# export is out of range from its very first geometry.
_CRS_SCAN_FEATURES = 200

_WGS84_CRS_NAMES = {
    "epsg:4326",
    "urn:ogc:def:crs:epsg::4326",
    "urn:ogc:def:crs:ogc:1.3:crs84",
    "urn:ogc:def:crs:ogc::crs84",
    "ogc:crs84",
    "crs84",
    "wgs84",
    "wgs 84",
}


def _declared_crs_name(feature_collection: dict[str, Any]) -> str | None:
    """The legacy GeoJSON ``crs`` member's name, when the writer emitted one."""
    crs = feature_collection.get("crs")
    if not isinstance(crs, dict):
        return None
    properties = crs.get("properties")
    name = properties.get("name") if isinstance(properties, dict) else None
    return name if isinstance(name, str) and name.strip() else None


def _first_position(coordinates: Any) -> tuple[float, float] | None:
    """The first ``[x, y]`` pair of an arbitrarily nested coordinates array."""
    node = coordinates
    while (
        isinstance(node, (list, tuple))
        and node
        and isinstance(node[0], (list, tuple))
    ):
        node = node[0]
    if (
        isinstance(node, (list, tuple))
        and len(node) >= 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in node[:2])
    ):
        return float(node[0]), float(node[1])
    return None


def _sample_positions(geometry: Any, out: list[tuple[float, float]]) -> None:
    """Collect one representative position from ``geometry`` into ``out``."""
    if not isinstance(geometry, dict):
        return
    if geometry.get("type") == "GeometryCollection":
        for part in geometry.get("geometries") or []:
            _sample_positions(part, out)
        return
    position = _first_position(geometry.get("coordinates"))
    if position is not None:
        out.append(position)


def ensure_wgs84(feature_collection: dict[str, Any]) -> None:
    """Raise :class:`GeoCrsError` when a layer is not in WGS84 lon/lat degrees.

    Both the declared ``crs`` member (when present) and the actual coordinate
    values are checked, because most offending exports declare nothing at all.
    """
    declared = _declared_crs_name(feature_collection)
    if declared is not None and declared.strip().lower() not in _WGS84_CRS_NAMES:
        raise GeoCrsError(
            f"объявлен в системе координат «{declared}», а сервис принимает "
            "только WGS 84 (EPSG:4326). Перепроецируйте слой: в QGIS — правый клик "
            "по слою → «Экспорт» → «Сохранить объекты как…» → в поле «СК» выбрать "
            "EPSG:4326 (WGS 84)."
        )

    features = feature_collection.get("features")
    if not isinstance(features, list):
        return
    positions: list[tuple[float, float]] = []
    for feature in features[:_CRS_SCAN_FEATURES]:
        if isinstance(feature, dict):
            _sample_positions(feature.get("geometry"), positions)

    for x, y in positions:
        if abs(x) > _LON_LIMIT or abs(y) > _LAT_LIMIT:
            return _raise_out_of_range(x, y)


def _raise_out_of_range(x: float, y: float) -> None:
    raise GeoCrsError(
        "координаты записаны не в WGS 84 (EPSG:4326), а в метрах: встречена "
        f"точка [{x:.2f}, {y:.2f}], тогда как долгота не может превышать 180°, а "
        "широта — 90°. Так выглядит выгрузка из местной системы координат "
        "(MapInfo NonEarth, МСК-* и подобные). Что сделать: открыть слой в QGIS, "
        "указать его исходную СК («Свойства слоя» → «Источник» → «Задать СК»), "
        "затем «Экспорт» → «Сохранить объекты как…» → «СК»: EPSG:4326 (WGS 84). "
        "Просто переподписать слой как EPSG:4326 без перепроецирования нельзя — "
        "объекты уедут в точку с нулевыми координатами."
    )


def supported_extensions() -> set[str]:
    """All upload extensions we accept (GeoJSON + convertible vector formats)."""
    return GEOJSON_EXTENSIONS | GEO_VECTOR_EXTENSIONS | ARCHIVE_EXTENSIONS


def multifile_part_hint(suffix: str) -> str:
    """User-facing hint for a lone part of a multi-file layer, else ``""``."""
    if suffix.lower() not in MULTIFILE_PART_EXTENSIONS:
        return ""
    return (
        " Shapefile (.shp, .shx, .dbf, .prj, .cpg) и MapInfo (.tab, .dat, .map, "
        ".id или .mif, .mid) состоят из нескольких файлов: заархивируйте все "
        "файлы слоя в один ZIP и загрузите архив."
    )


def is_geojson_filename(filename: str | None) -> bool:
    """True when the file should be treated as GeoJSON (no conversion).

    Missing/unknown extension is treated as GeoJSON to preserve the previous
    behaviour (uploads were assumed to be GeoJSON regardless of name).
    """
    suffix = Path(filename or "").suffix.lower()
    return suffix == "" or suffix in GEOJSON_EXTENSIONS


def _extract_archive(archive: Path, target: Path) -> None:
    """Unpack ``archive`` into ``target``, refusing anything unsafe or huge."""
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise GeoIngestError(f"файл не является корректным ZIP-архивом: {exc}") from exc
    with zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) > _MAX_ARCHIVE_MEMBERS:
            raise GeoIngestError(
                f"в архиве слишком много файлов ({len(members)}), ожидается один слой."
            )
        if sum(info.file_size for info in members) > _MAX_UNZIPPED_BYTES:
            raise GeoIngestError(
                "архив слишком велик в распакованном виде "
                f"(больше {_MAX_UNZIPPED_BYTES // 1024 ** 3} ГБ)."
            )
        root = target.resolve()
        for info in members:
            name = PurePosixPath(info.filename.replace("\\", "/"))
            destination = (root / Path(*name.parts)).resolve()
            if not destination.is_relative_to(root):  # zip-slip
                raise GeoIngestError(
                    f"архив содержит недопустимый путь «{info.filename}»."
                )
            # macOS resource forks and hidden files are never part of a layer.
            if any(part == "__MACOSX" or part.startswith(".") for part in name.parts):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
                # Corrupt data, an unsupported compression method, or encryption.
                raise GeoIngestError(
                    f"не удалось распаковать «{info.filename}»: {exc}"
                ) from exc


def _layer_rank(suffix: str) -> int:
    try:
        return _ARCHIVE_LAYER_EXTENSIONS.index(suffix)
    except ValueError:
        return len(_ARCHIVE_LAYER_EXTENSIONS)


def _find_archive_layer(root: Path) -> Path:
    """The single layer file inside an unpacked archive."""
    readable = (
        set(_ARCHIVE_LAYER_EXTENSIONS) | GEO_VECTOR_EXTENSIONS | GEOJSON_EXTENSIONS
    )
    layers: dict[tuple[Path, str], Path] = {}
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in readable:
            continue
        # The same layer saved as .TAB and .MIF counts once, preferring .tab.
        key = (path.parent, path.stem.lower())
        current = layers.get(key)
        if current is None or _layer_rank(suffix) < _layer_rank(current.suffix.lower()):
            layers[key] = path

    if not layers:
        raise GeoIngestError(
            "в архиве не найден слой: ожидается Shapefile (.shp с .shx, .dbf, "
            ".prj), MapInfo (.tab с .dat, .map, .id или .mif с .mid) либо файл "
            + ", ".join(sorted(GEO_VECTOR_EXTENSIONS | GEOJSON_EXTENSIONS))
            + "."
        )
    if len(layers) > 1:
        names = ", ".join(
            f"«{path.relative_to(root).as_posix()}»" for path in layers.values()
        )
        raise GeoIngestError(
            f"в архиве несколько слоёв ({names}), а загружать нужно по одному "
            "слою на архив."
        )
    return next(iter(layers.values()))


def _sibling(path: Path, suffix: str) -> Path | None:
    """``path`` with another extension, matched case-insensitively (.DBF/.dbf)."""
    wanted = f"{path.stem}{suffix}".lower()
    for candidate in path.parent.iterdir():
        if candidate.name.lower() == wanted:
            return candidate
    return None


# How much DBF record data to sample when guessing the encoding.
_DBF_SAMPLE_BYTES = 1024 * 1024


def _shapefile_encoding(shp: Path) -> str | None:
    """Encoding to force for a Shapefile whose DBF does not declare one.

    GDAL honours a ``.cpg`` file and the DBF language-driver byte; with neither
    it falls back to ISO-8859-1, which turns Russian attributes (zone codes,
    ВРИ) into mojibake. Russian Shapefiles without a ``.cpg`` are almost always
    cp1251 (MapInfo/ArcGIS/QGIS on Windows) or UTF-8 (recent QGIS), and the two
    are easy to tell apart: cp1251 Cyrillic is practically never valid UTF-8.
    """
    if _sibling(shp, ".cpg") is not None:
        return None
    dbf = _sibling(shp, ".dbf")
    if dbf is None:
        return None
    with dbf.open("rb") as fh:
        header = fh.read(32)
        if len(header) < 32 or header[29] != 0:
            return None  # a language driver is declared — let GDAL apply it
        fh.seek(int.from_bytes(header[8:10], "little"))
        sample = fh.read(_DBF_SAMPLE_BYTES)
    if sample.isascii():
        return None
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        # The sample may cut a multi-byte character in half at the very end.
        if exc.start < len(sample) - 3:
            return "cp1251"
    return "utf-8"


def _read_vector(path: Path):
    import geopandas as gpd  # lazy: heavy import

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(path)
    if suffix == ".shp":
        encoding = _shapefile_encoding(path)
        if encoding is not None:
            return gpd.read_file(path, encoding=encoding)
    return gpd.read_file(path)


_LOCAL_CRS_MESSAGE = (
    "слой записан в условной системе координат «{name}» (MapInfo NonEarth и "
    "подобные), которую нельзя автоматически пересчитать в WGS 84 (EPSG:4326): "
    "в файле нет привязки к земному эллипсоиду. Что сделать: открыть слой в "
    "QGIS, указать его настоящую СК («Свойства слоя» → «Источник» → «Задать "
    "СК», например нужную зону МСК), затем «Экспорт» → «Сохранить объекты "
    "как…» → «СК»: EPSG:4326 (WGS 84). Если настоящая СК неизвестна, запросите "
    "у поставщика данных выгрузку в WGS 84 или в МСК с параметрами."
)


def geo_file_to_geojson_dict(path: Path) -> dict[str, Any]:
    """Read a geo vector file and return a GeoJSON FeatureCollection (EPSG:4326).

    A ``.zip`` is unpacked next to ``path`` and its single layer is read; the
    unpacked files are removed before returning.
    """
    if path.suffix.lower() in ARCHIVE_EXTENSIONS:
        unpack_dir = Path(tempfile.mkdtemp(prefix=f"{path.stem}_", dir=path.parent))
        try:
            _extract_archive(path, unpack_dir)
            return geo_file_to_geojson_dict(_find_archive_layer(unpack_dir))
        finally:
            shutil.rmtree(unpack_dir, ignore_errors=True)

    import geopandas as gpd  # lazy: heavy import

    try:
        gdf = _read_vector(path)
    except Exception as exc:  # noqa: BLE001 — surface a clean 4xx upstream
        raise GeoIngestError(f"не удалось прочитать файл слоя: {exc}") from exc

    # An empty layer (typical for MapInfo template sets) or an attribute-only
    # table comes back as a plain DataFrame with no geometry column at all.
    if not isinstance(gdf, gpd.GeoDataFrame):
        if gdf.empty:
            raise GeoIngestError(
                f"слой «{path.name}» пустой: в нём нет ни одного объекта."
            )
        raise GeoIngestError(
            f"в слое «{path.name}» нет геометрии — это таблица атрибутов без карты."
        )

    if gdf.crs is not None:
        if gdf.crs.is_engineering:
            raise GeoCrsError(_LOCAL_CRS_MESSAGE.format(name=gdf.crs.name))
        try:
            gdf = gdf.to_crs(_TARGET_CRS)
        except Exception as exc:  # noqa: BLE001
            raise GeoCrsError(
                _LOCAL_CRS_MESSAGE.format(name=gdf.crs.name)
                + f" Ошибка пересчёта: {exc}"
            ) from exc

    feature_collection = json.loads(gdf.to_json())
    if not isinstance(feature_collection, dict) or "features" not in feature_collection:
        raise GeoIngestError("converted result is not a GeoJSON FeatureCollection")
    # ``to_crs`` above runs only when the source declared a CRS; a file that
    # declared none is still in its original units here.
    ensure_wgs84(feature_collection)
    return feature_collection
