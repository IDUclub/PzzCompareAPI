"""Convert uploaded geo files to a GeoJSON FeatureCollection.

The pipeline consumes GeoJSON FeatureCollections in EPSG:4326. We accept any
geopandas/pyogrio-readable vector format on upload (GeoPackage, GML, KML,
GeoParquet, …) but always persist GeoJSON, so the worker path is unchanged.

``geopandas`` is imported lazily so the API process doesn't pay its import
cost unless a non-GeoJSON upload actually needs conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Already GeoJSON — streamed + validated as JSON, no conversion.
GEOJSON_EXTENSIONS = {".geojson", ".json"}
# Converted to GeoJSON via geopandas.
GEO_VECTOR_EXTENSIONS = {".gpkg", ".gml", ".kml", ".parquet", ".geoparquet"}

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
            f"слой объявлен в системе координат «{declared}». Принимаются только "
            "данные в WGS 84 (EPSG:4326) — перевыгрузите слой в этой системе "
            "координат."
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
        f"координаты выходят за пределы WGS 84: встречена точка [{x:.2f}, {y:.2f}], "
        "тогда как долгота не может превышать 180°, а широта — 90°. Похоже, слой "
        "выгружен в метрах местной проекции. Принимаются только данные в "
        "WGS 84 (EPSG:4326) — перевыгрузите слой в этой системе координат."
    )


def supported_extensions() -> set[str]:
    """All upload extensions we accept (GeoJSON + convertible vector formats)."""
    return GEOJSON_EXTENSIONS | GEO_VECTOR_EXTENSIONS


def is_geojson_filename(filename: str | None) -> bool:
    """True when the file should be treated as GeoJSON (no conversion).

    Missing/unknown extension is treated as GeoJSON to preserve the previous
    behaviour (uploads were assumed to be GeoJSON regardless of name).
    """
    suffix = Path(filename or "").suffix.lower()
    return suffix == "" or suffix in GEOJSON_EXTENSIONS


def geo_file_to_geojson_dict(path: Path) -> dict[str, Any]:
    """Read a geo vector file and return a GeoJSON FeatureCollection (EPSG:4326)."""
    import geopandas as gpd  # lazy: heavy import

    suffix = path.suffix.lower()
    try:
        if suffix in {".parquet", ".geoparquet"}:
            gdf = gpd.read_parquet(path)
        else:
            gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 — surface a clean 4xx upstream
        raise GeoIngestError(f"could not read geo file: {exc}") from exc

    if gdf.crs is not None:
        try:
            gdf = gdf.to_crs(_TARGET_CRS)
        except Exception as exc:  # noqa: BLE001
            raise GeoIngestError(
                f"could not reproject to {_TARGET_CRS}: {exc}"
            ) from exc

    feature_collection = json.loads(gdf.to_json())
    if not isinstance(feature_collection, dict) or "features" not in feature_collection:
        raise GeoIngestError("converted result is not a GeoJSON FeatureCollection")
    # ``to_crs`` above runs only when the source declared a CRS; a file that
    # declared none is still in its original units here.
    ensure_wgs84(feature_collection)
    return feature_collection
