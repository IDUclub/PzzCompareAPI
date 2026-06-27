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
            raise GeoIngestError(f"could not reproject to {_TARGET_CRS}: {exc}") from exc

    feature_collection = json.loads(gdf.to_json())
    if not isinstance(feature_collection, dict) or "features" not in feature_collection:
        raise GeoIngestError("converted result is not a GeoJSON FeatureCollection")
    return feature_collection
