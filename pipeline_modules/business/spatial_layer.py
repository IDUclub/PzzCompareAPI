from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .common import *
import geopandas as gpd


from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union

def prepare_geometries(gdf: gpd.GeoDataFrame, target_crs: Optional[Any]=None, polygon_only: bool=False) -> gpd.GeoDataFrame:
    """
    Clean invalid or empty geometries, optionally keep only polygonal geometries,
    and optionally reproject them.
    """
    prepared = gdf.copy()
    if not isinstance(prepared, gpd.GeoDataFrame):
        prepared = gpd.GeoDataFrame(prepared, geometry='geometry', crs=getattr(gdf, 'crs', None))
    if prepared.crs is None and target_crs is not None:
        raise ValueError('Input GeoDataFrame has no CRS, so it cannot be reprojected.')
    prepared = prepared.loc[prepared.geometry.notna() & ~prepared.geometry.is_empty].copy()
    if hasattr(prepared.geometry, 'make_valid'):
        prepared['geometry'] = prepared.geometry.make_valid()
    else:
        prepared['geometry'] = prepared.geometry.buffer(0)
    prepared = prepared.loc[prepared.geometry.notna() & ~prepared.geometry.is_empty].copy()
    if polygon_only:
        prepared['geometry'] = prepared.geometry.apply(extract_polygonal_geometry)
        prepared = prepared.loc[prepared.geometry.notna() & ~prepared.geometry.is_empty].copy()
        geom_types = set(prepared.geometry.geom_type.dropna().unique().tolist())
        allowed_geom_types = {'Polygon', 'MultiPolygon'}
        prepared = prepared.loc[prepared.geometry.geom_type.isin(allowed_geom_types)].copy()
    if target_crs is not None and prepared.crs != target_crs:
        prepared = prepared.to_crs(target_crs)
    return prepared

def resolve_area_crs(gdf: gpd.GeoDataFrame) -> Any:
    """Resolve projected CRS for area calculations."""
    if gdf.crs is None:
        raise ValueError('Input GeoDataFrame has no CRS.')
    if not gdf.crs.is_geographic:
        return gdf.crs
    estimated = gdf.estimate_utm_crs()
    if estimated is None:
        raise ValueError('Failed to estimate projected CRS.')
    return estimated

def extract_polygonal_geometry(geom):
    """
    Keep only polygonal part of a geometry.

    Parameters
    ----------
    geom : BaseGeometry
        Input shapely geometry.

    Returns
    -------
    BaseGeometry | None
        Polygon or MultiPolygon geometry, or None if no polygonal part exists.
    """
    if geom is None or geom.is_empty:
        return None
    geom_type = geom.geom_type
    if geom_type in {'Polygon', 'MultiPolygon'}:
        return geom
    if geom_type == 'GeometryCollection':
        polygon_parts = [part for part in geom.geoms if part is not None and (not part.is_empty) and (part.geom_type in {'Polygon', 'MultiPolygon'})]
        if not polygon_parts:
            return None
        if len(polygon_parts) == 1:
            return polygon_parts[0]
        flattened_parts = []
        for part in polygon_parts:
            if part.geom_type == 'Polygon':
                flattened_parts.append(part)
            elif part.geom_type == 'MultiPolygon':
                flattened_parts.extend(list(part.geoms))
        if not flattened_parts:
            return None
        return MultiPolygon(flattened_parts)
    return None

_EXPECTED_INPUT_EPSG = 4326


def _validate_input_crs(gdf: gpd.GeoDataFrame, layer_name: str) -> None:
    """Ensure incoming GeoDataFrame is in EPSG:4326 (WGS84).

    The pipeline contract requires clients to upload geometries in EPSG:4326.
    Internally we reproject to the local UTM zone via ``estimate_utm_crs``,
    which assumes the input is in geographic coordinates. Accepting other
    CRSes leads to silently wrong area calculations or zone-estimation
    failures, so we fail fast with a clear message.
    """
    if gdf.crs is None:
        raise ValueError(
            f"{layer_name} has no CRS. EPSG:{_EXPECTED_INPUT_EPSG} expected."
        )
    epsg = gdf.crs.to_epsg()
    if epsg != _EXPECTED_INPUT_EPSG:
        raise ValueError(
            f"{layer_name} must be in EPSG:{_EXPECTED_INPUT_EPSG} (WGS84), "
            f"got EPSG:{epsg}."
        )


_POLYGON_GEOM_TYPES = {'Polygon', 'MultiPolygon'}
_LINE_GEOM_TYPES = {'LineString', 'MultiLineString', 'LinearRing'}
_POINT_GEOM_TYPES = {'Point', 'MultiPoint'}


def reduce_to_single_geometry_family(geom):
    """Collapse a GeometryCollection to its highest-dimension parts.

    ``gpd.overlay`` rejects frames that mix geometry families, and ``make_valid``
    can turn a self-intersecting input into a GeometryCollection, so such
    geometries have to be reduced before any overlay is attempted.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type != 'GeometryCollection':
        return geom
    for family in (_POLYGON_GEOM_TYPES, _LINE_GEOM_TYPES, _POINT_GEOM_TYPES):
        parts = [part for part in geom.geoms if part is not None and (not part.is_empty) and (part.geom_type in family)]
        if not parts:
            continue
        if len(parts) == 1:
            return parts[0]
        flattened = []
        for part in parts:
            if part.geom_type.startswith('Multi'):
                flattened.extend(list(part.geoms))
            else:
                flattened.append(part)
        return unary_union(flattened)
    return None


def _measure_geometries(geoseries: gpd.GeoSeries, geom_family: str) -> Any:
    """Measure geometries with the metric that is meaningful for their family."""
    if geom_family == 'polygon':
        return geoseries.area
    if geom_family == 'line':
        return geoseries.length
    return geoseries.map(
        lambda geom: float(len(geom.geoms)) if geom.geom_type == 'MultiPoint' else 1.0
    )


def _meaningful_intersection_mask(frame: pd.DataFrame, geom_family: str) -> pd.Series:
    """Exclude boundary-only contacts and insignificant numeric slivers."""
    if geom_family == 'polygon':
        absolute_minimum = SPATIAL_MIN_POLYGON_INTERSECTION_AREA_M2
    elif geom_family == 'line':
        absolute_minimum = SPATIAL_MIN_LINE_INTERSECTION_LENGTH_M
    else:
        absolute_minimum = 0.0
    relative_minimum = frame['__parcel_size__'].astype(float) * SPATIAL_MIN_INTERSECTION_SHARE
    minimum = np.maximum(absolute_minimum, relative_minimum)
    return frame['__intersection_size__'].astype(float) > minimum


def _split_by_geometry_family(gdf: gpd.GeoDataFrame) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Split a layer into polygonal, linear and point subsets."""
    families = (('polygon', _POLYGON_GEOM_TYPES), ('line', _LINE_GEOM_TYPES), ('point', _POINT_GEOM_TYPES))
    parts: list[tuple[str, gpd.GeoDataFrame]] = []
    for geom_family, geom_types in families:
        subset = gdf.loc[gdf.geometry.geom_type.isin(geom_types)]
        if not subset.empty:
            parts.append((geom_family, subset.copy()))
    return parts


def _empty_spatial_result(parcels: pd.DataFrame) -> pd.DataFrame:
    result = parcels[['__cad_id__']].copy()
    result['PZZ_ACTUAL_CODE'] = pd.NA
    result['PZZ_ACTUAL_NAME'] = pd.NA
    result['PZZ_INTERSECT_CODES'] = pd.NA
    result['PZZ_INTERSECT_COUNT'] = 0
    result['PZZ_ACTUAL_INTERSECTION_AREA'] = np.nan
    result['PZZ_ACTUAL_SHARE'] = np.nan
    result['PZZ_SPATIAL_NOTE'] = 'No intersection with PZZ'
    return result


# Below this many parcels a single overlay call is faster than chunking.
_PARALLEL_OVERLAY_MIN_PARCELS = 20_000


def _overlay_intersection(parcels: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Intersect parcels with zones, splitting large layers across threads.

    Each parcel is intersected independently, so chunking the parcels gives the
    same pieces as one call; GEOS releases the GIL, so threads scale.
    """
    workers = min(SPATIAL_JOIN_WORKERS, os.cpu_count() or 1)
    if workers < 2 or len(parcels) < _PARALLEL_OVERLAY_MIN_PARCELS:
        return gpd.overlay(parcels, zones, how='intersection', keep_geom_type=False)
    chunks = np.array_split(np.arange(len(parcels)), workers * 4)

    def _run(positions: np.ndarray) -> gpd.GeoDataFrame:
        return gpd.overlay(parcels.iloc[positions], zones, how='intersection', keep_geom_type=False)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = [part for part in executor.map(_run, chunks) if not part.empty]
    if not parts:
        return _run(chunks[0])
    return pd.concat(parts, ignore_index=True)


def _merge_pieces_per_zone(
    overlay_part: gpd.GeoDataFrame,
    zone_code_col: str,
    zone_name_col: Optional[str],
) -> gpd.GeoDataFrame:
    """Collapse overlay pieces to one geometry per (parcel, zone code).

    Only pairs split into several pieces need a geometric union (overlapping
    features of one zone must not be counted twice); dissolving every pair
    runs a Python-level union per group and dominates large inputs.
    """
    keep_cols = ['__cad_id__', zone_code_col, 'geometry', '__parcel_size__']
    if zone_name_col and zone_name_col in overlay_part.columns:
        keep_cols.append(zone_name_col)
    overlay_part = overlay_part[keep_cols]
    split = overlay_part.duplicated(['__cad_id__', zone_code_col], keep=False)
    if not split.any():
        return overlay_part.copy()
    dissolve_agg = {col: 'first' for col in keep_cols[3:]}
    merged = overlay_part.loc[split].dissolve(
        by=['__cad_id__', zone_code_col],
        as_index=False,
        aggfunc=dissolve_agg,
    )
    return pd.concat(
        [overlay_part.loc[~split], merged[keep_cols]],
        ignore_index=True,
    )


def _summarize_dominant_zones(
    overlay: pd.DataFrame,
    zone_code_col: str,
    zone_name_col: Optional[str],
) -> pd.DataFrame:
    """Pick each parcel's dominant zone from its per-zone intersection sizes.

    Zones are ranked by total intersection size; ties keep zone-code order.
    """
    has_name = bool(zone_name_col and zone_name_col in overlay.columns)
    agg: dict[str, str] = {'__intersection_size__': 'sum', '__parcel_size__': 'first'}
    if has_name:
        agg[zone_name_col] = 'first'
    sizes = overlay.groupby(['__cad_id__', zone_code_col], sort=True, as_index=False).agg(agg)
    sizes = sizes.sort_values(
        ['__cad_id__', '__intersection_size__'],
        ascending=[True, False],
        kind='stable',
    )
    dominant = sizes.drop_duplicates('__cad_id__', keep='first').set_index('__cad_id__')
    # Most parcels lie in one zone: collect code lists only where there are several.
    zone_count = sizes.groupby('__cad_id__', sort=False).size()
    intersect_codes = dominant[zone_code_col].map(lambda code: collect_unique_codes([code]))
    several = sizes.loc[sizes['__cad_id__'].map(zone_count) > 1]
    if not several.empty:
        lists = several.groupby('__cad_id__', sort=False)[zone_code_col].agg(list)
        intersect_codes.loc[lists.index] = lists.map(collect_unique_codes)

    dominant_area = dominant['__intersection_size__'].astype(float)
    parcel_area = dominant['__parcel_size__'].astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dominant_share = (dominant_area / parcel_area).where(parcel_area > 0)
    code_count = intersect_codes.map(len)
    multiple = code_count > 1
    note = np.select(
        [multiple & (dominant_share < DOMINANT_PZZ_MIN_SHARE), multiple],
        [
            'Dominant zone share is below threshold; parcel intersects multiple PZZ zones.',
            'Parcel intersects multiple PZZ zones.',
        ],
        default='',
    )

    def _or_na(values: Any) -> np.ndarray:
        return np.array([value or pd.NA for value in values], dtype=object)

    actual_names = dominant[zone_name_col].map(normalize_text) if has_name else pd.Series('', index=dominant.index)
    return pd.DataFrame(
        {
            '__cad_id__': dominant.index.to_numpy(),
            'PZZ_ACTUAL_CODE': _or_na(dominant[zone_code_col].map(normalize_text)),
            'PZZ_ACTUAL_NAME': _or_na(actual_names),
            'PZZ_INTERSECT_CODES': _or_na(intersect_codes.map(' | '.join)),
            'PZZ_INTERSECT_COUNT': code_count.to_numpy(),
            'PZZ_ACTUAL_INTERSECTION_AREA': dominant_area.to_numpy(),
            'PZZ_ACTUAL_SHARE': dominant_share.to_numpy(),
            'PZZ_SPATIAL_NOTE': _or_na(note.tolist()),
        }
    )


def attach_spatial_pzz_attributes(parcels_gdf: gpd.GeoDataFrame, pzz_gdf: gpd.GeoDataFrame, zone_code_col: str='PZZ', zone_name_col: Optional[str]=None) -> pd.DataFrame:
    """Attach dominant factual PZZ attributes to parcels.

    Parcels of any geometry family are supported: polygonal parcels are compared
    by intersection area, linear ones (roads, utility corridors) by intersection
    length, point ones by containment. The dominant zone is the zone index with
    the largest total overlap, summed over every intersection piece.

    Both input layers must be in EPSG:4326. The pipeline reprojects internally
    to the appropriate UTM zone (via ``estimate_utm_crs``) for overlay and area
    computations.
    """
    _validate_input_crs(parcels_gdf, 'Cadastral parcels layer')
    _validate_input_crs(pzz_gdf, 'PZZ zones layer')
    parcels = parcels_gdf.copy().reset_index(drop=True)
    parcels['__cad_id__'] = np.arange(len(parcels))
    parcels_work = prepare_geometries(parcels[['__cad_id__', 'geometry']].copy())
    parcels_work['geometry'] = parcels_work.geometry.apply(reduce_to_single_geometry_family)
    parcels_work = parcels_work.loc[parcels_work.geometry.notna() & ~parcels_work.geometry.is_empty].copy()
    if parcels_work.crs is None:
        raise ValueError('Parcels layer has no CRS.')
    if zone_name_col == zone_code_col:
        # A zones layer with a single usable column gets it selected as both code
        # and name. Keeping it twice makes ``pzz_work[zone_code_col]`` a DataFrame,
        # and every later selection fails; the name adds nothing here anyway.
        zone_name_col = None
    keep_cols = [zone_code_col, 'geometry']
    if zone_name_col and zone_name_col in pzz_gdf.columns:
        keep_cols.append(zone_name_col)
    pzz_work = prepare_geometries(pzz_gdf[keep_cols].copy(), target_crs=parcels_work.crs, polygon_only=True)
    pzz_work[zone_code_col] = pzz_work[zone_code_col].map(normalize_text)
    pzz_work = pzz_work.loc[pzz_work[zone_code_col] != ''].copy()
    if zone_name_col and zone_name_col in pzz_work.columns:
        pzz_work[zone_name_col] = pzz_work[zone_name_col].map(normalize_text)
    print('parcels_work geom types:', parcels_work.geometry.geom_type.value_counts(dropna=False).to_dict())
    print('pzz_work geom types:', pzz_work.geometry.geom_type.value_counts(dropna=False).to_dict())
    if parcels_work.empty or pzz_work.empty:
        return _empty_spatial_result(parcels)
    area_crs = resolve_area_crs(parcels_work)
    pzz_metric = pzz_work.to_crs(area_crs)
    overlay_frames: list[pd.DataFrame] = []
    for geom_family, parcels_part in _split_by_geometry_family(parcels_work):
        parcels_part = parcels_part.to_crs(area_crs)
        parcels_part['__parcel_size__'] = _measure_geometries(
            parcels_part.geometry,
            geom_family,
        ).to_numpy()
        overlay_part = _overlay_intersection(parcels_part, pzz_metric)
        if overlay_part.empty:
            continue
        overlay_part = _merge_pieces_per_zone(overlay_part, zone_code_col, zone_name_col)
        overlay_part['__intersection_size__'] = _measure_geometries(
            overlay_part.geometry,
            geom_family,
        ).to_numpy()
        overlay_part = overlay_part.loc[
            _meaningful_intersection_mask(overlay_part, geom_family)
        ].copy()
        if overlay_part.empty:
            continue
        overlay_part = overlay_part.drop(columns=['geometry'])
        overlay_frames.append(overlay_part)
    if not overlay_frames:
        return _empty_spatial_result(parcels)
    overlay = pd.concat(overlay_frames, ignore_index=True)
    result_df = _summarize_dominant_zones(overlay, zone_code_col, zone_name_col)
    all_parcels_df = parcels[['__cad_id__']].copy()
    result_df = all_parcels_df.merge(result_df, on='__cad_id__', how='left')
    result_df['PZZ_INTERSECT_COUNT'] = result_df['PZZ_INTERSECT_COUNT'].fillna(0).astype(int)
    missing_intersection = result_df['PZZ_ACTUAL_CODE'].isna()
    result_df.loc[missing_intersection, 'PZZ_SPATIAL_NOTE'] = 'No intersection with PZZ'
    return result_df

def build_source_with_spatial_attributes(
    source_gdf: gpd.GeoDataFrame,
    pzz_zones_gdf: gpd.GeoDataFrame,
    *,
    vri_col: str,
    pzz_zone_code_col: str,
    pzz_zone_name_col: str,
) -> gpd.GeoDataFrame:
    """Attach spatial attributes and build stable keys for downstream matching."""
    spatial_attributes_df = attach_spatial_pzz_attributes(
        parcels_gdf=source_gdf,
        pzz_gdf=pzz_zones_gdf,
        zone_code_col=pzz_zone_code_col,
        zone_name_col=pzz_zone_name_col if pzz_zone_name_col in pzz_zones_gdf.columns else None,
    )
    source_with_spatial_gdf = source_gdf.reset_index(drop=True).copy()
    source_with_spatial_gdf["__cad_id__"] = np.arange(len(source_with_spatial_gdf))
    source_with_spatial_gdf = source_with_spatial_gdf.merge(spatial_attributes_df, on="__cad_id__", how="left")
    source_with_spatial_gdf = source_with_spatial_gdf.drop(columns=["__cad_id__"])
    vri_values = (
        source_with_spatial_gdf[vri_col].tolist()
        if vri_col in source_with_spatial_gdf.columns
        else [None] * len(source_with_spatial_gdf)
    )
    actual_codes = source_with_spatial_gdf["PZZ_ACTUAL_CODE"].tolist()
    intersect_codes = source_with_spatial_gdf["PZZ_INTERSECT_CODES"].tolist()
    source_with_spatial_gdf["__actual_zone_key__"] = [
        build_actual_zone_key(vri_text=vri, actual_code=code)
        for vri, code in zip(vri_values, actual_codes)
    ]
    source_with_spatial_gdf["__fallback_key__"] = [
        build_fallback_key(vri_text=vri, actual_code=code, intersect_codes=codes)
        for vri, code, codes in zip(vri_values, actual_codes, intersect_codes)
    ]
    source_with_spatial_gdf["__comparison_key__"] = source_with_spatial_gdf["__fallback_key__"]
    return source_with_spatial_gdf
