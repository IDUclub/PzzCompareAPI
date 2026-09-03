from __future__ import annotations

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
        overlay_part = gpd.overlay(parcels_part, pzz_metric, how='intersection', keep_geom_type=False)
        if overlay_part.empty:
            continue
        dissolve_agg: dict[str, str] = {'__parcel_size__': 'first'}
        if zone_name_col and zone_name_col in overlay_part.columns:
            dissolve_agg[zone_name_col] = 'first'
        overlay_part = overlay_part.dissolve(
            by=['__cad_id__', zone_code_col],
            as_index=False,
            aggfunc=dissolve_agg,
        )
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
    rows: list[dict[str, Any]] = []
    for cad_id, group_df in overlay.groupby('__cad_id__'):
        size_by_zone = group_df.groupby(zone_code_col, sort=False)['__intersection_size__'].sum().sort_values(ascending=False, kind='stable')
        dominant_code = size_by_zone.index[0]
        actual_code = normalize_text(dominant_code)
        dominant_area = float(size_by_zone.iloc[0])
        if zone_name_col and zone_name_col in group_df.columns:
            dominant_rows = group_df.loc[group_df[zone_code_col] == dominant_code].sort_values('__intersection_size__', ascending=False)
            actual_name = normalize_text(dominant_rows.iloc[0].get(zone_name_col))
        else:
            actual_name = ''
        parcel_area = float(group_df['__parcel_size__'].iloc[0])
        dominant_share = dominant_area / parcel_area if parcel_area and (not np.isnan(parcel_area)) and (parcel_area > 0) else np.nan
        intersect_codes = collect_unique_codes(size_by_zone.index.tolist())
        note = ''
        if len(intersect_codes) > 1 and pd.notna(dominant_share) and (dominant_share < DOMINANT_PZZ_MIN_SHARE):
            note = 'Dominant zone share is below threshold; parcel intersects multiple PZZ zones.'
        elif len(intersect_codes) > 1:
            note = 'Parcel intersects multiple PZZ zones.'
        rows.append({'__cad_id__': cad_id, 'PZZ_ACTUAL_CODE': actual_code or pd.NA, 'PZZ_ACTUAL_NAME': actual_name or pd.NA, 'PZZ_INTERSECT_CODES': ' | '.join(intersect_codes) if intersect_codes else pd.NA, 'PZZ_INTERSECT_COUNT': len(intersect_codes), 'PZZ_ACTUAL_INTERSECTION_AREA': dominant_area, 'PZZ_ACTUAL_SHARE': dominant_share, 'PZZ_SPATIAL_NOTE': note or pd.NA})
    result_df = pd.DataFrame(rows)
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
    source_with_spatial_gdf["__actual_zone_key__"] = source_with_spatial_gdf.apply(
        lambda row: build_actual_zone_key(vri_text=row.get(vri_col), actual_code=row.get("PZZ_ACTUAL_CODE")),
        axis=1,
    )
    source_with_spatial_gdf["__fallback_key__"] = source_with_spatial_gdf.apply(
        lambda row: build_fallback_key(
            vri_text=row.get(vri_col),
            actual_code=row.get("PZZ_ACTUAL_CODE"),
            intersect_codes=row.get("PZZ_INTERSECT_CODES"),
        ),
        axis=1,
    )
    source_with_spatial_gdf["__comparison_key__"] = source_with_spatial_gdf["__fallback_key__"]
    return source_with_spatial_gdf
