from __future__ import annotations

import numpy as np

from .common import *
import geopandas as gpd


from shapely.geometry.multipolygon import MultiPolygon

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


def attach_spatial_pzz_attributes(parcels_gdf: gpd.GeoDataFrame, pzz_gdf: gpd.GeoDataFrame, zone_code_col: str='PZZ', zone_name_col: Optional[str]=None) -> pd.DataFrame:
    """Attach dominant factual PZZ attributes to parcels.

    Both input layers must be in EPSG:4326. The pipeline reprojects internally
    to the appropriate UTM zone (via ``estimate_utm_crs``) for overlay and area
    computations.
    """
    _validate_input_crs(parcels_gdf, 'Cadastral parcels layer')
    _validate_input_crs(pzz_gdf, 'PZZ zones layer')
    parcels = parcels_gdf.copy().reset_index(drop=True)
    parcels['__cad_id__'] = np.arange(len(parcels))
    parcels_work = prepare_geometries(parcels[['__cad_id__', 'geometry']].copy(), polygon_only=True)
    if parcels_work.crs is None:
        raise ValueError('Parcels layer has no CRS.')
    keep_cols = [zone_code_col, 'geometry']
    if zone_name_col and zone_name_col in pzz_gdf.columns:
        keep_cols.append(zone_name_col)
    pzz_work = prepare_geometries(pzz_gdf[keep_cols].copy(), target_crs=parcels_work.crs, polygon_only=True)
    print('parcels_work geom types:', parcels_work.geometry.geom_type.value_counts(dropna=False).to_dict())
    print('pzz_work geom types:', pzz_work.geometry.geom_type.value_counts(dropna=False).to_dict())
    area_crs = resolve_area_crs(parcels_work)
    parcels_area = parcels_work[['__cad_id__', 'geometry']].to_crs(area_crs).copy()
    parcels_area['__parcel_area__'] = parcels_area.geometry.area
    parcels_work = parcels_work.merge(parcels_area[['__cad_id__', '__parcel_area__']], on='__cad_id__', how='left')
    overlay = gpd.overlay(parcels_work, pzz_work, how='intersection', keep_geom_type=False)
    if overlay.empty:
        result = parcels[['__cad_id__']].copy()
        result['PZZ_ACTUAL_CODE'] = pd.NA
        result['PZZ_ACTUAL_NAME'] = pd.NA
        result['PZZ_INTERSECT_CODES'] = pd.NA
        result['PZZ_INTERSECT_COUNT'] = 0
        result['PZZ_ACTUAL_INTERSECTION_AREA'] = np.nan
        result['PZZ_ACTUAL_SHARE'] = np.nan
        result['PZZ_SPATIAL_NOTE'] = 'No intersection with PZZ'
        return result
    overlay_area = overlay.to_crs(area_crs).copy()
    overlay_area['__intersection_area__'] = overlay_area.geometry.area
    overlay = overlay.drop(columns=['geometry']).merge(overlay_area[['__cad_id__', zone_code_col, '__intersection_area__']], on=['__cad_id__', zone_code_col], how='left')
    overlay[zone_code_col] = overlay[zone_code_col].map(normalize_text)
    if zone_name_col and zone_name_col in overlay.columns:
        overlay[zone_name_col] = overlay[zone_name_col].map(normalize_text)
    rows: list[dict[str, Any]] = []
    for cad_id, group_df in overlay.groupby('__cad_id__'):
        group_df = group_df.sort_values('__intersection_area__', ascending=False).reset_index(drop=True)
        dominant_row = group_df.iloc[0]
        actual_code = normalize_text(dominant_row.get(zone_code_col))
        actual_name = normalize_text(dominant_row.get(zone_name_col)) if zone_name_col and zone_name_col in group_df.columns else ''
        parcel_area = float(dominant_row.get('__parcel_area__', np.nan))
        dominant_area = float(dominant_row.get('__intersection_area__', np.nan))
        dominant_share = dominant_area / parcel_area if parcel_area and (not np.isnan(parcel_area)) and (parcel_area > 0) else np.nan
        intersect_codes = collect_unique_codes(group_df[zone_code_col].tolist())
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
