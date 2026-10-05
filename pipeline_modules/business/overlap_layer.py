"""Topology checks of the input layers: overlapping parcels and PZZ zones.

A territorial-zone check is only as good as its inputs. Two land parcels that
overlap, two zones that claim the same ground, or a parcel cut by a zone
boundary (a parcel must lie within a single territorial zone, ГрК РФ ст. 30 ч. 4)
are defects of the source data, not of the parcel's permitted use, so they are
reported apart from the ВРИ verdict. With the optional layer of municipal (МО)
boundaries, parcels and zones crossing an МО boundary or sticking out of every
МО, and МО overlapping each other, are reported the same way:

- per parcel, as extra result columns (``OverlapCheckResult.parcel_columns``);
- for the whole task, as a GeoJSON FeatureCollection of the overlap geometries
  with a summary (``OverlapCheckResult.report``), stored as the ``overlaps``
  member of the result GeoJSON and served by ``GET /tasks/{id}/overlaps``.

Digitising noise is not a defect: an overlap counts only when it is larger than
both ``OVERLAP_MIN_AREA_M2`` and ``OVERLAP_MIN_SHARE`` of the smaller object
(for a parcel in several zones — of the parcel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import mapping

from .runtime_settings import OVERLAP_MIN_AREA_M2, OVERLAP_MIN_SHARE
from .spatial_layer import prepare_geometries, resolve_area_crs
from .text_utils import normalize_text

# Key of the report inside the result GeoJSON (a GeoJSON "foreign member").
OVERLAPS_KEY = "overlaps"

KIND_PARCEL_PARCEL = "parcel_parcel"
KIND_ZONE_ZONE = "zone_zone"
KIND_PARCEL_MULTI_ZONE = "parcel_multi_zone"
# Checks against the optional municipal (МО) boundaries layer.
KIND_MO_MO = "mo_mo"
KIND_PARCEL_MULTI_MO = "parcel_multi_mo"
KIND_PARCEL_OUTSIDE_MO = "parcel_outside_mo"
KIND_ZONE_MULTI_MO = "zone_multi_mo"
KIND_ZONE_OUTSIDE_MO = "zone_outside_mo"
KIND_LABELS = {
    KIND_PARCEL_PARCEL: "Наложение земельных участков",
    KIND_ZONE_ZONE: "Наложение территориальных зон",
    KIND_PARCEL_MULTI_ZONE: "Участок в нескольких территориальных зонах",
    KIND_MO_MO: "Наложение границ муниципальных образований",
    KIND_PARCEL_MULTI_MO: "Участок пересекает границу муниципальных образований",
    KIND_PARCEL_OUTSIDE_MO: "Участок за пределами границ муниципальных образований",
    KIND_ZONE_MULTI_MO: "Зона пересекает границу муниципальных образований",
    KIND_ZONE_OUTSIDE_MO: "Зона за пределами границ муниципальных образований",
}

# Per-parcel result columns (source names; postprocess_layer renames them).
COL_INTERSECT_ZONES = "OVERLAP_INTERSECT_ZONES"
COL_ACTUAL_SHARE_PCT = "OVERLAP_ACTUAL_ZONE_SHARE_PCT"
COL_OVERLAP_NOTES = "OVERLAP_NOTES"
# Only present when an МО layer was given.
COL_MO = "OVERLAP_MO"
PARCEL_COLUMNS = (COL_INTERSECT_ZONES, COL_ACTUAL_SHARE_PCT, COL_OVERLAP_NOTES, COL_MO)

# Name columns of an МО layer, by normalised column name, most specific first.
_MO_NAME_COLUMN_HINTS = (
    "наименование_мо", "название_мо", "mo_name", "municipality", "наименование",
    "название", "name", "nazvanie", "naimenovanie", "мо", "oktmo", "октмо",
)

# ЕГРН cadastral number: округ:район:квартал:участок, e.g. 47:01:0101001:11.
_CADASTRAL_NUMBER_RE = r"^\d{2}:\d{2}:\d{6,7}:\d+$"
_CADASTRAL_NUMBER_MIN_MATCH = 0.8
_CADASTRAL_NUMBER_SAMPLE = 500


def _pct(share: float) -> str:
    """Share as a short Russian percentage: 0.784 → «78,4 %»."""
    text = f"{share * 100:.1f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', ',')} %"


def _area(value: float) -> str:
    """Area for a sentence: «1 234 м²» (whole metres, «<1» never occurs here)."""
    return f"{value:,.0f}".replace(",", " ") + " м²"


def _is_significant(area: np.ndarray, base_area: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(base_area > 0, area / base_area, 0.0)
    return (area > OVERLAP_MIN_AREA_M2) & (share > OVERLAP_MIN_SHARE)


def detect_cadastral_number_column(frame: pd.DataFrame) -> Optional[str]:
    """The column holding ЕГРН cadastral numbers, recognised by its values.

    Exports name it anything (``Кадастровый_номер``, ``cad_num``, ``KN``…), but
    the values have a fixed shape, so they are what we look at.
    """
    for column in frame.columns:
        if column == "geometry" or frame[column].dtype.kind not in {"O", "U", "S"}:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        values = values.loc[values != ""].head(_CADASTRAL_NUMBER_SAMPLE)
        if values.empty:
            continue
        if values.str.match(_CADASTRAL_NUMBER_RE).mean() >= _CADASTRAL_NUMBER_MIN_MATCH:
            return column
    return None


def _parcel_labels(parcels: pd.DataFrame) -> list[str]:
    """«№ 5 (КН 47:01:0101001:11)»: the feature index always, the number if known."""
    cad_col = detect_cadastral_number_column(parcels)
    numbers = (
        parcels[cad_col].tolist() if cad_col else [None] * len(parcels)
    )
    labels = []
    for idx, number in enumerate(numbers):
        number = normalize_text(number) if number is not None and pd.notna(number) else ""
        labels.append(f"№ {idx} (КН {number})" if number else f"№ {idx}")
    return labels


def _zone_label(code: str, name: str) -> str:
    if code and name and name != code:
        return f"{code} «{name}»"
    return code or name or "без индекса"


@dataclass
class LayerOverlap:
    """One overlapping pair inside a layer (``left`` < ``right`` row positions)."""

    left: int
    right: int
    area_m2: float
    share_of_smaller: float
    geometry: Any  # in the metric CRS


def _metric_polygons(gdf: gpd.GeoDataFrame, metric_crs: Any) -> gpd.GeoDataFrame:
    """Valid polygonal geometries in ``metric_crs``; ``__pos__`` = source row."""
    work = gpd.GeoDataFrame(
        {"__pos__": np.arange(len(gdf))},
        geometry=gdf.geometry.to_numpy(),
        crs=gdf.crs,
    )
    return prepare_geometries(work, target_crs=metric_crs, polygon_only=True)


def find_layer_overlaps(metric: gpd.GeoDataFrame) -> list[LayerOverlap]:
    """Pairs of polygons of one layer sharing more than a sliver of area.

    ``metric`` comes from ``_metric_polygons``. Touching boundaries and numeric
    slivers are ignored (see the module docstring).
    """
    if len(metric) < 2:
        return []
    geoms = metric.geometry.to_numpy()
    positions = metric["__pos__"].to_numpy()
    left_idx, right_idx = metric.sindex.query(geoms, predicate="intersects")
    keep = left_idx < right_idx
    left_idx, right_idx = left_idx[keep], right_idx[keep]
    if len(left_idx) == 0:
        return []
    intersections = shapely.intersection(geoms[left_idx], geoms[right_idx])
    areas = shapely.area(intersections)
    own_areas = shapely.area(geoms)
    smaller = np.minimum(own_areas[left_idx], own_areas[right_idx])
    return [
        LayerOverlap(
            left=int(positions[left_idx[k]]),
            right=int(positions[right_idx[k]]),
            area_m2=float(areas[k]),
            share_of_smaller=float(areas[k] / smaller[k]),
            geometry=intersections[k],
        )
        for k in np.flatnonzero(_is_significant(areas, smaller))
    ]


@dataclass
class _ZoneShare:
    code: str
    area_m2: float
    share: float
    geometry: Any  # parcel ∩ zones of this code, metric CRS


def _parcel_zone_shares(
    parcels: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, zone_codes: list[str]
) -> dict[int, list[_ZoneShare]]:
    """Significant zone pieces of every parcel, largest first, by parcel position."""
    if parcels.empty or zones.empty:
        return {}
    parcel_geoms = parcels.geometry.to_numpy()
    zone_geoms = zones.geometry.to_numpy()
    parcel_idx, zone_idx = zones.sindex.query(parcel_geoms, predicate="intersects")
    if len(parcel_idx) == 0:
        return {}
    pieces = shapely.intersection(parcel_geoms[parcel_idx], zone_geoms[zone_idx])
    frame = pd.DataFrame(
        {
            "parcel": parcels["__pos__"].to_numpy()[parcel_idx],
            "code": [zone_codes[p] for p in zones["__pos__"].to_numpy()[zone_idx]],
            "area": shapely.area(pieces),
            "parcel_area": shapely.area(parcel_geoms)[parcel_idx],
            "piece": pieces,
        }
    )
    frame = frame.loc[frame["code"] != ""]
    shares: dict[int, list[_ZoneShare]] = {}
    for (parcel, code), group in frame.groupby(["parcel", "code"], sort=False):
        area = float(group["area"].sum())
        parcel_area = float(group["parcel_area"].iloc[0])
        if not _is_significant(np.array([area]), np.array([parcel_area]))[0]:
            continue
        geometry = (
            group["piece"].iloc[0]
            if len(group) == 1
            else shapely.union_all(group["piece"].to_numpy())
        )
        shares.setdefault(int(parcel), []).append(
            _ZoneShare(code, area, area / parcel_area, geometry)
        )
    for items in shares.values():
        items.sort(key=lambda item: item.area_m2, reverse=True)
    return shares


def detect_mo_name_column(frame: pd.DataFrame) -> Optional[str]:
    """The column naming the municipalities of an МО layer, or ``None``.

    Tried by column name first; failing that, the first text column whose
    values tell the features apart.
    """
    text_columns = [
        column
        for column in frame.columns
        if column != "geometry" and frame[column].dtype.kind in {"O", "U", "S"}
    ]
    by_key = {str(column).strip().lower().replace(" ", "_"): column for column in text_columns}
    for hint in _MO_NAME_COLUMN_HINTS:
        if hint in by_key and frame[by_key[hint]].notna().any():
            return by_key[hint]
    for column in text_columns:
        values = frame[column].dropna().astype(str).str.strip()
        values = values.loc[values != ""]
        if not values.empty and values.nunique() == len(values) == frame[column].size:
            return column
    return None


def _mo_labels(mo_gdf: pd.DataFrame, name_col: Optional[str]) -> list[str]:
    """«МО «Холмский городской округ»», or «МО № 2» when the name is unknown."""
    labels = []
    for idx in range(len(mo_gdf)):
        value = mo_gdf[name_col].iloc[idx] if name_col else None
        name = normalize_text(value) if value is not None and pd.notna(value) else ""
        labels.append(f"МО «{name}»" if name else f"МО № {idx}")
    return labels


@dataclass
class _MoPlacement:
    """Where one object lies relative to the МО layer."""

    inside: list[_ZoneShare]  # significant pieces per МО (``code`` = МО label)
    outside_area_m2: float
    outside_share: float
    outside_geometry: Any  # metric CRS; empty when nothing is outside


def _mo_placements(
    objects: gpd.GeoDataFrame, mo: gpd.GeoDataFrame, mo_labels: list[str]
) -> dict[int, _MoPlacement]:
    """Objects that straddle an МО boundary or stick out of every МО.

    Keyed by object position (``__pos__``); objects lying wholly inside one
    МО are left out. Both frames come from ``_metric_polygons``.
    """
    if objects.empty or mo.empty:
        return {}
    shares = _parcel_zone_shares(objects, mo, mo_labels)
    object_geoms = objects.geometry.to_numpy()
    object_areas = shapely.area(object_geoms)
    mo_geoms = mo.geometry.to_numpy()
    object_idx, mo_idx = mo.sindex.query(object_geoms, predicate="intersects")
    near: dict[int, list[int]] = {}
    for o, m in zip(object_idx, mo_idx):
        near.setdefault(int(o), []).append(int(m))
    placements: dict[int, _MoPlacement] = {}
    for k, pos in enumerate(objects["__pos__"].to_numpy()):
        pos = int(pos)
        area = float(object_areas[k])
        if area <= 0:
            continue
        neighbours = near.get(k)
        if neighbours:
            outside = shapely.difference(
                object_geoms[k], shapely.union_all(mo_geoms[neighbours])
            )
        else:
            outside = object_geoms[k]
        outside_area = float(shapely.area(outside))
        outside_significant = bool(
            _is_significant(np.array([outside_area]), np.array([area]))[0]
        )
        inside = shares.get(pos, [])
        if len(inside) < 2 and not outside_significant:
            continue
        placements[pos] = _MoPlacement(
            inside=inside,
            outside_area_m2=outside_area if outside_significant else 0.0,
            outside_share=outside_area / area if outside_significant else 0.0,
            outside_geometry=outside if outside_significant else None,
        )
    return placements


def _add_mo_features(
    add_feature: Any,
    placement: _MoPlacement,
    *,
    label: str,
    index_props: dict[str, Any],
    multi_kind: str,
    outside_kind: str,
) -> None:
    """Report features of one object's МО placement: the parts in other МО
    than its main one, and the part outside every МО."""
    inside = placement.inside
    if len(inside) > 1:
        others = inside[1:]
        add_feature(
            multi_kind,
            shapely.union_all([s.geometry for s in others]),
            {
                "Объект_1": label,
                "Объект_2": ", ".join(s.code for s in others),
                **index_props,
                "Основное_МО": inside[0].code,
                "Доля_в_основном_МО_%": round(inside[0].share * 100, 1),
                "Площадь_наложения_м2": round(sum(s.area_m2 for s in others), 2),
                "Доля_наложения_%": round(sum(s.share for s in others) * 100, 2),
            },
        )
    if placement.outside_geometry is not None:
        add_feature(
            outside_kind,
            placement.outside_geometry,
            {
                "Объект_1": label,
                "Объект_2": "Вне границ МО",
                **index_props,
                "Площадь_наложения_м2": round(placement.outside_area_m2, 2),
                "Доля_наложения_%": round(placement.outside_share * 100, 2),
            },
        )


def _mo_note(subject: str, placement: _MoPlacement) -> str:
    """«Участок пересекает границу МО: …; вне границ МО — 120 м² (12 %)»."""
    parts = []
    if len(placement.inside) > 1:
        listed = ", ".join(f"{s.code} ({_pct(s.share)})" for s in placement.inside)
        parts.append(f"{subject} пересекает границу МО: {listed}")
    if placement.outside_geometry is not None:
        if not placement.inside:
            parts.append(f"{subject} расположен вне границ МО")
        else:
            outside = (
                f"{_area(placement.outside_area_m2)} ({_pct(placement.outside_share)})"
            )
            parts.append(
                f"вне границ МО — {outside}" if parts
                else f"{subject} частично вне границ МО: {outside}"
            )
    return "; ".join(parts)


@dataclass
class OverlapCheckResult:
    """Per-parcel result columns and the task-level report."""

    parcel_columns: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        return self.report.get("summary", {})


def run_overlap_checks(
    parcels_gdf: gpd.GeoDataFrame,
    pzz_zones_gdf: gpd.GeoDataFrame,
    *,
    zone_code_col: str,
    zone_name_col: Optional[str] = None,
    mo_gdf: Optional[gpd.GeoDataFrame] = None,
    mo_name_col: Optional[str] = None,
) -> OverlapCheckResult:
    """Check parcels and zones for overlaps.

    Row positions of ``parcels_gdf`` are the result's ``feature_index``; the
    returned ``parcel_columns`` has one row per parcel in the same order. All
    layers are expected in EPSG:4326, like everywhere in the pipeline.

    With ``mo_gdf`` (municipal boundaries) parcels and zones are also checked
    against the МО boundaries, and МО against each other; ``mo_name_col`` is
    detected when not given.
    """
    check_mo = mo_gdf is not None
    mo_gdf = mo_gdf.reset_index(drop=True) if check_mo else None
    if check_mo and (mo_name_col is None or mo_name_col not in mo_gdf.columns):
        mo_name_col = detect_mo_name_column(mo_gdf)
    mo_labels = _mo_labels(mo_gdf, mo_name_col) if check_mo else []
    parcels_gdf = parcels_gdf.reset_index(drop=True)
    zones_gdf = pzz_zones_gdf.reset_index(drop=True)
    if zone_name_col == zone_code_col or (
        zone_name_col and zone_name_col not in zones_gdf.columns
    ):
        zone_name_col = None
    zone_codes = [normalize_text(v) if pd.notna(v) else "" for v in zones_gdf[zone_code_col]]
    zone_names = (
        [normalize_text(v) if pd.notna(v) else "" for v in zones_gdf[zone_name_col]]
        if zone_name_col
        else [""] * len(zones_gdf)
    )
    parcel_labels = _parcel_labels(parcels_gdf)

    notes: list[list[str]] = [[] for _ in range(len(parcels_gdf))]
    intersect_zones: list[Optional[str]] = [None] * len(parcels_gdf)
    actual_share_pct: list[Optional[float]] = [None] * len(parcels_gdf)
    features: list[dict[str, Any]] = []
    parcel_pairs: list[LayerOverlap] = []
    zone_pairs: list[LayerOverlap] = []
    multi_zone: dict[int, list[_ZoneShare]] = {}
    mo_column: list[Optional[str]] = [None] * len(parcels_gdf)
    mo_pairs: list[LayerOverlap] = []
    parcel_mo: dict[int, _MoPlacement] = {}
    zone_mo: dict[int, _MoPlacement] = {}

    reference = prepare_geometries(
        (parcels_gdf if not parcels_gdf.empty else zones_gdf)[["geometry"]].copy()
    )
    metric_crs = resolve_area_crs(reference) if not reference.empty else None

    if metric_crs is not None:
        parcels_metric = _metric_polygons(parcels_gdf, metric_crs)
        zones_metric = _metric_polygons(zones_gdf, metric_crs)

        def add_feature(kind: str, geometry: Any, props: dict[str, Any]) -> None:
            wgs = gpd.GeoSeries([geometry], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(wgs),
                    "properties": {
                        "kind": kind,
                        "Тип_наложения": KIND_LABELS[kind],
                        **props,
                    },
                }
            )

        # 1. Parcels overlapping each other.
        parcel_pairs = find_layer_overlaps(parcels_metric)
        for pair in parcel_pairs:
            share = _pct(pair.share_of_smaller)
            for own, other in ((pair.left, pair.right), (pair.right, pair.left)):
                notes[own].append(
                    f"Наложение на участок {parcel_labels[other]}: "
                    f"{_area(pair.area_m2)} ({share} меньшего участка)"
                )
            add_feature(
                KIND_PARCEL_PARCEL,
                pair.geometry,
                {
                    "Объект_1": f"Участок {parcel_labels[pair.left]}",
                    "Объект_2": f"Участок {parcel_labels[pair.right]}",
                    "feature_index_1": pair.left,
                    "feature_index_2": pair.right,
                    "Площадь_наложения_м2": round(pair.area_m2, 2),
                    "Доля_наложения_%": round(pair.share_of_smaller * 100, 2),
                },
            )

        # 2. PZZ zones overlapping each other.
        zone_pairs = find_layer_overlaps(zones_metric)
        for pair in zone_pairs:
            add_feature(
                KIND_ZONE_ZONE,
                pair.geometry,
                {
                    "Объект_1": "Зона "
                    + _zone_label(zone_codes[pair.left], zone_names[pair.left]),
                    "Объект_2": "Зона "
                    + _zone_label(zone_codes[pair.right], zone_names[pair.right]),
                    "zone_index_1": pair.left,
                    "zone_index_2": pair.right,
                    "Площадь_наложения_м2": round(pair.area_m2, 2),
                    "Доля_наложения_%": round(pair.share_of_smaller * 100, 2),
                },
            )

        # 3. Parcels cut by a zone boundary.
        for pos, shares in _parcel_zone_shares(
            parcels_metric, zones_metric, zone_codes
        ).items():
            actual_share_pct[pos] = round(shares[0].share * 100, 1)
            if len(shares) < 2:
                continue
            multi_zone[pos] = shares
            listed = ", ".join(f"{s.code} ({_pct(s.share)})" for s in shares)
            intersect_zones[pos] = listed
            notes[pos].insert(0, f"Участок расположен в нескольких зонах: {listed}")
            outside = shares[1:]
            outside_area = sum(s.area_m2 for s in outside)
            add_feature(
                KIND_PARCEL_MULTI_ZONE,
                shapely.union_all([s.geometry for s in outside]),
                {
                    "Объект_1": f"Участок {parcel_labels[pos]}",
                    "Объект_2": "Зоны " + ", ".join(s.code for s in outside),
                    "feature_index_1": pos,
                    "Основная_зона": shares[0].code,
                    "Доля_в_основной_зоне_%": actual_share_pct[pos],
                    "Площадь_наложения_м2": round(outside_area, 2),
                    "Доля_наложения_%": round(
                        sum(s.share for s in outside) * 100, 2
                    ),
                },
            )

        if check_mo:
            mo_metric = _metric_polygons(mo_gdf, metric_crs)

            # 4. МО overlapping each other.
            mo_pairs = find_layer_overlaps(mo_metric)
            for pair in mo_pairs:
                add_feature(
                    KIND_MO_MO,
                    pair.geometry,
                    {
                        "Объект_1": mo_labels[pair.left],
                        "Объект_2": mo_labels[pair.right],
                        "mo_index_1": pair.left,
                        "mo_index_2": pair.right,
                        "Площадь_наложения_м2": round(pair.area_m2, 2),
                        "Доля_наложения_%": round(pair.share_of_smaller * 100, 2),
                    },
                )

            # 5. Parcels across an МО boundary or outside every МО.
            parcel_mo = _mo_placements(parcels_metric, mo_metric, mo_labels)
            for pos, placement in parcel_mo.items():
                label = f"Участок {parcel_labels[pos]}"
                pieces = [f"{s.code} ({_pct(s.share)})" for s in placement.inside]
                if placement.outside_geometry is not None:
                    pieces.append(f"вне МО ({_pct(placement.outside_share)})")
                mo_column[pos] = ", ".join(pieces)
                _add_mo_features(
                    add_feature,
                    placement,
                    label=label,
                    index_props={"feature_index_1": pos},
                    multi_kind=KIND_PARCEL_MULTI_MO,
                    outside_kind=KIND_PARCEL_OUTSIDE_MO,
                )
                notes[pos].append(_mo_note("Участок", placement))

            # 6. Zones across an МО boundary or outside every МО.
            zone_mo = _mo_placements(zones_metric, mo_metric, mo_labels)
            for pos, placement in zone_mo.items():
                _add_mo_features(
                    add_feature,
                    placement,
                    label="Зона " + _zone_label(zone_codes[pos], zone_names[pos]),
                    index_props={"zone_index_1": pos},
                    multi_kind=KIND_ZONE_MULTI_MO,
                    outside_kind=KIND_ZONE_OUTSIDE_MO,
                )

            # Parcels wholly inside one МО still get its name in the column.
            for pos, shares in _parcel_zone_shares(
                parcels_metric, mo_metric, mo_labels
            ).items():
                if mo_column[pos] is None and len(shares) == 1:
                    mo_column[pos] = shares[0].code

    columns = {
        COL_INTERSECT_ZONES: intersect_zones,
        COL_ACTUAL_SHARE_PCT: actual_share_pct,
        COL_OVERLAP_NOTES: ["; ".join(n) if n else None for n in notes],
    }
    if check_mo:
        columns[COL_MO] = mo_column
    parcel_columns = pd.DataFrame(columns)
    involved = {p.left for p in parcel_pairs} | {p.right for p in parcel_pairs}
    summary = {
        "parcel_overlaps": len(parcel_pairs),
        "parcels_with_overlaps": len(involved),
        "parcels_in_multiple_zones": len(multi_zone),
        "zone_overlaps": len(zone_pairs),
        "min_area_m2": OVERLAP_MIN_AREA_M2,
        "min_share": OVERLAP_MIN_SHARE,
        "mo_checked": check_mo,
    }
    if check_mo:
        summary.update(
            {
                "mo_overlaps": len(mo_pairs),
                "parcels_in_multiple_mo": sum(
                    len(p.inside) > 1 for p in parcel_mo.values()
                ),
                "parcels_outside_mo": sum(
                    p.outside_geometry is not None for p in parcel_mo.values()
                ),
                "zones_in_multiple_mo": sum(len(p.inside) > 1 for p in zone_mo.values()),
                "zones_outside_mo": sum(
                    p.outside_geometry is not None for p in zone_mo.values()
                ),
            }
        )
    report = {"summary": summary, "type": "FeatureCollection", "features": features}
    return OverlapCheckResult(parcel_columns=parcel_columns, report=report)

