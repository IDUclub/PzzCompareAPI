from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd


@dataclass(frozen=True)
class EffectiveReferencePaths:
    pzz_zone_labels_path: Path
    vri_classifier_path: Path
    services_hierarchy_path: Path
    physical_objects_hierarchy_path: Path


class InputDataLoader:
    """Loads runtime GeoJSON inputs and converts them to GeoDataFrame."""

    @staticmethod
    def load_geojson_to_gdf(path: str | Path) -> gpd.GeoDataFrame:
        return gpd.read_file(Path(path))


class ReferenceDataProvider:
    """Resolves reference files from optional overrides or repository defaults."""

    def __init__(self) -> None:
        self._defaults = {
            "pzz_zone_labels": "data/pzz_zone_llm_labels_template.json",
            "vri_classifier": "data/rosreestr_vri_classifier_2024_12_24.json",
            "services_hierarchy": "data/services_hierarchy.json",
            "physical_hierarchy": "data/physical_objects_hierarchy.json",
        }
        try:
            from service.settings import get_settings

            settings = get_settings()
            self._defaults.update(
                {
                    "pzz_zone_labels": settings.default_pzz_zone_labels_path,
                    "vri_classifier": settings.default_vri_classifier_path,
                    "services_hierarchy": settings.default_services_hierarchy_path,
                    "physical_hierarchy": settings.default_physical_objects_hierarchy_path,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    def resolve_paths(
        self,
        *,
        pzz_zone_labels_override_path: str = "",
        vri_classifier_override_path: str = "",
        services_hierarchy_override_path: str = "",
        physical_objects_hierarchy_override_path: str = "",
    ) -> EffectiveReferencePaths:
        return EffectiveReferencePaths(
            pzz_zone_labels_path=self._pick_path(
                override_path=pzz_zone_labels_override_path,
                default_path=self._defaults["pzz_zone_labels"],
            ),
            vri_classifier_path=self._pick_path(
                override_path=vri_classifier_override_path,
                default_path=self._defaults["vri_classifier"],
            ),
            services_hierarchy_path=self._pick_path(
                override_path=services_hierarchy_override_path,
                default_path=self._defaults["services_hierarchy"],
            ),
            physical_objects_hierarchy_path=self._pick_path(
                override_path=physical_objects_hierarchy_override_path,
                default_path=self._defaults["physical_hierarchy"],
            ),
        )

    @staticmethod
    def _pick_path(*, override_path: str, default_path: str) -> Path:
        chosen = Path(override_path).expanduser() if override_path.strip() else Path(default_path)
        resolved = chosen.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Reference dataset not found: {resolved}")
        return resolved

    @staticmethod
    def load_json(path: str | Path) -> Any:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
