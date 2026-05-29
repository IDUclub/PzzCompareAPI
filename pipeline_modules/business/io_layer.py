import os
from datetime import datetime
from pathlib import Path

from .data_loading import ReferenceDataProvider
from .types import PipelinePaths


def build_paths(
    cadastral_geojson: str,
    pzz_zones_geojson: str,
    pzz_zone_vri_labels_path: str,
    vri_classifier_path: str,
    include_pzz_check: bool,
    cadastral_vri_col: str,
    pzz_zone_code_col: str,
    pzz_zone_name_col: str,
    task_external_id: str,
    outputs_dir: str,
) -> PipelinePaths:
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"pzz_compare_spatial_first_{task_external_id}_{ts}"

    pzz_codes_path = Path(os.getenv("PZZ_CODES_PATH", "data/PZZ-codes..csv"))

    reference_paths = ReferenceDataProvider().resolve_paths(
        pzz_zone_labels_override_path=pzz_zone_vri_labels_path,
        vri_classifier_override_path=vri_classifier_path,
    )

    return PipelinePaths(
        pzz_codes_path=pzz_codes_path,
        cadastral_geojson_path=Path(cadastral_geojson),
        pzz_zones_geojson_path=Path(pzz_zones_geojson),
        pzz_zone_vri_labels_path=reference_paths.pzz_zone_labels_path,
        vri_classifier_path=reference_paths.vri_classifier_path,
        include_pzz_check=include_pzz_check,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        output_geojson_path=out_dir / f"{basename}.geojson",
        unique_results_xlsx_path=out_dir / f"{basename}.xlsx",
        unique_results_json_path=out_dir / f"{basename}.json",
    )
