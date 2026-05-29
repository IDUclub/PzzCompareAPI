from .io_layer import build_paths
from .pipeline_layer import run_pipeline_with_typed_interfaces
from .types import PipelineArtifacts, PipelineSettings


def run_for_task(
    cadastral_features_path: str,
    pzz_zones_features_path: str,
    pzz_zone_vri_labels_path: str,
    vri_classifier_path: str,
    include_pzz_check: bool,
    cadastral_vri_col: str,
    pzz_zone_code_col: str,
    pzz_zone_name_col: str,
    task_external_id: str,
    outputs_dir: str,
    pipeline_settings: PipelineSettings,
) -> PipelineArtifacts:
    """Run pipeline for a prepared task payload.

    Accepts ``pipeline_settings`` explicitly — no implicit import of service settings.
    """
    paths = build_paths(
        cadastral_geojson=cadastral_features_path,
        pzz_zones_geojson=pzz_zones_features_path,
        pzz_zone_vri_labels_path=pzz_zone_vri_labels_path,
        vri_classifier_path=vri_classifier_path,
        include_pzz_check=include_pzz_check,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        task_external_id=task_external_id,
        outputs_dir=outputs_dir,
    )

    return run_pipeline_with_typed_interfaces(paths=paths, settings=pipeline_settings)
