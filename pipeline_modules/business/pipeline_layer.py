from __future__ import annotations

from .types import PipelineArtifacts, PipelinePaths, PipelineSettings
from .pipeline_impl import run_pipeline


def run_pipeline_with_typed_interfaces(paths: PipelinePaths, settings: PipelineSettings) -> PipelineArtifacts:
    """
    Adapter from service-level typed paths/settings to the low-level pipeline callable.
    """
    run_pipeline(
        pzz_codes_path=str(paths.pzz_codes_path),
        cadastral_geojson_path=str(paths.cadastral_geojson_path),
        pzz_zones_geojson_path=str(paths.pzz_zones_geojson_path),
        pzz_zone_vri_labels_path=str(paths.pzz_zone_vri_labels_path),
        vri_classifier_path=str(paths.vri_classifier_path),
        include_pzz_check=paths.include_pzz_check,
        cadastral_vri_col=paths.cadastral_vri_col,
        pzz_zone_code_col=paths.pzz_zone_code_col,
        pzz_zone_name_col=paths.pzz_zone_name_col,
        output_geojson_path=str(paths.output_geojson_path),
        unique_results_xlsx_path=str(paths.unique_results_xlsx_path),
        unique_results_json_path=str(paths.unique_results_json_path),
        base_url=settings.base_url,
        embed_model=settings.embed_model,
        generate_model=settings.generate_model,
        top_k=settings.top_k,
        batch_size=settings.batch_size,
    )

    return PipelineArtifacts(
        output_geojson_path=paths.output_geojson_path,
        unique_results_xlsx_path=paths.unique_results_xlsx_path,
        unique_results_json_path=paths.unique_results_json_path,
    )
