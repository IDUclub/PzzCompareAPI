from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineSettings:
    base_url: str
    embed_model: str
    generate_model: str
    top_k: int
    batch_size: int


@dataclass(frozen=True)
class PipelinePaths:
    pzz_codes_path: Path
    cadastral_geojson_path: Path
    pzz_zones_geojson_path: Path
    pzz_zone_vri_labels_path: Path
    vri_classifier_path: Path
    include_pzz_check: bool
    cadastral_vri_col: str
    pzz_zone_code_col: str
    pzz_zone_name_col: str
    output_geojson_path: Path
    unique_results_xlsx_path: Path
    unique_results_json_path: Path


@dataclass(frozen=True)
class PipelineArtifacts:
    output_geojson_path: Path
    unique_results_xlsx_path: Path
    unique_results_json_path: Path
