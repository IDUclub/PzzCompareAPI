from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PipelineRequest:
    task_external_id: str
    cadastral_data_path: str
    pzz_zones_data_path: str
    pzz_zone_vri_labels_path: str
    vri_classifier_path: str
    include_pzz_check: bool
    cadastral_vri_col: str
    pzz_zone_code_col: str
    pzz_zone_name_col: str
    outputs_dir: str
