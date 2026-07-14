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
    # True for urban_api-backed scenario tasks (idempotency key prefixed "sc:").
    # Routes to the deterministic, no-LLM classifier when enabled in settings.
    is_scenario: bool = False
    # True for the uploaded-building PZZ check (idempotency key prefixed "bld:"):
    # user-supplied buildings (physical_object_type_id / service_type_id + floors)
    # against user-supplied PZZ zones. Routes to UploadedBuildingPzzRunner.
    is_building_upload: bool = False
    # Building-layer column names (uploaded-building flow only).
    building_type_col: str = ""
    building_service_col: str = ""
    building_floors_col: str = ""
