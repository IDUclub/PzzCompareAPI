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
    # Optional municipal boundary layer for the overlap checks; "" when absent.
    mo_boundaries_data_path: str = ""
    # True for the service-placement check (idempotency key prefixed "plc:"):
    # classify PZZ zones by whether a target VRI may be placed there, optionally
    # refined by a cadastre layer. Routes to PlacementPzzRunner.
    is_placement: bool = False
    # Target VRI code resolved from the user's service name (placement only).
    placement_vri_code: str = ""
    # The service name exactly as the user phrased it (placement only).
    placement_service_name: str = ""
    # NormGraph context of the placement (ПЗЗ zones + the service's norms), built
    # by the API; "" when absent (placement only).
    placement_context_path: str = ""
