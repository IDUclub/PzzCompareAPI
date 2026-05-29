"""Typed pipeline entrypoint for CLI / subprocess runs.

When the service uses SubprocessPipelineRunner, it spawns this module via
``python -m pipeline_modules.pipeline_v25`` and passes configuration as env vars.
InProcessPipelineRunner bypasses this file and calls pipeline_impl:run_pipeline directly.
"""

from __future__ import annotations

import os

from pipeline_modules.business import PipelineArtifacts, run_for_task
from pipeline_modules.business.types import PipelineSettings


def run_pipeline_job(
    *,
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
    if not cadastral_features_path:
        raise ValueError("cadastral_features_path must be provided")
    if include_pzz_check and not pzz_zones_features_path:
        raise ValueError("pzz_zones_features_path must be provided when include_pzz_check=True")

    return run_for_task(
        cadastral_features_path=cadastral_features_path,
        pzz_zones_features_path=pzz_zones_features_path,
        pzz_zone_vri_labels_path=pzz_zone_vri_labels_path,
        vri_classifier_path=vri_classifier_path,
        include_pzz_check=include_pzz_check,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        task_external_id=task_external_id,
        outputs_dir=outputs_dir,
        pipeline_settings=pipeline_settings,
    )


def _require_env(name: str) -> str:
    """Return env var or raise — used for inputs the API must always set."""
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(
            f"Environment variable {name!r} is required. "
            f"It is set by the API/worker from the task record; if you are "
            f"running the pipeline manually, export it explicitly."
        )
    return value


def run_pipeline() -> PipelineArtifacts:
    """Env-var-based entrypoint called when module is run as subprocess.

    Only ``CADASTRAL_VRI_COL`` is unconditionally required — it names the
    column that drives classification for every row. ``PZZ_ZONE_CODE_COL``
    and ``PZZ_ZONE_NAME_COL`` are only consumed when ``INCLUDE_PZZ_CHECK``
    is on (classifier-only runs ignore them), so they fall back to empty
    strings rather than failing loudly.
    """
    pipeline_settings = PipelineSettings(
        base_url=os.getenv("OLLAMA_BASE_URL", "").strip(),
        embed_model=os.getenv("EMBED_MODEL", "").strip(),
        generate_model=os.getenv("GENERATE_MODEL", "").strip(),
        top_k=int(os.getenv("TOP_K", "10")),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
    )
    include_pzz_check = os.getenv("INCLUDE_PZZ_CHECK", "1").strip().lower() in {"1", "true", "yes", "on"}
    # PZZ column names are only needed in pzz-check mode; require them only
    # then, so classifier-only runs can omit the env vars entirely.
    if include_pzz_check:
        pzz_zone_code_col = _require_env("PZZ_ZONE_CODE_COL")
        pzz_zone_name_col = _require_env("PZZ_ZONE_NAME_COL")
    else:
        pzz_zone_code_col = (os.getenv("PZZ_ZONE_CODE_COL") or "").strip()
        pzz_zone_name_col = (os.getenv("PZZ_ZONE_NAME_COL") or "").strip()
    return run_pipeline_job(
        cadastral_features_path=os.getenv("CADASTRAL_FEATURES_PATH", "").strip(),
        pzz_zones_features_path=os.getenv("PZZ_ZONES_FEATURES_PATH", "").strip(),
        pzz_zone_vri_labels_path=os.getenv("PZZ_ZONE_VRI_LABELS_PATH", "").strip(),
        vri_classifier_path=os.getenv("VRI_CLASSIFIER_PATH", "").strip(),
        include_pzz_check=include_pzz_check,
        cadastral_vri_col=_require_env("CADASTRAL_VRI_COL"),
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        task_external_id=os.getenv("TASK_EXTERNAL_ID", "manual").strip() or "manual",
        outputs_dir=os.getenv("OUTPUTS_DIR", "results").strip() or "results",
        pipeline_settings=pipeline_settings,
    )


if __name__ == "__main__":
    run_pipeline()
