from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from service.domain import PipelineRequest
from service.infrastructure.storage import ObjectStorage, get_object_storage, is_remote_path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.settings import Settings


class PipelineRunner(ABC):
    """Common interface for pipeline execution backends."""

    @abstractmethod
    def run(self, request: PipelineRequest) -> str:
        """Execute pipeline and return primary result file path."""


class InProcessPipelineRunner(PipelineRunner):
    """Import and execute configured pipeline callable in current process."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def run(self, request: PipelineRequest) -> str:
        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        module_name, callable_name = self._settings.pipeline_callable.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        pipeline_callable = getattr(module, callable_name)

        pipeline_callable(
            cadastral_features_path=request.cadastral_data_path,
            pzz_zones_features_path=request.pzz_zones_data_path,
            pzz_zone_vri_labels_path=request.pzz_zone_vri_labels_path,
            vri_classifier_path=request.vri_classifier_path,
            include_pzz_check=request.include_pzz_check,
            cadastral_vri_col=request.cadastral_vri_col,
            pzz_zone_code_col=request.pzz_zone_code_col,
            pzz_zone_name_col=request.pzz_zone_name_col,
            task_external_id=request.task_external_id,
            outputs_dir=str(output_dir),
        )

        return _build_output_glob(output_dir, request.task_external_id)


class SubprocessPipelineRunner(PipelineRunner):
    """Run pipeline module in a separate Python process."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def run(self, request: PipelineRequest) -> str:
        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CADASTRAL_FEATURES_PATH"] = request.cadastral_data_path
        env["PZZ_ZONES_FEATURES_PATH"] = request.pzz_zones_data_path
        env["PZZ_ZONE_VRI_LABELS_PATH"] = request.pzz_zone_vri_labels_path
        env["VRI_CLASSIFIER_PATH"] = request.vri_classifier_path
        env["INCLUDE_PZZ_CHECK"] = "1" if request.include_pzz_check else "0"
        env["CADASTRAL_VRI_COL"] = request.cadastral_vri_col
        env["PZZ_ZONE_CODE_COL"] = request.pzz_zone_code_col
        env["PZZ_ZONE_NAME_COL"] = request.pzz_zone_name_col
        env["TASK_EXTERNAL_ID"] = request.task_external_id
        env["OUTPUTS_DIR"] = str(output_dir)

        env["OLLAMA_BASE_URL"] = self._settings.ollama_base_url
        env["LLM_BACKEND"] = self._settings.llm_backend
        env["VLLM_BASE_URL"] = self._settings.vllm_base_url
        env["VLLM_API_KEY"] = self._settings.vllm_api_key
        env["EMBED_MODEL"] = self._settings.embed_model
        env["GENERATE_MODEL"] = self._settings.generate_model
        env["TOP_K"] = str(self._settings.top_k)
        env["EMBED_BATCH_SIZE"] = str(self._settings.embed_batch_size)
        env["PIPELINE_CALLABLE"] = self._settings.pipeline_callable

        result = subprocess.run(
            [sys.executable, "-m", self._settings.pipeline_module],
            env=env,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-3000:]
            raise subprocess.CalledProcessError(result.returncode, result.args, stderr=stderr_tail)

        return _build_output_glob(output_dir, request.task_external_id)


class StorageAwarePipelineRunner(PipelineRunner):
    """Wrap an inner runner with MinIO download / upload around the pipeline.

    The pipeline modules expect local filesystem paths, so this decorator:
      1. Downloads any ``minio://`` inputs into a per-task scratch directory
      2. Rebuilds the ``PipelineRequest`` with local paths
      3. Delegates to the inner runner (in-process or subprocess)
      4. Uploads the result back to MinIO and returns the stored path
      5. Cleans the scratch directory afterwards

    When storage is local (no MinIO configured), the wrapper short-circuits
    and returns the inner result unchanged — zero overhead.
    """

    def __init__(self, inner: PipelineRunner, storage: ObjectStorage) -> None:
        self._inner = inner
        self._storage = storage

    def run(self, request: PipelineRequest) -> str:
        if not self._storage.is_remote():
            return self._inner.run(request)

        scratch = Path(tempfile.mkdtemp(prefix=f"task_{request.task_external_id}_"))
        try:
            local_request = self._materialise_inputs(request, scratch)
            local_output_path = self._inner.run(local_request)
            return self._upload_output(local_output_path, request.task_external_id)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _materialise_inputs(self, request: PipelineRequest, scratch: Path) -> PipelineRequest:
        """Download MinIO inputs to scratch; leave local paths untouched."""
        def materialise(stored_path: str, filename: str) -> str:
            if not stored_path:
                return stored_path
            if not is_remote_path(stored_path):
                return stored_path
            local_path = scratch / filename
            self._storage.download_file(stored_path, str(local_path))
            return str(local_path)

        local_outputs_dir = scratch / "outputs"
        local_outputs_dir.mkdir(parents=True, exist_ok=True)

        return replace(
            request,
            cadastral_data_path=materialise(request.cadastral_data_path, "cadastral_feature_collection.geojson"),
            pzz_zones_data_path=materialise(request.pzz_zones_data_path, "pzz_zones_feature_collection.geojson"),
            pzz_zone_vri_labels_path=materialise(request.pzz_zone_vri_labels_path, "pzz_zone_vri_labels.json"),
            vri_classifier_path=materialise(request.vri_classifier_path, "vri_classifier.json"),
            outputs_dir=str(local_outputs_dir),
        )

    def _upload_output(self, local_output_path: str, task_external_id: str) -> str:
        filename = Path(local_output_path).name
        object_key = f"outputs/{task_external_id}/{filename}"
        return self._storage.upload_file(local_output_path, object_key)


class PipelineRunnerFactory:
    """Create pipeline runner by mode with optional fallback."""

    @staticmethod
    def create(settings: "Settings") -> PipelineRunner:
        inner = PipelineRunnerFactory._create_inner(settings)
        return StorageAwarePipelineRunner(inner, get_object_storage())

    @staticmethod
    def _create_inner(settings: "Settings") -> PipelineRunner:
        mode = settings.pipeline_runner_mode
        if mode == "in_process":
            return InProcessPipelineRunner(settings)
        if mode == "subprocess":
            return SubprocessPipelineRunner(settings)

        if settings.pipeline_runner_fallback_enabled:
            fallback = settings.pipeline_runner_fallback_mode
            if fallback == "in_process":
                return InProcessPipelineRunner(settings)
            if fallback == "subprocess":
                return SubprocessPipelineRunner(settings)

        raise ValueError(f"Unsupported pipeline_runner_mode: {mode}")


def _build_output_glob(output_dir: Path, task_external_id: str) -> str:
    """
    Return the primary GeoJSON artifact produced by a pipeline run.

    Selection rule when multiple artifacts match:
    1) Prefer files whose stem starts with ``pzz_compare_spatial_first_``.
    2) Within preferred (or all) candidates choose the newest by mtime.
    3) If mtimes are equal choose lexicographically last path for stability.
    """

    geojson_candidates = sorted(output_dir.glob(f"*{task_external_id}*.geojson"))
    if not geojson_candidates:
        raise FileNotFoundError(
            f"No GeoJSON artifacts were produced for task_external_id={task_external_id} in {output_dir}"
        )

    preferred_candidates = [path for path in geojson_candidates if path.stem.startswith("pzz_compare_spatial_first_")]
    candidates = preferred_candidates or geojson_candidates
    primary_result_path = max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))
    return str(primary_result_path)
