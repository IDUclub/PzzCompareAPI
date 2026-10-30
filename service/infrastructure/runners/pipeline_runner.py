from __future__ import annotations

import importlib
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import replace
from pathlib import Path

from loguru import logger as _loguru

from service.domain import PipelineRequest
from service.infrastructure.storage import (
    ObjectStorage,
    get_object_storage,
    is_remote_path,
)
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

        module_name, callable_name = self._settings.pipeline_callable.split(
            ":", maxsplit=1
        )
        module = importlib.import_module(module_name)
        pipeline_callable = getattr(module, callable_name)

        # Passed only when present, so callables that predate the optional
        # municipal boundary layer keep working.
        extra_kwargs = {}
        if request.mo_boundaries_data_path:
            extra_kwargs["mo_boundaries_features_path"] = (
                request.mo_boundaries_data_path
            )
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
            **extra_kwargs,
        )

        return _build_output_glob(output_dir, request.task_external_id)


class SubprocessPipelineRunner(PipelineRunner):
    """Run pipeline module in a separate Python process."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def run(self, request: PipelineRequest) -> str:
        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pull the latest runtime config overrides into this worker's env, then
        # read settings *after* so the explicit assignments below reflect live
        # config (otherwise a stale self._settings would clobber the overrides
        # the subprocess inherits via os.environ).
        settings = self._settings
        try:
            from service.infrastructure.config_runtime import apply_overrides
            from service.settings import get_settings

            apply_overrides()
            settings = get_settings()
        except Exception:  # noqa: BLE001 — overrides must never block a pipeline run
            pass

        env = os.environ.copy()
        env["CADASTRAL_FEATURES_PATH"] = request.cadastral_data_path
        env["PZZ_ZONES_FEATURES_PATH"] = request.pzz_zones_data_path
        env["MO_BOUNDARIES_FEATURES_PATH"] = request.mo_boundaries_data_path
        env["PZZ_ZONE_VRI_LABELS_PATH"] = request.pzz_zone_vri_labels_path
        env["VRI_CLASSIFIER_PATH"] = request.vri_classifier_path
        env["INCLUDE_PZZ_CHECK"] = "1" if request.include_pzz_check else "0"
        env["CADASTRAL_VRI_COL"] = request.cadastral_vri_col
        env["PZZ_ZONE_CODE_COL"] = request.pzz_zone_code_col
        env["PZZ_ZONE_NAME_COL"] = request.pzz_zone_name_col
        env["TASK_EXTERNAL_ID"] = request.task_external_id
        env["OUTPUTS_DIR"] = str(output_dir)

        env["OLLAMA_BASE_URL"] = settings.ollama_base_url
        env["LLM_BACKEND"] = settings.llm_backend
        env["VLLM_BASE_URL"] = settings.vllm_base_url
        env["VLLM_API_KEY"] = settings.vllm_api_key
        env["VLLM_REQUEST_PRIORITY"] = str(settings.vllm_request_priority)
        env["EMBED_MODEL"] = settings.embed_model
        env["GENERATE_MODEL"] = settings.generate_model
        env["TOP_K"] = str(settings.top_k)
        env["EMBED_BATCH_SIZE"] = str(settings.embed_batch_size)
        env["EMBED_CACHE_DIR"] = settings.embed_cache_dir
        env["LLM_CACHE_DIR"] = settings.llm_cache_dir
        env["LLM_CACHE_TTL_DAYS"] = settings.llm_cache_ttl_days

        args = [sys.executable, "-m", settings.pipeline_module]
        returncode, stderr_tail = _run_pipeline_process(args, env)
        if returncode != 0:
            # Surfaced into the task's error_text / failed event, so the task
            # shows the real cause, not just "exit status 1".
            raise subprocess.CalledProcessError(returncode, args, stderr=stderr_tail)

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

    def _materialise_inputs(
        self, request: PipelineRequest, scratch: Path
    ) -> PipelineRequest:
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
            cadastral_data_path=materialise(
                request.cadastral_data_path, "cadastral_feature_collection.geojson"
            ),
            pzz_zones_data_path=materialise(
                request.pzz_zones_data_path, "pzz_zones_feature_collection.geojson"
            ),
            mo_boundaries_data_path=materialise(
                request.mo_boundaries_data_path,
                "mo_boundaries_feature_collection.geojson",
            ),
            pzz_zone_vri_labels_path=materialise(
                request.pzz_zone_vri_labels_path, "pzz_zone_vri_labels.json"
            ),
            vri_classifier_path=materialise(
                request.vri_classifier_path, "vri_classifier.json"
            ),
            placement_context_path=materialise(
                request.placement_context_path, "placement_context.json"
            ),
            existing_objects_data_path=materialise(
                request.existing_objects_data_path, "existing_objects.geojson"
            ),
            outputs_dir=str(local_outputs_dir),
        )

    def _upload_output(self, local_output_path: str, task_external_id: str) -> str:
        filename = Path(local_output_path).name
        object_key = f"outputs/{task_external_id}/{filename}"
        return self._storage.upload_file(local_output_path, object_key)


class PipelineRunnerFactory:
    """Create pipeline runner by mode with optional fallback."""

    @staticmethod
    def create(
        settings: "Settings", request: "PipelineRequest | None" = None
    ) -> PipelineRunner:
        if request is not None and getattr(request, "is_placement", False):
            from service.infrastructure.runners.placement_pzz_runner import (
                PlacementPzzRunner,
            )

            inner: PipelineRunner = PlacementPzzRunner(settings)
        elif request is not None and getattr(request, "is_building_upload", False):
            from service.infrastructure.runners.uploaded_building_pzz_runner import (
                UploadedBuildingPzzRunner,
            )

            inner = UploadedBuildingPzzRunner(settings)
        elif (
            request is not None
            and getattr(request, "is_scenario", False)
            and (
                getattr(settings, "scenario_deterministic", False)
                or request.existing_objects_data_path
            )
        ):
            from service.infrastructure.runners.deterministic_scenario_runner import (
                DeterministicScenarioRunner,
            )

            inner = DeterministicScenarioRunner(settings)
        else:
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


# Line format of the pipeline's logging.basicConfig (pipeline_modules/*:__main__):
# "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_PIPELINE_LOG_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| (?P<level>[A-Z]+) \| "
    r"(?P<name>[^|]+?) \| (?P<message>.*)$"
)
_TRACEBACK_START = "Traceback (most recent call last):"
# tqdm frames ("45%|████▌     | 9/20 [00:03<00:04, 2.6it/s]") — noise in logs.
_PROGRESS_BAR_LINE = re.compile(r"\d+%\|.*\||it/s\]|s/it\]")
_STDERR_TAIL_CHARS = 8000
_STDERR_TAIL_LINES = 500


# A record is held back until the next line shows whether a traceback follows;
# if the subprocess goes quiet (a long step) it is flushed after this delay.
_PENDING_FLUSH_SECONDS = 0.5


class _PipelineRecord:
    """One log record of the pipeline subprocess, possibly spanning lines."""

    def __init__(self, level: str, name: str, message: str) -> None:
        self.level = level
        self.name = name
        self.message_lines = [message]
        self.traceback_lines: list[str] = []

    def add_line(self, line: str) -> bool:
        """Attach a continuation line; False if *line* is not part of it."""
        if self.traceback_lines or line.startswith(_TRACEBACK_START):
            self.traceback_lines.append(line)
        elif line[:1].isspace():
            self.message_lines.append(line)
        else:
            return False
        return True

    def emit(self) -> None:
        try:
            level = _loguru.level(self.level).name
        except ValueError:
            level = "INFO"
        bound = _loguru.bind(logger_name=self.name)
        if self.traceback_lines:
            bound = bound.bind(exception_text="\n".join(self.traceback_lines))
        bound.log(level, "\n".join(self.message_lines))


def _run_pipeline_process(args: list[str], env: dict[str, str]) -> tuple[int, str]:
    """Run the pipeline subprocess, streaming its logs into ours.

    The subprocess logs to stdout in its own format; each record is re-emitted
    through loguru (stdout + Redis /logs sinks) under its original logger name,
    tagged with the current task context, and with any traceback attached as
    the entry's exception. Non-log stdout (print) goes to "pipeline.stdout".

    stderr is kept as a bounded tail (progress bars dropped). On failure it is
    logged as an error and returned for the caller's CalledProcessError; on
    success any remaining stderr (warnings) is logged once as a warning.

    Returns (returncode, stderr_tail).
    """
    env = {**env, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stderr_lines: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
    stdout_lines: queue.Queue[str | None] = queue.Queue()

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.rstrip("\n")
            if line.strip() and not _PROGRESS_BAR_LINE.search(line):
                stderr_lines.append(line)

    def _drain_stdout() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            stdout_lines.put(raw.rstrip("\n"))
        stdout_lines.put(None)

    readers = [
        threading.Thread(target=_drain_stderr, name="pipeline-stderr", daemon=True),
        threading.Thread(target=_drain_stdout, name="pipeline-stdout", daemon=True),
    ]
    for reader in readers:
        reader.start()

    pending: _PipelineRecord | None = None
    try:
        while True:
            try:
                line = stdout_lines.get(timeout=_PENDING_FLUSH_SECONDS)
            except queue.Empty:
                if pending is not None:
                    pending.emit()
                    pending = None
                continue
            if line is None:
                break
            match = _PIPELINE_LOG_LINE.match(line)
            if match:
                if pending is not None:
                    pending.emit()
                pending = _PipelineRecord(
                    match["level"], match["name"].strip(), match["message"]
                )
            elif pending is None or not pending.add_line(line):
                if pending is not None:
                    pending.emit()
                    pending = None
                if line.strip():
                    _loguru.bind(logger_name="pipeline.stdout").info(line)
        if pending is not None:
            pending.emit()
        returncode = proc.wait()
    except BaseException:
        # Interrupted (time limit, worker shutdown): never leave the child
        # running on its own.
        proc.kill()
        proc.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)

    stderr_text = "\n".join(stderr_lines)
    # A Python traceback ends with the exception type + message; start the tail
    # at the last traceback header so it isn't diluted by earlier warnings.
    traceback_at = stderr_text.rfind(_TRACEBACK_START)
    if traceback_at != -1:
        stderr_text = stderr_text[traceback_at:]
    stderr_tail = stderr_text.strip()[-_STDERR_TAIL_CHARS:]

    log = _loguru.bind(logger_name="pipeline.stderr")
    if returncode != 0:
        log.bind(exception_text=stderr_tail or None).error(
            "pipeline subprocess exited with code {}", returncode
        )
    elif stderr_tail:
        log.warning("pipeline subprocess stderr:\n{}", stderr_tail)
    return returncode, stderr_tail


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

    preferred_candidates = [
        path
        for path in geojson_candidates
        if path.stem.startswith("pzz_compare_spatial_first_")
    ]
    candidates = preferred_candidates or geojson_candidates
    primary_result_path = max(
        candidates, key=lambda path: (path.stat().st_mtime, str(path))
    )
    return str(primary_result_path)
