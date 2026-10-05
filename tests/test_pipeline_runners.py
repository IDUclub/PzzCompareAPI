from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

from service.infrastructure.runners import pipeline_runner as runner_mod
from service.infrastructure.runners.pipeline_runner import (
    InProcessPipelineRunner,
    PipelineRunnerFactory,
    StorageAwarePipelineRunner,
    SubprocessPipelineRunner,
    _build_output_glob,
)
from service.domain import PipelineRequest


class _FakeStorage:
    def is_remote(self) -> bool:
        return False


def _patch_local_storage(monkeypatch) -> None:
    monkeypatch.setattr(runner_mod, "get_object_storage", lambda: _FakeStorage())


def _request(tmp_path) -> PipelineRequest:
    return PipelineRequest(
        task_external_id="task-123",
        cadastral_data_path="/tmp/cadastral.geojson",
        pzz_zones_data_path="/tmp/pzz.geojson",
        pzz_zone_vri_labels_path="/tmp/labels.json",
        vri_classifier_path="/tmp/classifier.json",
        include_pzz_check=True,
        cadastral_vri_col="vri",
        pzz_zone_code_col="code",
        pzz_zone_name_col="name",
        outputs_dir=str(tmp_path),
    )


def _settings(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_runner_mode=mode,
        pipeline_runner_fallback_enabled=True,
        pipeline_runner_fallback_mode="subprocess",
        pipeline_callable="fake_module:fake_callable",
        pipeline_module="fake.pipeline.module",
        ollama_base_url="http://ollama",
        llm_backend="vllm",
        vllm_base_url="http://vllm",
        vllm_api_key="key",
        embed_model="embed",
        generate_model="generate",
        top_k=10,
        embed_batch_size=4,
    )


def test_factory_selects_in_process_mode(monkeypatch) -> None:
    _patch_local_storage(monkeypatch)
    runner = PipelineRunnerFactory.create(_settings("in_process"))
    assert isinstance(runner, StorageAwarePipelineRunner)
    assert isinstance(runner._inner, InProcessPipelineRunner)


def test_factory_selects_subprocess_mode(monkeypatch) -> None:
    _patch_local_storage(monkeypatch)
    runner = PipelineRunnerFactory.create(_settings("subprocess"))
    assert isinstance(runner, StorageAwarePipelineRunner)
    assert isinstance(runner._inner, SubprocessPipelineRunner)


def test_runners_return_same_result_contract(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path)
    settings = _settings("in_process")

    class FakeModule:
        @staticmethod
        def fake_callable(**kwargs):
            assert kwargs["task_external_id"] == request.task_external_id
            (tmp_path / "pzz_compare_spatial_first_task-123_result.geojson").write_text(
                "{}"
            )

    monkeypatch.setattr(
        runner_mod, "importlib", SimpleNamespace(import_module=lambda _: FakeModule)
    )

    def fake_subprocess_run(*args, **kwargs):
        (tmp_path / "pzz_compare_spatial_first_task-123_result.geojson").write_text(
            "{}"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_subprocess_run)

    in_process_result = InProcessPipelineRunner(settings).run(request)
    subprocess_result = SubprocessPipelineRunner(settings).run(request)

    assert isinstance(in_process_result, str)
    assert isinstance(subprocess_result, str)
    assert in_process_result == subprocess_result
    assert "task-123" in in_process_result


def test_build_output_glob_selects_primary_result_by_prefix_and_mtime(tmp_path) -> None:
    older_preferred = tmp_path / "pzz_compare_spatial_first_task-123_old.geojson"
    newer_nonpreferred = tmp_path / "secondary_task-123.geojson"
    newer_preferred = tmp_path / "pzz_compare_spatial_first_task-123_new.geojson"
    older_preferred.write_text("{}")
    time.sleep(0.01)
    newer_nonpreferred.write_text("{}")
    time.sleep(0.01)
    newer_preferred.write_text("{}")

    selected = _build_output_glob(tmp_path, "task-123")
    assert selected == str(newer_preferred)


def test_build_output_glob_raises_if_no_geojson(tmp_path) -> None:
    try:
        _build_output_glob(tmp_path, "task-123")
    except FileNotFoundError as exc:
        assert "task_external_id=task-123" in str(exc)
    else:
        raise AssertionError(
            "FileNotFoundError is expected when no geojson artifacts are produced"
        )


def _in_process_kwargs(tmp_path, monkeypatch, request) -> dict:
    seen: dict = {}

    class FakeModule:
        @staticmethod
        def fake_callable(**kwargs):
            seen.update(kwargs)
            (tmp_path / "pzz_compare_spatial_first_task-123_result.geojson").write_text(
                "{}"
            )

    monkeypatch.setattr(
        runner_mod, "importlib", SimpleNamespace(import_module=lambda _: FakeModule)
    )
    InProcessPipelineRunner(_settings("in_process")).run(request)
    return seen


def test_in_process_passes_mo_layer_only_when_present(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path)
    assert "mo_boundaries_features_path" not in _in_process_kwargs(
        tmp_path, monkeypatch, request
    )

    with_mo = replace(request, mo_boundaries_data_path="/tmp/mo.geojson")
    kwargs = _in_process_kwargs(tmp_path, monkeypatch, with_mo)
    assert kwargs["mo_boundaries_features_path"] == "/tmp/mo.geojson"


def test_subprocess_passes_mo_layer_via_env(tmp_path, monkeypatch) -> None:
    envs = []

    def fake_subprocess_run(*args, **kwargs):
        envs.append(kwargs["env"])
        (tmp_path / "pzz_compare_spatial_first_task-123_result.geojson").write_text(
            "{}"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_subprocess_run)
    request = replace(_request(tmp_path), mo_boundaries_data_path="/tmp/mo.geojson")
    SubprocessPipelineRunner(_settings("subprocess")).run(request)

    assert envs[0]["MO_BOUNDARIES_FEATURES_PATH"] == "/tmp/mo.geojson"


def test_storage_runner_downloads_mo_layer(tmp_path) -> None:
    downloaded = []

    class RemoteStorage:
        def is_remote(self) -> bool:
            return True

        def download_file(self, stored_path, local_path):
            downloaded.append(stored_path)

    runner = StorageAwarePipelineRunner(inner=None, storage=RemoteStorage())
    request = replace(
        _request(tmp_path), mo_boundaries_data_path="minio://inputs/t/mo.geojson"
    )

    local = runner._materialise_inputs(request, tmp_path)

    assert downloaded == ["minio://inputs/t/mo.geojson"]
    assert local.mo_boundaries_data_path == str(
        tmp_path / "mo_boundaries_feature_collection.geojson"
    )
    # An absent МО layer stays absent.
    without_mo = runner._materialise_inputs(_request(tmp_path), tmp_path)
    assert without_mo.mo_boundaries_data_path == ""
