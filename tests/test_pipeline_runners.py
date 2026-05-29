from __future__ import annotations

import time
from types import SimpleNamespace

from service.infrastructure.runners.pipeline_runner import (
    InProcessPipelineRunner,
    PipelineRunnerFactory,
    SubprocessPipelineRunner,
    _build_output_glob,
)
from service.domain import PipelineRequest


def _request(tmp_path) -> PipelineRequest:
    return PipelineRequest(
        task_external_id="task-123",
        cadastral_data_path="/tmp/cadastral.geojson",
        pzz_zones_data_path="/tmp/pzz.geojson",
        pzz_zone_vri_labels_path="/tmp/labels.json",
        vri_classifier_path="/tmp/classifier.json",
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


def test_factory_selects_in_process_mode() -> None:
    runner = PipelineRunnerFactory.create(_settings("in_process"))
    assert isinstance(runner, InProcessPipelineRunner)


def test_factory_selects_subprocess_mode() -> None:
    runner = PipelineRunnerFactory.create(_settings("subprocess"))
    assert isinstance(runner, SubprocessPipelineRunner)


def test_runners_return_same_result_contract(tmp_path, monkeypatch) -> None:
    request = _request(tmp_path)
    settings = _settings("in_process")

    class FakeModule:
        @staticmethod
        def fake_callable(**kwargs):
            assert kwargs["task_external_id"] == request.task_external_id
            (tmp_path / "pzz_compare_spatial_first_task-123_result.geojson").write_text("{}")

    monkeypatch.setattr(
        "service.application.runners.pipeline_runner.importlib.import_module",
        lambda _: FakeModule,
    )
    monkeypatch.setattr(
        "service.application.runners.pipeline_runner.subprocess.run",
        lambda *args, **kwargs: None,
    )

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
        raise AssertionError("FileNotFoundError is expected when no geojson artifacts are produced")
