"""Scratch-directory settings: absolute paths and the startup writability check.

Regression coverage for the prod incident where ``uploads_dir`` defaulted to a
relative path, resolved against whatever the process's CWD happened to be, and
a missing volume mount surfaced as an unhandled ``PermissionError`` on the
first request that touched it instead of failing at startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from service.settings import Settings, ensure_writable_dirs


def _settings(tmp_path: Path, **overrides: str) -> Settings:
    return Settings(
        uploads_dir=overrides.pop("uploads_dir", str(tmp_path / "uploads")),
        task_inputs_dir=overrides.pop("task_inputs_dir", str(tmp_path / "task_inputs")),
        outputs_dir=overrides.pop("outputs_dir", str(tmp_path / "outputs")),
        **overrides,
    )


def test_relative_scratch_dir_is_resolved_to_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, uploads_dir="uploads")

    assert Path(settings.uploads_dir).is_absolute()
    assert settings.uploads_dir == str((tmp_path / "uploads").resolve())


def test_ensure_writable_dirs_passes_for_writable_tmp_dirs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    ensure_writable_dirs(settings)  # must not raise

    assert Path(settings.uploads_dir).is_dir()
    assert Path(settings.task_inputs_dir).is_dir()
    assert Path(settings.outputs_dir).is_dir()


def test_ensure_writable_dirs_fails_fast_when_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    def _raise(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", _raise)

    with pytest.raises(RuntimeError, match="uploads_dir"):
        ensure_writable_dirs(settings)
