"""Celery worker startup: fail fast if a shared scratch directory isn't writable.

Mirrors the API's ``ensure_writable_dirs`` lifespan check (see
``test_settings.py``) for the worker process, which is a separate process with
its own volume mounts and writes into the same
uploads_dir/task_inputs_dir/outputs_dir as the API.
"""

from __future__ import annotations

import pytest
from celery.signals import worker_init

from service import tasks


def test_worker_init_calls_ensure_writable_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(tasks, "ensure_writable_dirs", calls.append)

    tasks._ensure_worker_storage_writable()

    assert calls == [tasks.settings]


def _unwritable(settings: object) -> None:
    raise RuntimeError("outputs_dir ('/no/such/mount') is not writable")


def test_worker_init_exits_when_storage_is_not_writable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "ensure_writable_dirs", _unwritable)

    with pytest.raises(SystemExit, match="not writable"):
        tasks._ensure_worker_storage_writable()


def test_failure_escapes_celery_signal_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery's Signal.send swallows any ``Exception`` from a receiver, so a
    plain RuntimeError here would let the worker start anyway."""
    monkeypatch.setattr(tasks, "ensure_writable_dirs", _unwritable)

    with pytest.raises(SystemExit, match="not writable"):
        worker_init.send(sender=None)
