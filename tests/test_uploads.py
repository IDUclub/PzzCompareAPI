import json
import time

import pytest

from service.api.utils import durable_url
from service.application.use_cases.uploads import (
    UploadError,
    describe_upload,
    new_upload_dir,
    purge_expired_uploads,
    register_upload,
    resolve_upload,
)
from service.settings import Settings


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        uploads_dir=str(tmp_path / "uploads"),
        uploads_max_age_hours=overrides.pop("uploads_max_age_hours", 24),
        **overrides,
    )


def _store(settings, content=b"{}", owner="user-1", name="layer.geojson"):
    upload_id, directory = new_upload_dir(settings)
    payload = directory / name
    payload.write_bytes(content)
    record = register_upload(
        upload_id=upload_id,
        payload_path=payload,
        content_type="application/geo+json",
        owner_id=owner,
        settings=settings,
    )
    return record


def test_stored_upload_is_resolved_back(tmp_path):
    settings = _settings(tmp_path)
    record = _store(settings, content=b'{"type":"FeatureCollection"}')

    path = resolve_upload(record.upload_id, owner_id="user-1", settings=settings)

    assert path.read_bytes() == b'{"type":"FeatureCollection"}'
    assert record.size == len(b'{"type":"FeatureCollection"}')


def test_another_users_upload_is_refused(tmp_path):
    """An upload id travels through URLs, logs and chat — it is not a capability."""
    settings = _settings(tmp_path)
    record = _store(settings, owner="user-1")

    with pytest.raises(UploadError) as exc_info:
        resolve_upload(record.upload_id, owner_id="user-2", settings=settings)

    assert exc_info.value.status_code == 403


def test_anonymous_caller_may_use_an_owned_upload(tmp_path):
    """Submission endpoints accept anonymous calls; identity is checked only when given."""
    settings = _settings(tmp_path)
    record = _store(settings, owner="user-1")

    assert resolve_upload(record.upload_id, owner_id="", settings=settings).is_file()


def test_expired_upload_is_refused(tmp_path):
    settings = _settings(tmp_path, uploads_max_age_hours=0)
    record = _store(settings)
    time.sleep(0.01)

    with pytest.raises(UploadError) as exc_info:
        resolve_upload(record.upload_id, owner_id="user-1", settings=settings)

    assert exc_info.value.status_code == 410


def test_unknown_upload_id_is_not_found(tmp_path):
    settings = _settings(tmp_path)

    with pytest.raises(UploadError) as exc_info:
        resolve_upload("0" * 32, owner_id="user-1", settings=settings)

    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("bad_id", ["../etc", "a/b", "..", "sub/../../x"])
def test_upload_id_cannot_escape_the_uploads_directory(tmp_path, bad_id):
    settings = _settings(tmp_path)
    _store(settings)

    with pytest.raises(UploadError) as exc_info:
        resolve_upload(bad_id, owner_id="user-1", settings=settings)

    assert exc_info.value.status_code == 404


def test_missing_payload_is_reported_as_gone(tmp_path):
    settings = _settings(tmp_path)
    record = _store(settings)
    resolve_upload(record.upload_id, owner_id="user-1", settings=settings).unlink()

    with pytest.raises(UploadError) as exc_info:
        resolve_upload(record.upload_id, owner_id="user-1", settings=settings)

    assert exc_info.value.status_code == 410


def test_metadata_records_owner_and_expiry(tmp_path):
    settings = _settings(tmp_path)
    record = _store(settings, owner="user-7")

    meta_path = tmp_path / "uploads" / record.upload_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["owner_id"] == "user-7"
    assert meta["expires_at"] > meta["created_at"]
    assert meta["filename"] == "layer.geojson"


def test_description_carries_what_a_download_needs(tmp_path):
    settings = _settings(tmp_path)
    record = _store(settings, content=b"abc")

    described, payload = describe_upload(
        record.upload_id, owner_id="user-1", settings=settings
    )

    assert described.filename == "layer.geojson"
    assert described.content_type == "application/geo+json"
    assert payload.read_bytes() == b"abc"


def test_description_refuses_another_users_upload(tmp_path):
    settings = _settings(tmp_path)
    record = _store(settings, owner="user-1")

    with pytest.raises(UploadError) as exc_info:
        describe_upload(record.upload_id, owner_id="user-2", settings=settings)

    assert exc_info.value.status_code == 403


def test_durable_link_is_absolute_when_a_public_base_is_configured():
    """A link stored in chat history outlives the request that produced it."""
    assert (
        durable_url("/uploads/8f3c", "https://pzz.example.org")
        == "https://pzz.example.org/uploads/8f3c"
    )


def test_durable_link_falls_back_to_a_relative_path():
    assert durable_url("/uploads/8f3c", "") == "/uploads/8f3c"


def test_expired_uploads_are_purged_and_live_ones_kept(tmp_path):
    """The directory is a volume now — nothing else ever reclaims it."""
    settings = _settings(tmp_path)
    alive = _store(settings, name="alive.geojson")
    expired = _store(_settings(tmp_path, uploads_max_age_hours=0), name="old.geojson")

    removed = purge_expired_uploads(settings)

    assert removed == 1
    assert resolve_upload(
        alive.upload_id, owner_id="user-1", settings=settings
    ).is_file()
    assert not (tmp_path / "uploads" / expired.upload_id).exists()


def test_purge_keeps_a_directory_that_has_no_metadata_yet(tmp_path):
    """An upload still streaming to disk has no meta.json — it is not rubbish."""
    settings = _settings(tmp_path)
    _upload_id, directory = new_upload_dir(settings)
    (directory / "part.geojson").write_bytes(b"{}")

    assert purge_expired_uploads(settings) == 0
    assert directory.is_dir()


def test_purge_on_an_empty_root_is_not_an_error(tmp_path):
    assert purge_expired_uploads(_settings(tmp_path)) == 0


def test_corrupt_metadata_is_not_found(tmp_path):
    settings = _settings(tmp_path)
    record = _store(settings)
    (tmp_path / "uploads" / record.upload_id / "meta.json").write_text(
        "{", encoding="utf-8"
    )

    with pytest.raises(UploadError) as exc_info:
        resolve_upload(record.upload_id, owner_id="user-1", settings=settings)

    assert exc_info.value.status_code == 404
