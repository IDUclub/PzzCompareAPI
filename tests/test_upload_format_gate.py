"""Uploads are refused unless the service actually has a reader for the format.

Without this gate an unreadable file (the ПЗЗ regulations as .docx, a .pdf, an
archive) reached ``json.load`` and came back as "must contain valid JSON" — true
but useless, and it hid the real answer: the format is not supported at all.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from service.api.classifier import (
    _JSON_SLOT_EXTENSIONS,
    _ensure_supported_extension,
    _unsupported_geo_format_error,
)


def _upload(filename: str | None) -> UploadFile:
    return UploadFile(file=BytesIO(b"x"), filename=filename)


@pytest.mark.parametrize("filename", ["regs.docx", "regs.pdf", "regs.zip", "a.txt"])
def test_unreadable_formats_are_refused_with_415(filename: str) -> None:
    with pytest.raises(HTTPException) as exc:
        _ensure_supported_extension(
            _upload(filename), "vri_classifier_file", _JSON_SLOT_EXTENSIONS
        )
    assert exc.value.status_code == 415
    assert "классификатор ВРИ" in exc.value.detail
    assert filename.rsplit(".", 1)[-1] in exc.value.detail


def test_refusal_points_at_the_table_converter() -> None:
    with pytest.raises(HTTPException) as exc:
        _ensure_supported_extension(
            _upload("zones.csv"), "pzz_zone_vri_labels_file", _JSON_SLOT_EXTENSIONS
        )
    # A table is not wrong data, just wrong door — name the door.
    assert "/pzz/zone-descriptions/convert" in exc.value.detail


def test_json_passes() -> None:
    _ensure_supported_extension(
        _upload("labels.json"), "pzz_zone_vri_labels_file", _JSON_SLOT_EXTENSIONS
    )


@pytest.mark.parametrize("filename", [None, "", "payload"])
def test_missing_extension_is_tolerated(filename: str | None) -> None:
    # Callers have always been able to post bytes without a usable filename; the
    # content check downstream still applies to them.
    _ensure_supported_extension(
        _upload(filename), "vri_classifier_file", _JSON_SLOT_EXTENSIONS
    )


def test_geo_slot_refusal_uses_the_same_wording() -> None:
    exc = _unsupported_geo_format_error("cadastral_feature_collection_file", ".docx")
    assert exc.status_code == 415
    assert "слой земельных участков" in exc.detail
    assert "не поддерживается" in exc.detail
    assert ".geojson" in exc.detail
