"""Uploads are refused unless the service actually has a reader for the format.

Without this gate an unreadable file (the ПЗЗ regulations as .docx, a .pdf, an
archive) reached ``json.load`` and came back as "must contain valid JSON" — true
but useless, and it hid the real answer: the format is not supported at all.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from service.api.classifier import (
    _JSON_SLOT_EXTENSIONS,
    _ensure_supported_extension,
    _geo_unreadable_error,
    _resolve_file_slot,
    _unsupported_geo_format_error,
    _upload_to_feature_collection,
    _validate_json_file,
)
from service.api.utils import stream_upload_to_file, upload_too_large_error
from service.infrastructure.geo_ingest import GeoIngestError


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


@pytest.mark.parametrize("suffix", [".shp", ".dbf", ".TAB", ".mif"])
def test_lone_multifile_part_is_told_to_zip_the_layer(suffix: str) -> None:
    # A bare .shp or .tab is unreadable without its sibling files; the refusal
    # must say how to send it, not only that it was refused.
    exc = _unsupported_geo_format_error("pzz_zones_feature_collection_file", suffix)
    assert exc.status_code == 415
    assert ".zip" in exc.detail
    assert "заархивируйте" in exc.detail


def test_unrelated_format_gets_no_zip_hint() -> None:
    exc = _unsupported_geo_format_error("cadastral_feature_collection_file", ".docx")
    assert "заархивируйте" not in exc.detail


def test_unreadable_layer_names_the_slot_and_the_file() -> None:
    # Worded like the 422 CRS refusal: the user sees the layer as the UI names
    # it, not the multipart key.
    exc = _geo_unreadable_error(
        "pzz_zones_feature_collection_file",
        GeoIngestError("в архиве несколько слоёв"),
        "zones.zip",
    )
    assert exc.status_code == 400
    assert exc.detail == "слой зон ПЗЗ («zones.zip»): в архиве несколько слоёв"


@pytest.mark.parametrize("payload", [b"{not json", b"PK\x03\x04\xff\xfe binary"])
def test_broken_geojson_is_a_400_in_russian(tmp_path: Path, payload: bytes) -> None:
    # A binary used to escape as UnicodeDecodeError, i.e. a 500.
    upload = UploadFile(file=BytesIO(payload), filename="parcels.geojson")
    with pytest.raises(HTTPException) as exc:
        _upload_to_feature_collection(
            upload, tmp_path, "cadastral_feature_collection_file", 1024
        )
    assert exc.value.status_code == 400
    assert exc.value.detail == (
        "слой земельных участков («parcels.geojson»): "
        "файл не является корректным JSON/GeoJSON."
    )


def test_geojson_that_is_not_an_object_is_a_400_in_russian(tmp_path: Path) -> None:
    upload = UploadFile(file=BytesIO(b"[1, 2]"), filename="zones.geojson")
    with pytest.raises(HTTPException) as exc:
        _upload_to_feature_collection(
            upload, tmp_path, "pzz_zones_feature_collection_file", 1024
        )
    assert exc.value.status_code == 400
    assert exc.value.detail == (
        "слой зон ПЗЗ («zones.geojson»): содержимое файла должно быть JSON-объектом."
    )


@pytest.mark.parametrize(
    ("payload", "expected", "message"),
    [
        (b"{oops", (dict, list), "файл не является корректным JSON/GeoJSON."),
        (b"\xff\xfe", (dict, list), "файл не является корректным JSON/GeoJSON."),
        (b"{}", list, "содержимое файла должно быть JSON-массивом."),
        (
            b"42",
            (dict, list),
            "содержимое файла должно быть JSON-объектом или массивом.",
        ),
    ],
)
def test_json_slot_refusals_are_in_russian(
    tmp_path: Path, payload: bytes, expected, message: str
) -> None:
    path = tmp_path / "classifier.json"
    path.write_bytes(payload)
    with pytest.raises(HTTPException) as exc:
        _validate_json_file(path, expected, "vri_classifier_file")
    assert exc.value.status_code == 400
    assert exc.value.detail == f"классификатор ВРИ: {message}"


def test_too_large_upload_is_a_413_in_russian(tmp_path: Path) -> None:
    upload = UploadFile(file=BytesIO(b"x" * 2048), filename="parcels.geojson")
    with pytest.raises(HTTPException) as exc:
        stream_upload_to_file(
            upload, tmp_path / "out", 1024, "cadastral_feature_collection_file"
        )
    assert exc.value.status_code == 413
    assert exc.value.detail.startswith(
        "слой земельных участков («parcels.geojson»): файл больше допустимого размера"
    )


@pytest.mark.parametrize(
    ("max_bytes", "limit"),
    [(200 * 1024 * 1024, "200 МБ"), (1024 * 1024 // 2, "0.5 МБ")],
)
def test_size_limit_is_shown_in_megabytes(max_bytes: int, limit: str) -> None:
    exc = upload_too_large_error("file", max_bytes, "zones.xlsx")
    assert (
        exc.detail == f"файл («zones.xlsx»): файл больше допустимого размера {limit}."
    )


def test_missing_required_file_is_a_422_in_russian(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        _resolve_file_slot(
            None,
            None,
            "pzz_zones_feature_collection_file",
            required=True,
            owner_id="u1",
            app_settings=None,  # not consulted when neither form is given
            scratch=tmp_path,
        )
    assert exc.value.status_code == 422
    assert exc.value.detail == (
        "слой зон ПЗЗ: файл не приложен: передайте его в поле "
        "«pzz_zones_feature_collection_file» или укажите "
        "«pzz_zones_feature_collection_file_upload_id»."
    )


def test_upload_id_refusal_names_the_slot_in_russian(tmp_path: Path) -> None:
    from service.settings import Settings

    settings = Settings(uploads_dir=str(tmp_path / "uploads"))
    with pytest.raises(HTTPException) as exc:
        _resolve_file_slot(
            None,
            "0" * 32,
            "pzz_zones_feature_collection_file",
            required=True,
            owner_id="u1",
            app_settings=settings,
            scratch=tmp_path / "scratch",
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == (
        f"слой зон ПЗЗ: файл с upload_id «{'0' * 32}» не найден: загрузите его заново."
    )
