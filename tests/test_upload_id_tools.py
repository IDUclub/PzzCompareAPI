"""The MCP layer must be able to submit a task without carrying the layer bytes."""

import asyncio

import pytest

from service.mcp_server.api_client import _geojson_files

FC = {"type": "FeatureCollection", "features": []}


def test_upload_id_slot_sends_no_body_part():
    parts = _geojson_files(
        ("cadastral_feature_collection_file", "cadastral.geojson", None, "8f3c"),
    )

    assert parts == {}


def test_inline_slot_still_sends_the_payload():
    parts = _geojson_files(
        ("cadastral_feature_collection_file", "cadastral.geojson", FC, None),
    )

    assert "cadastral_feature_collection_file" in parts
    filename, payload, content_type = parts["cadastral_feature_collection_file"]
    assert filename == "cadastral.geojson"
    assert b"FeatureCollection" in payload
    assert content_type == "application/geo+json"


def test_upload_id_wins_over_inline_data():
    """A caller sending both is migrating; the id is the newer contract."""
    parts = _geojson_files(
        ("cadastral_feature_collection_file", "cadastral.geojson", FC, "8f3c"),
    )

    assert parts == {}


def test_slots_are_independent():
    parts = _geojson_files(
        ("cadastral_feature_collection_file", "cadastral.geojson", FC, None),
        ("pzz_zones_feature_collection_file", "pzz_zones.geojson", None, "1a90"),
    )

    assert list(parts) == ["cadastral_feature_collection_file"]


class _Recorder:
    def __init__(self):
        self.calls = []

    async def post(self, url, files=None, data=None, headers=None):
        self.calls.append({"url": url, "files": files, "data": data})
        raise _Captured


class _Captured(Exception):
    pass


def _capture(coro):
    with pytest.raises(_Captured):
        asyncio.run(coro)


def test_pzz_check_forwards_both_upload_ids_as_form_fields():
    from service.mcp_server.api_client import ApiClient

    api = ApiClient.__new__(ApiClient)
    recorder = _Recorder()
    api._client = recorder

    _capture(
        api.submit_pzz_check(
            cadastral_upload_id="8f3c",
            pzz_zones_upload_id="1a90",
            cadastral_vri_col="Вид разреш",
            pzz_zone_code_col="Индекс_зоны",
            pzz_zone_name_col="Код_объекта",
        )
    )

    call = recorder.calls[0]
    assert call["url"] == "/tasks/pzz-check"
    assert call["files"] == {}
    assert call["data"]["cadastral_feature_collection_upload_id"] == "8f3c"
    assert call["data"]["pzz_zones_feature_collection_upload_id"] == "1a90"


def test_classify_only_forwards_the_upload_id():
    from service.mcp_server.api_client import ApiClient

    api = ApiClient.__new__(ApiClient)
    recorder = _Recorder()
    api._client = recorder

    _capture(
        api.submit_classify_only(
            cadastral_upload_id="8f3c", cadastral_vri_col="Вид разреш"
        )
    )

    call = recorder.calls[0]
    assert call["files"] == {}
    assert call["data"]["cadastral_feature_collection_upload_id"] == "8f3c"


def test_missing_layer_is_refused_before_the_request():
    from service.mcp_server.tools.tasks import _require_layer

    with pytest.raises(ValueError, match="cadastral"):
        _require_layer(None, None, "cadastral")


def _building_call(**kwargs):
    from service.mcp_server.api_client import ApiClient

    api = ApiClient.__new__(ApiClient)
    recorder = _Recorder()
    api._client = recorder
    _capture(
        api.submit_building_pzz_check(
            buildings_upload_id="8f3c", pzz_zones_upload_id="1a90", **kwargs
        )
    )
    return recorder.calls[0]


def test_building_check_sends_ids_and_no_body_parts():
    call = _building_call()

    assert call["url"] == "/tasks/building-pzz-check"
    assert call["files"] is None
    assert call["data"]["buildings_feature_collection_upload_id"] == "8f3c"
    assert call["data"]["pzz_zones_feature_collection_upload_id"] == "1a90"


def test_building_check_omits_untouched_optional_fields():
    """A blank descriptions slot must not look like an uploaded empty file."""
    data = _building_call()["data"]

    assert "pzz_descriptions_upload_id" not in data
    assert "confirmed_zone_map" not in data


def test_confirmed_zone_map_travels_as_json_with_readable_codes():
    data = _building_call(confirmed_zone_map={"СХ-3": "АГ-1"})["data"]

    assert data["confirmed_zone_map"] == '{"СХ-3": "АГ-1"}'


def test_building_check_refuses_a_missing_layer_id():
    from service.mcp_server.tools.tasks import submit_building_pzz_check_task

    fn = getattr(submit_building_pzz_check_task, "fn", submit_building_pzz_check_task)
    with pytest.raises(ValueError, match="upload_id"):
        asyncio.run(fn(buildings_upload_id="8f3c", pzz_zones_upload_id=""))
