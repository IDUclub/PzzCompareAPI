"""Tests for the low-memory GeoJSON reader used by upload validation/detection."""

import json
from pathlib import Path

import pytest

from service.infrastructure import geo_ingest
from service.infrastructure.geo_ingest import GeoIngestError, read_geojson_light


def _layer(count: int) -> dict:
    return {
        "type": "FeatureCollection",
        "name": "parcels",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"id": i, "area": i * 1.5e3, "vri": f"код {i}"},
                "geometry": {"type": "Point", "coordinates": [30.0 + i * 1e-4, 60.0]},
            }
            for i in range(count)
        ],
    }


def _write(tmp_path: Path, data: object, name: str = "layer.geojson") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def test_head_keeps_members_and_first_features(tmp_path: Path) -> None:
    layer = _layer(500)
    head = read_geojson_light(_write(tmp_path, layer), max_features=10)
    assert head["type"] == "FeatureCollection"
    assert head["crs"] == layer["crs"]
    assert head["features"] == layer["features"][:10]


def test_full_read_drops_geometry_after_limit(tmp_path: Path) -> None:
    layer = _layer(50)
    data = read_geojson_light(
        _write(tmp_path, layer), max_features=None, geometry_features=5
    )
    assert [f["properties"] for f in data["features"]] == [
        f["properties"] for f in layer["features"]
    ]
    assert data["features"][4]["geometry"] == layer["features"][4]["geometry"]
    assert all(f["geometry"] is None for f in data["features"][5:])


def test_values_split_by_chunk_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(geo_ingest, "_HEAD_CHUNK_CHARS", 7)
    layer = _layer(30)
    data = read_geojson_light(_write(tmp_path, layer), max_features=None)
    assert [f["properties"] for f in data["features"]] == [
        f["properties"] for f in layer["features"]
    ]


@pytest.mark.parametrize(
    "data", [{}, {"type": "FeatureCollection", "features": []}, [1, 2]]
)
def test_small_documents_round_trip(tmp_path: Path, data: object) -> None:
    assert read_geojson_light(_write(tmp_path, data)) == data


def test_bom_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bom.geojson"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(_layer(2)).encode("utf-8"))
    assert len(read_geojson_light(path)["features"]) == 2


@pytest.mark.parametrize(
    "content",
    [
        "{oops",
        json.dumps(_layer(3))[:-30],  # truncated inside the parsed head
        json.dumps(_layer(500))[:-5],  # truncated after the parsed head
        json.dumps(_layer(3)) + " {}",  # trailing data
    ],
    ids=["garbage", "truncated_head", "truncated_tail", "trailing_data"],
)
def test_malformed_json_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.geojson"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(GeoIngestError):
        read_geojson_light(path)


def test_binary_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "doc.geojson"
    path.write_bytes(b"PK\x03\x04\xff\xfe\x00binary")
    with pytest.raises(GeoIngestError):
        read_geojson_light(path)
