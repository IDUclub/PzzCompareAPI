"""Tests for zone-description table ingest: reader, conversion, convert endpoint."""

import io
import json

from fastapi.testclient import TestClient
from openpyxl import Workbook

from service import app as app_module
from service.api import zone_descriptions as zd
from service.application.use_cases.convert_zone_table import (
    ConversionError,
    convert_zone_table,
)
from service.infrastructure.table_reader import TableReadError, read_table


def _xlsx_bytes(headers, rows, *, sheet_title=None) -> bytes:
    wb = Workbook()
    ws = wb.active
    if sheet_title:
        ws.title = sheet_title
    ws.append(headers)
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# --- table_reader -----------------------------------------------------------


def test_read_csv_cp1251_semicolon():
    text = "Zone;Code;VRI_Code;VRI\nЖ-1;Основной;2.1;Для ИЖС\n"
    rows, headers = read_table(text.encode("cp1251"), "z.csv")
    assert headers == ["Zone", "Code", "VRI_Code", "VRI"]
    assert rows[0]["VRI"] == "Для ИЖС"
    assert rows[0]["Zone"] == "Ж-1"


def test_read_csv_utf8_bom_comma():
    text = "Zone,Code,VRI_Code\nО-2,Вспомогательный,3.1.1\n"
    rows, headers = read_table(text.encode("utf-8-sig"), "z.csv")
    assert headers == ["Zone", "Code", "VRI_Code"]
    assert rows[0]["Zone"] == "О-2"
    assert rows[0]["Code"] == "Вспомогательный"


def test_read_xlsx_named_sheet():
    data = _xlsx_bytes(
        ["Zone", "Code", "VRI_Code", "VRI"],
        [["Ж-1", "Основной", "2.1", "Для ИЖС"]],
        sheet_title="pzz_Regl",
    )
    rows, headers = read_table(data, "z.xlsx", sheet="pzz_Regl")
    assert rows[0]["VRI_Code"] == "2.1"
    assert rows[0]["VRI"] == "Для ИЖС"


def test_read_xlsx_missing_sheet_raises():
    data = _xlsx_bytes(["Zone"], [["Ж-1"]])
    try:
        read_table(data, "z.xlsx", sheet="nope")
    except TableReadError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected TableReadError")


def test_read_unsupported_extension_raises():
    try:
        read_table(b"whatever", "z.docx")
    except TableReadError:
        pass
    else:
        raise AssertionError("expected TableReadError")


# --- convert_zone_table ------------------------------------------------------

_MAP = {
    "zone_code": "Zone",
    "permission": "Code",
    "vri_code": "VRI_Code",
    "vri_name": "VRI",
    "zone_name": None,
}


def test_convert_groups_and_buckets():
    rows = [
        {"Zone": "Ж-1", "Code": "Основной", "VRI_Code": "2.1", "VRI": "Для ИЖС"},
        {
            "Zone": "Ж-1",
            "Code": "Условно разрешенный",
            "VRI_Code": "4.7",
            "VRI": "Гостиницы",
        },
        {"Zone": "Ж-1", "Code": "Вспомогательный", "VRI_Code": "4.9", "VRI": "Гаражи"},
        {"Zone": "П-1", "Code": "Основной", "VRI_Code": "6.0", "VRI": "Производство"},
    ]
    zones, report = convert_zone_table(rows, _MAP)
    assert report.zones_count == 2
    assert report.vri_count == 4
    zh1 = next(z for z in zones if z["zone_code"] == "Ж-1")
    assert [v["vri_code"] for v in zh1["main"]] == ["2.1"]
    assert [v["vri_code"] for v in zh1["conditional"]] == ["4.7"]
    assert [v["vri_code"] for v in zh1["auxiliary"]] == ["4.9"]
    assert zh1["zone_name"] == "Ж-1"  # no zone_name column -> defaults to code


def test_convert_folds_zone_code_spelling():
    rows = [
        {"Zone": "Ж-1", "Code": "Основной", "VRI_Code": "2.1", "VRI": "a"},
        {"Zone": "ж 1", "Code": "Основной", "VRI_Code": "2.2", "VRI": "b"},
    ]
    zones, report = convert_zone_table(rows, _MAP)
    assert report.zones_count == 1
    assert {v["vri_code"] for v in zones[0]["main"]} == {"2.1", "2.2"}


def test_convert_no_permission_all_main_with_warning():
    rows = [
        {"Zone": "Ж-1", "VRI_Code": "2.1", "VRI": "a"},
        {"Zone": "Ж-1", "VRI_Code": "2.2", "VRI": "b"},
    ]
    column_map = {"zone_code": "Zone", "vri_code": "VRI_Code", "vri_name": "VRI"}
    zones, report = convert_zone_table(rows, column_map)
    assert {v["vri_code"] for v in zones[0]["main"]} == {"2.1", "2.2"}
    assert not zones[0]["conditional"]
    assert any("тип" in w.lower() for w in report.warnings)


def test_convert_unknown_permission_defaults_main_and_warns():
    rows = [{"Zone": "Ж-1", "Code": "Запрещённый", "VRI_Code": "2.1", "VRI": "a"}]
    zones, report = convert_zone_table(rows, _MAP)
    assert [v["vri_code"] for v in zones[0]["main"]] == ["2.1"]
    assert any("не распознан" in w.lower() for w in report.warnings)


def test_convert_dedupes_and_keeps_name():
    rows = [
        {"Zone": "Ж-1", "Code": "Основной", "VRI_Code": "2.1", "VRI": ""},
        {"Zone": "Ж-1", "Code": "Основной", "VRI_Code": "2.1", "VRI": "Для ИЖС"},
    ]
    zones, report = convert_zone_table(rows, _MAP)
    assert len(zones[0]["main"]) == 1
    assert zones[0]["main"][0]["vri_name"] == "Для ИЖС"


def test_convert_skips_empty_and_reports():
    rows = [
        {"Zone": "", "Code": "Основной", "VRI_Code": "2.1", "VRI": "a"},
        {"Zone": "Ж-1", "Code": "Основной", "VRI_Code": "", "VRI": "b"},
        {"Zone": "Ж-1", "Code": "Основной", "VRI_Code": "2.1", "VRI": "c"},
    ]
    zones, report = convert_zone_table(rows, _MAP)
    assert report.rows_used == 1
    assert report.zones_count == 1


def test_convert_missing_required_column_raises():
    try:
        convert_zone_table([{"Zone": "Ж-1"}], {"zone_code": "Zone", "vri_code": None})
    except ConversionError:
        pass
    else:
        raise AssertionError("expected ConversionError")


# --- convert endpoint --------------------------------------------------------


class _FakeOllama:
    """Async-context Ollama stub; returns no LLM column suggestions."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def complete_json(self, messages, schema=None, model=None):
        return {}


def test_convert_endpoint_xlsx(monkeypatch):
    monkeypatch.setattr(
        zd, "build_chat_llm_client", lambda settings=None: _FakeOllama()
    )
    data = _xlsx_bytes(
        ["Zone", "Code", "VRI_Code", "VRI"],
        [
            ["Ж-1", "Основной", "2.1", "Для ИЖС"],
            ["Ж-1", "Условно разрешенный", "4.7", "Гостиницы"],
            ["П-1", "Вспомогательный", "4.9", "Гаражи"],
        ],
    )
    client = TestClient(app_module.app)
    resp = client.post(
        "/pzz/zone-descriptions/convert",
        files={
            "file": (
                "z.xlsx",
                data,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["report"]["zones_count"] == 2
    assert body["columns_detected"]["zone_code"]["column"] == "Zone"
    assert body["columns_detected"]["permission"]["column"] == "Code"
    zh1 = next(z for z in body["zones"] if z["zone_code"] == "Ж-1")
    assert [v["vri_code"] for v in zh1["main"]] == ["2.1"]
    assert [v["vri_code"] for v in zh1["conditional"]] == ["4.7"]


class _ScriptedOllama:
    """Async-context Ollama stub returning a fixed column-mapping result."""

    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def complete_json(self, messages, schema=None, model=None):
        return self._result


def test_convert_endpoint_llm_maps_nonstandard_headers(monkeypatch):
    scripted = _ScriptedOllama(
        {
            "zone_code": {
                "column": "Территориальная зона",
                "reason": "короткий индекс",
            },
            "zone_name": {"column": None, "reason": "нет"},
            "permission": {"column": "Разрешение", "reason": "категории"},
            "vri_code": {"column": "Индекс ВРИ", "reason": "числа с точками"},
            "vri_name": {"column": "Наименование ВРИ", "reason": "длинный текст"},
        }
    )
    monkeypatch.setattr(zd, "build_chat_llm_client", lambda settings=None: scripted)
    data = _xlsx_bytes(
        ["Территориальная зона", "Разрешение", "Индекс ВРИ", "Наименование ВРИ"],
        [
            ["Ж-1", "основной", "2.1", "Для ИЖС"],
            ["Ж-1", "условно разрешённый", "4.7", "Гостиницы"],
        ],
    )
    client = TestClient(app_module.app)
    resp = client.post(
        "/pzz/zone-descriptions/convert",
        files={"file": ("z.xlsx", data, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns_detected"]["permission"]["column"] == "Разрешение"
    assert body["columns_detected"]["permission"]["source"] == "llm"
    assert body["columns_detected"]["vri_code"]["column"] == "Индекс ВРИ"
    zh1 = next(z for z in body["zones"] if z["zone_code"] == "Ж-1")
    assert [v["vri_code"] for v in zh1["main"]] == ["2.1"]
    assert [v["vri_code"] for v in zh1["conditional"]] == ["4.7"]


def test_convert_endpoint_offline_backstop(monkeypatch):
    """LLM unreachable (returns nothing) → heuristic backstop still resolves standard headers."""
    monkeypatch.setattr(
        zd, "build_chat_llm_client", lambda settings=None: _FakeOllama()
    )
    data = _xlsx_bytes(
        ["Zone", "Code", "VRI_Code", "VRI"],
        [["Ж-1", "Основной", "2.1", "Для ИЖС"]],
    )
    client = TestClient(app_module.app)
    resp = client.post(
        "/pzz/zone-descriptions/convert",
        files={"file": ("z.xlsx", data, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns_detected"]["zone_code"]["column"] == "Zone"
    assert body["columns_detected"]["zone_code"]["source"] == "heuristic"


def test_convert_endpoint_bad_format(monkeypatch):
    monkeypatch.setattr(
        zd, "build_chat_llm_client", lambda settings=None: _FakeOllama()
    )
    client = TestClient(app_module.app)
    resp = client.post(
        "/pzz/zone-descriptions/convert",
        files={"file": ("z.docx", b"nope", "application/octet-stream")},
    )
    assert resp.status_code == 400


# --- inline table conversion in the auto/chat/stream building flow ------------


class _NoLLMOllama:
    """Async-context Ollama stub with no column suggestions (heuristic backstop)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def complete_json(self, messages, schema=None, model=None):
        return {}


def _run_convert_helper(monkeypatch, upload):
    import asyncio
    from types import SimpleNamespace
    import service.api.classifier as clf

    monkeypatch.setattr(
        clf, "build_chat_llm_client", lambda settings=None: _NoLLMOllama()
    )
    return asyncio.run(
        clf._convert_descriptions_if_table(upload, SimpleNamespace(), None)
    )


def test_inline_table_converted_to_json_upload(monkeypatch):
    from fastapi import UploadFile

    data = _xlsx_bytes(
        ["Zone", "Code", "VRI_Code", "VRI"],
        [
            ["Ж-1", "Основной", "2.1", "Для ИЖС"],
            ["П-1", "Вспомогательный", "6.9", "Склад"],
        ],
    )
    up = UploadFile(file=io.BytesIO(data), filename="descr.xlsx")
    new_file, note = _run_convert_helper(monkeypatch, up)
    assert "descriptions_from_table" in new_file.filename
    payload = json.loads(new_file.file.read().decode("utf-8"))
    assert {z["zone_code"] for z in payload} == {"Ж-1", "П-1"}
    assert "2 зон" in note
    assert "ВРИ" not in note and "ПЗЗ" not in note  # spelled out before definition
    assert "Подобранные колонки таблицы" in note
    assert "код зоны — «Zone»" in note


def test_inline_json_descriptions_pass_through(monkeypatch):
    from fastapi import UploadFile

    up = UploadFile(file=io.BytesIO(b"[]"), filename="descr.json")
    out, note = _run_convert_helper(monkeypatch, up)
    assert out is up and note == ""


def test_inline_none_descriptions_pass_through(monkeypatch):
    out, note = _run_convert_helper(monkeypatch, None)
    assert out is None and note == ""


def test_inline_table_missing_required_column_raises(monkeypatch):
    from fastapi import UploadFile
    import service.api.classifier as clf

    up = UploadFile(
        file=io.BytesIO(_xlsx_bytes(["Zone"], [["Ж-1"]])), filename="descr.xlsx"
    )
    try:
        _run_convert_helper(monkeypatch, up)
    except clf._DescriptionsTableError:
        pass
    else:
        raise AssertionError("expected _DescriptionsTableError")
