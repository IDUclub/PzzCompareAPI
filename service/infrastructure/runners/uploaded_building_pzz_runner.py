"""Deterministic PZZ check for user-uploaded buildings — no LLM, no embeddings.

The building counterpart of the cadastral upload flow: instead of a parcel's VRI
text, the user uploads *buildings* (Urban-API-shaped: ``physical_object_type_id``
/ ``service_type_id`` or their text names/codes + floors) plus their own PZZ
zones, and optionally a zone descriptions file. Each building is resolved to a
VRI code and tested against its containing zone's permitted-VRI set — the same
verdict logic as the scenario runner (see ``_deterministic_pzz``), driven by
user-named columns instead of urban_api's fixed schema.

Resolution priority for a building's VRI:
    1. residential  -> floor-band VRI (uses the floors column);
    2. service       -> service_type_id/name/code → VRI (service_type_to_vri.json);
    3. otherwise     -> physical_object_type_id/name → VRI.

Zone permitted-VRI set comes from the uploaded descriptions file when supplied,
else the built-in fz_to_pzz mapping (fallback). Heavy geo deps load lazily.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from service.domain import PipelineRequest
from service.infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError
from service.infrastructure.runners._deterministic_pzz import (
    CATEGORY_BUILDING,
    CATEGORY_SERVICE,
    build_zone_gdf,
    clean_result_properties,
    join_objects_to_zones,
    load_zone_mapping,
    resolve_po_type_vri,
    verdict as compute_verdict,
)
from service.infrastructure.runners.pipeline_runner import PipelineRunner, _build_output_glob
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.settings import Settings

logger = logging.getLogger("service.tasks")

_FLOORS_FIELD = "Количество этажей"
_RESIDENTIAL_PO_TYPE = 4  # urban_api "жилой дом"
_RESIDENTIAL_TEXT = ("жил", "residential", "жилое", "жилой")
_COMMON_SERVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "21": ("детсад", "детский садик", "дошкольное учреждение", "доу"),
    "22": ("сош", "общеобразовательная школа"),
}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_alias(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).casefold().replace("ё", "е")
    return "".join(ch for ch in text if ch.isalnum())


def _add_alias(alias_map: dict[str, str | None], value: Any, object_id: str) -> None:
    key = _normalise_alias(value)
    if not key:
        return
    if key not in alias_map:
        alias_map[key] = object_id
    elif alias_map[key] != object_id:
        alias_map[key] = None


def _lookup_alias(alias_map: dict[str, str | None], value: Any) -> int | None:
    key = _normalise_alias(value)
    if not key:
        return None

    exact = alias_map.get(key)
    if exact:
        return _as_int(exact)

    prefix_matches = {
        object_id
        for alias, object_id in alias_map.items()
        if object_id and len(alias) >= 5 and key.startswith(alias)
    }
    if len(prefix_matches) == 1:
        return _as_int(next(iter(prefix_matches)))
    return None


def _dict_value(raw: Any, *keys: str) -> Any:
    if not isinstance(raw, dict):
        return raw
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return None


_LLM_NOTE = " (название сопоставлено ИИ)"


def _llm_note(from_llm: bool) -> str:
    return _LLM_NOTE if from_llm else ""


class UploadedBuildingPzzRunner(PipelineRunner):
    """Classify user-uploaded buildings against user-uploaded PZZ zones."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._po2vri = json.loads(
            Path(settings.physical_object_type_to_vri_path).read_text(encoding="utf-8")
        )
        raw_service = json.loads(
            Path(settings.service_type_to_vri_path).read_text(encoding="utf-8")
        )
        self._service_map: dict[str, Any] = raw_service.get("by_service_type_id", {})
        self._service_aliases = self._build_service_aliases()
        self._po_type_aliases = self._build_po_type_aliases()
        # Per-run LLM fallback: {normalised text name -> catalogue id} for names
        # that don't resolve deterministically. Reset at the start of each run().
        self._llm_type_overrides: dict[str, int] = {}
        self._llm_service_overrides: dict[str, int] = {}

    def _build_service_aliases(self) -> dict[str, str | None]:
        aliases: dict[str, str | None] = {}
        for service_type_id, entry in self._service_map.items():
            _add_alias(aliases, service_type_id, service_type_id)
            _add_alias(aliases, entry.get("name"), service_type_id)
            _add_alias(aliases, entry.get("code"), service_type_id)
            for alias in entry.get("aliases") or ():
                _add_alias(aliases, alias, service_type_id)
            for alias in _COMMON_SERVICE_ALIASES.get(str(service_type_id), ()):
                _add_alias(aliases, alias, service_type_id)
        return aliases

    def _build_po_type_aliases(self) -> dict[str, str | None]:
        aliases: dict[str, str | None] = {}
        for po_type_id, entry in self._po2vri.get("by_type_id", {}).items():
            _add_alias(aliases, po_type_id, po_type_id)
            _add_alias(aliases, entry.get("name"), po_type_id)
            for alias in entry.get("aliases") or ():
                _add_alias(aliases, alias, po_type_id)
        return aliases

    def _load_zone_mapping(self, request: PipelineRequest):
        """User descriptions file when supplied+usable, else built-in fallback."""
        descriptions_path = request.pzz_zone_vri_labels_path
        if descriptions_path and Path(descriptions_path).is_file():
            try:
                allowed, nick = load_zone_mapping(descriptions_path)
                if allowed:
                    return allowed, nick
                logger.warning(
                    "uploaded zone descriptions had no usable mappings; "
                    "falling back to built-in fz_to_pzz mapping"
                )
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning("failed to read uploaded zone descriptions (%s); fallback", exc)
        return load_zone_mapping(self._settings.default_fz_to_pzz_mapping_path)

    # --- LLM name fallback -----------------------------------------------------
    def _needs_llm(self, value: Any, aliases: dict[str, str | None]) -> bool:
        """True when ``value`` is a text name that resolves neither as an id nor
        via the deterministic alias map (a numeric unknown id is NOT an LLM case
        — the model maps names, not invented ids)."""
        if not isinstance(value, str) or not _normalise_alias(value):
            return False
        return _as_int(value) is None and _lookup_alias(aliases, value) is None

    def _llm_complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        """Blocking structured LLM call (seam for tests). Runs the async Ollama
        client in a private event loop — the runner executes in a worker thread
        with no running loop."""
        async def _run() -> dict[str, Any]:
            async with OllamaChatClient(
                base_url=self._settings.ollama_base_url,
                default_model=self._settings.chat_model or self._settings.generate_model,
            ) as client:
                return await client.complete_json(messages, schema=schema)

        return asyncio.run(_run())

    def _llm_map_names(
        self, names: list[str], catalogue: dict[str, Any], kind_label: str
    ) -> dict[str, int]:
        """Map unknown text ``names`` to the closest catalogue id via one LLM call.

        Returns ``{normalised_name -> id}`` for confident hits; names the model
        can't place (returns null) or invalid ids are dropped -> manual review.
        """
        if not names:
            return {}
        id_enum = list(catalogue.keys())
        catalogue_lines = "\n".join(
            f"- {cid}: {entry.get('name')}" for cid, entry in catalogue.items() if entry.get("name")
        )
        indexed = {f"n{i}": name for i, name in enumerate(names)}
        schema = {
            "type": "object",
            "properties": {
                key: {"type": ["string", "null"], "enum": [*id_enum, None]}
                for key in indexed
            },
            "required": list(indexed),
            "additionalProperties": False,
        }
        listing = "\n".join(f"{key} = «{name}»" for key, name in indexed.items())
        messages = [
            {"role": "system", "content": (
                "Ты сопоставляешь произвольные названия объектов с каталогом "
                f"({kind_label}). Для каждого названия верни id ближайшей по смыслу "
                "записи каталога ИЗ ПРЕДЛОЖЕННОГО СПИСКА id, либо null, если "
                "подходящей записи нет. Никогда не придумывай id. Ответ строго JSON "
                "по схеме."
            )},
            {"role": "user", "content": (
                f"Каталог ({kind_label}) — id: название:\n{catalogue_lines}\n\n"
                f"Названия для сопоставления:\n{listing}"
            )},
        ]
        try:
            parsed = self._llm_complete(messages, schema)
        except OllamaChatError as exc:
            logger.warning("building LLM name fallback failed (%s); names -> manual review", exc)
            return {}
        out: dict[str, int] = {}
        for key, name in indexed.items():
            raw = parsed.get(key)
            cid = _as_int(raw)
            if cid is not None and str(cid) in catalogue:
                out[_normalise_alias(name)] = cid
        return out

    def _resolve_unknown_names_via_llm(self, feats: list[dict[str, Any]], request: PipelineRequest) -> None:
        """Populate the per-run override maps for text type/service names that
        don't resolve deterministically. No-op when disabled or nothing unknown."""
        self._llm_type_overrides = {}
        self._llm_service_overrides = {}
        if not getattr(self._settings, "building_llm_name_fallback", True):
            return

        type_names: dict[str, str] = {}
        service_names: dict[str, str] = {}
        for feature in feats:
            props = feature.get("properties") or {}
            type_raw, service_raw = self._raw_type_service(props, request)
            # residential text ("жилой дом") resolves via the floor band already.
            if self._needs_llm(type_raw, self._po_type_aliases):
                low = type_raw.strip().lower()
                if not any(low.startswith(t) for t in _RESIDENTIAL_TEXT):
                    type_names.setdefault(_normalise_alias(type_raw), type_raw)
            if self._needs_llm(service_raw, self._service_aliases):
                service_names.setdefault(_normalise_alias(service_raw), service_raw)

        if type_names:
            self._llm_type_overrides = self._llm_map_names(
                list(type_names.values()), self._po2vri.get("by_type_id", {}), "тип физического объекта"
            )
        if service_names:
            self._llm_service_overrides = self._llm_map_names(
                list(service_names.values()), self._service_map, "тип сервиса"
            )
        if self._llm_type_overrides or self._llm_service_overrides:
            logger.info(json.dumps({
                "stage": "uploaded_building_pzz", "status": "llm_name_fallback",
                "external_id": request.task_external_id,
                "types_resolved": len(self._llm_type_overrides),
                "services_resolved": len(self._llm_service_overrides),
            }))

    def _raw_type_service(self, props: dict[str, Any], request: PipelineRequest):
        """Return the raw (type_value, service_value) from a feature's properties.

        Reads the configured columns and tolerates the urban_api nested shapes
        (``physical_object_type`` / ``service_type`` objects), yielding either a
        numeric id or a text name/code — the same raw form the aliases and the
        LLM fallback both consume.
        """
        type_raw = props.get(request.building_type_col) if request.building_type_col else None
        type_raw = _dict_value(type_raw, "physical_object_type_id", "id", "name", "code")
        if type_raw is None:
            type_raw = _dict_value(
                props.get("physical_object_type"),
                "physical_object_type_id", "id", "name", "code",
            )
        service_raw = props.get(request.building_service_col) if request.building_service_col else None
        service_raw = _dict_value(service_raw, "service_type_id", "id", "name", "code")
        if service_raw is None:
            service_raw = _dict_value(
                props.get("service_type"), "service_type_id", "id", "name", "code"
            )
        return type_raw, service_raw

    def _extract(self, props: dict[str, Any], request: PipelineRequest):
        """Return (po_type_id, is_residential, service_type_id, floors, label, llm_flags).

        ``llm_flags`` = (type_from_llm, service_from_llm): whether the id was
        recovered via the LLM name fallback rather than a deterministic match.
        """
        nested = props.get("properties") if isinstance(props.get("properties"), dict) else {}
        type_raw, service_raw = self._raw_type_service(props, request)

        floors = props.get(request.building_floors_col) if request.building_floors_col else None
        if floors is None:
            floors = nested.get(_FLOORS_FIELD, props.get(_FLOORS_FIELD))

        # Deterministic first: numeric id, then exact/prefix alias match. Only if
        # both miss do we consult the per-run LLM override (text names only).
        type_from_llm = False
        po_type_id = _as_int(type_raw) or _lookup_alias(self._po_type_aliases, type_raw)
        if po_type_id is None:
            override = self._llm_type_overrides.get(_normalise_alias(type_raw))
            if override is not None:
                po_type_id, type_from_llm = override, True

        is_residential = po_type_id == _RESIDENTIAL_PO_TYPE
        if po_type_id is None and isinstance(type_raw, str):
            low = type_raw.strip().lower()
            if any(low.startswith(t) for t in _RESIDENTIAL_TEXT):
                is_residential = True

        service_from_llm = False
        service_type_id = _as_int(service_raw) or _lookup_alias(self._service_aliases, service_raw)
        if service_type_id is None:
            override = self._llm_service_overrides.get(_normalise_alias(service_raw))
            if override is not None:
                service_type_id, service_from_llm = override, True

        label = " / ".join(
            str(x) for x in (type_raw, service_raw) if x not in (None, "")
        ) or None
        return po_type_id, is_residential, service_type_id, floors, label, (type_from_llm, service_from_llm)

    def _resolve_vri(
        self,
        po_type_id: int | None,
        is_residential: bool,
        service_type_id: int | None,
        floors: Any,
        type_from_llm: bool = False,
        service_from_llm: bool = False,
    ) -> tuple[str | None, str | None, str]:
        """Return ``(vri_code, vri_name, basis)`` — ``basis`` records HOW the ВРИ
        was picked (by floors / service type / object type) for the report; a
        suffix flags names recovered by the LLM fallback rather than the catalogue."""
        if is_residential:
            code, name = resolve_po_type_vri(self._po2vri, _RESIDENTIAL_PO_TYPE, floors)
            if code:
                floors_txt = f", {floors} эт." if floors not in (None, "") else " (этажность не указана)"
                return code, name, f"жилое здание — ВРИ подобран по этажности{floors_txt}{_llm_note(type_from_llm)}"
        if service_type_id is not None:
            entry = self._service_map.get(str(service_type_id))
            if entry and entry.get("vri_code"):
                return (
                    entry["vri_code"],
                    entry.get("vri_name") or None,
                    f"сервис (service_type_id={service_type_id}) — ВРИ подобран по типу сервиса{_llm_note(service_from_llm)}",
                )
        if po_type_id is not None:
            code, name = resolve_po_type_vri(self._po2vri, po_type_id, floors)
            if code:
                return (
                    code,
                    name,
                    f"физический объект (physical_object_type_id={po_type_id}) — ВРИ подобран по типу объекта{_llm_note(type_from_llm)}",
                )
        return None, None, ""

    def run(self, request: PipelineRequest) -> str:
        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        buildings = json.loads(Path(request.cadastral_data_path).read_text(encoding="utf-8"))
        zones = json.loads(Path(request.pzz_zones_data_path).read_text(encoding="utf-8"))
        code_col = request.pzz_zone_code_col or "zone_code"
        zone_allowed, zone_nick = self._load_zone_mapping(request)

        zgdf = build_zone_gdf(zones, code_col)
        feats = [f for f in (buildings.get("features") or []) if f.get("geometry") is not None]
        fz_by_obj = join_objects_to_zones(feats, zgdf)

        # Deterministic-first: recover text type/service names that don't match the
        # catalogue via one batched LLM call each; ids/known names never reach it.
        self._resolve_unknown_names_via_llm(feats, request)

        for i, feature in enumerate(feats):
            props = feature.get("properties") or {}
            po_type_id, is_residential, service_type_id, floors, label, llm_flags = self._extract(props, request)
            vri, vri_name, vri_basis = self._resolve_vri(
                po_type_id, is_residential, service_type_id, floors, *llm_flags
            )

            fz = fz_by_obj.get(i)
            machine_verdict, reason, mcode, _ = compute_verdict(vri, fz, zone_allowed, zone_nick)
            # Split key for the two download layers: a «Сервис» is any row whose
            # service column is populated and isn't residential — keyed on the raw
            # value, not on resolution, so an unresolved service name (manual review)
            # still lands in the services layer rather than among the buildings.
            _, service_raw = self._raw_type_service(props, request)
            category = (
                CATEGORY_SERVICE
                if service_raw not in (None, "") and not is_residential
                else CATEGORY_BUILDING
            )
            feature["properties"] = clean_result_properties(
                vri_text=label,
                fz_type_id=fz,
                zone_nick=zone_nick,
                machine_verdict=machine_verdict,
                reason=reason,
                matched_vri_code=mcode,
                matched_vri_name=vri_name,
                resolution_basis=vri_basis,
                category=category,
            )

        result = {"type": "FeatureCollection", "features": feats}
        out_path = output_dir / f"pzz_compare_spatial_first_{request.task_external_id}.geojson"
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        logger.info(
            json.dumps({
                "stage": "uploaded_building_pzz", "status": "finished",
                "external_id": request.task_external_id,
                "buildings": len(feats), "zones": len(zgdf), "matched_zone": len(fz_by_obj),
            })
        )
        return _build_output_glob(output_dir, request.task_external_id)
