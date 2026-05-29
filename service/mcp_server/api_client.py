"""Thin async HTTP client wrapping the PZZ Pipeline REST API.

One responsibility: turn typed Python calls into HTTP requests against
``MCP_API_BASE_URL`` and parse the JSON response. Tools depend on this
class instead of going to httpx directly so error-handling, timeouts and
auth are centralised.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


class ApiError(RuntimeError):
    """Raised when the upstream API returns a non-2xx response."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"upstream API returned {status}: {body!r}")


class ApiClient:
    """Async HTTP wrapper. Reuses one httpx.AsyncClient per process."""

    def __init__(self, base_url: str, timeout_seconds: float = 60.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()


    async def get_task(self, external_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/tasks/{external_id}")
        return self._json_or_raise(resp)

    async def list_tasks(
        self, status: str | None = None, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        resp = await self._client.get("/tasks_list", params=params)
        return self._json_or_raise(resp)

    async def get_task_events(self, external_id: str) -> list[dict[str, Any]]:
        resp = await self._client.get(f"/tasks/{external_id}/events")
        data = self._json_or_raise(resp)
        return data if isinstance(data, list) else []

    async def get_task_result(self, external_id: str) -> dict[str, Any]:
        """Download the result GeoJSON and return it as a parsed dict."""
        resp = await self._client.get(f"/tasks/{external_id}/result")
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, self._safe_body(resp))
        return json.loads(resp.content.decode("utf-8"))

    async def cancel_task(self, external_id: str) -> dict[str, Any]:
        resp = await self._client.delete(f"/tasks/{external_id}")
        return self._json_or_raise(resp)

    async def recompute_task(self, external_id: str) -> dict[str, Any]:
        resp = await self._client.post(f"/tasks/{external_id}/recompute")
        return self._json_or_raise(resp)


    async def submit_pzz_check(
        self,
        *,
        cadastral_geojson: dict[str, Any],
        pzz_zones_geojson: dict[str, Any],
        cadastral_vri_col: str,
        pzz_zone_code_col: str,
        pzz_zone_name_col: str,
        priority: int = 1,
        force_recompute: bool = False,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        files = {
            "cadastral_feature_collection_file": (
                "cadastral.geojson",
                json.dumps(cadastral_geojson).encode("utf-8"),
                "application/geo+json",
            ),
            "pzz_zones_feature_collection_file": (
                "pzz_zones.geojson",
                json.dumps(pzz_zones_geojson).encode("utf-8"),
                "application/geo+json",
            ),
        }
        data: dict[str, Any] = {
            "cadastral_vri_col": cadastral_vri_col,
            "pzz_zone_code_col": pzz_zone_code_col,
            "pzz_zone_name_col": pzz_zone_name_col,
            "priority": str(priority),
            "force_recompute": "true" if force_recompute else "false",
        }
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        resp = await self._client.post(
            "/tasks/pzz-check", files=files, data=data, headers=headers
        )
        return self._json_or_raise(resp)

    async def submit_classify_only(
        self,
        *,
        cadastral_geojson: dict[str, Any],
        cadastral_vri_col: str,
        priority: int = 1,
        force_recompute: bool = False,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        files = {
            "cadastral_feature_collection_file": (
                "cadastral.geojson",
                json.dumps(cadastral_geojson).encode("utf-8"),
                "application/geo+json",
            ),
        }
        data: dict[str, Any] = {
            "cadastral_vri_col": cadastral_vri_col,
            "priority": str(priority),
            "force_recompute": "true" if force_recompute else "false",
        }
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        resp = await self._client.post(
            "/tasks/classify-only", files=files, data=data, headers=headers
        )
        return self._json_or_raise(resp)


    @staticmethod
    def _bearer(token: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def submit_scenario_classify(
        self,
        *,
        scenario_id: int,
        year: int,
        source: str,
        physical_object_type_id: int = 4,
        priority: int = 1,
        force_recompute: bool = False,
        token: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "year": str(year),
            "source": source,
            "physical_object_type_id": str(physical_object_type_id),
            "priority": str(priority),
            "force_recompute": "true" if force_recompute else "false",
        }
        resp = await self._client.post(
            f"/scenarios/{scenario_id}/classify",
            data=data,
            headers=self._bearer(token),
        )
        return self._json_or_raise(resp)

    async def get_scenario_zones_info(
        self, *, scenario_id: int, year: int, source: str, token: str | None = None
    ) -> dict[str, Any]:
        resp = await self._client.get(
            f"/scenarios/{scenario_id}/zones-info",
            params={"year": year, "source": source},
            headers=self._bearer(token),
        )
        return self._json_or_raise(resp)

    async def get_scenario_task(
        self, *, scenario_id: int, external_id: str, token: str | None = None
    ) -> dict[str, Any]:
        resp = await self._client.get(
            f"/scenarios/{scenario_id}/tasks/{external_id}",
            headers=self._bearer(token),
        )
        return self._json_or_raise(resp)

    async def get_scenario_object_zone_fit(
        self,
        *,
        scenario_id: int,
        external_id: str,
        group_by: str = "zone",
        token: str | None = None,
    ) -> dict[str, Any]:
        resp = await self._client.get(
            f"/scenarios/{scenario_id}/tasks/{external_id}/object-zone-fit",
            params={"group_by": group_by},
            headers=self._bearer(token),
        )
        return self._json_or_raise(resp)

    async def recompute_scenario_task(
        self, *, scenario_id: int, external_id: str, token: str | None = None
    ) -> dict[str, Any]:
        resp = await self._client.post(
            f"/scenarios/{scenario_id}/tasks/{external_id}/recompute",
            headers=self._bearer(token),
        )
        return self._json_or_raise(resp)


    def _json_or_raise(self, resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, self._safe_body(resp))
        return resp.json()

    @staticmethod
    def _safe_body(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return resp.text
