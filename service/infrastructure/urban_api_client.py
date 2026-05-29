"""Async HTTP client for IDU urban_api.

Wraps the three endpoints we consume from the ``/scenarios/*`` flow:

- ``GET /scenarios/{id}/functional_zone_sources`` — list of (year, source)
  pairs available for the scenario.
- ``GET /scenarios/{id}/functional_zones`` — GeoJSON FeatureCollection of
  the functional zones (one feature per zone polygon).
- ``GET /scenarios/{id}/physical_objects_with_geometry`` — GeoJSON
  FeatureCollection of physical objects with their geometry. NOTE: the
  plain ``/physical_objects`` endpoint does NOT return geometry, so we
  must use the ``_with_geometry`` variant for any spatial work.

Auth: pass the incoming user's ``Authorization: Bearer ...`` header
through unchanged.
"""
from __future__ import annotations

from typing import Any

import httpx


class UrbanApiError(RuntimeError):
    """Non-2xx response from urban_api."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"urban_api returned {status}: {body!r}")


class UrbanApiClient:
    """Thin async wrapper. One instance per request — auth header is per-call."""

    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        if not base_url:
            raise RuntimeError(
                "urban_api_base_url is not configured. Set URBAN_API_BASE_URL "
                "in the environment to enable /scenarios/* endpoints."
            )
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "UrbanApiClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    @staticmethod
    def _auth_headers(token: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"} if token else {}


    async def get_scenario_info(
        self, scenario_id: int, *, token: str | None = None
    ) -> dict[str, Any]:
        """Return the scenario descriptor (id, project, name, timestamps).

        We use the ``updated_at`` timestamp to invalidate cached
        classification results when the scenario's data changes
        upstream — see ``/scenarios/{id}/classify``.
        """
        resp = await self._client.get(
            f"/api/v1/scenarios/{scenario_id}",
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)

    async def list_functional_zone_sources(
        self, scenario_id: int, *, token: str | None = None
    ) -> list[dict[str, Any]]:
        """Return [{year, source}, ...] available for a scenario."""
        resp = await self._client.get(
            f"/api/v1/scenarios/{scenario_id}/functional_zone_sources",
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)

    async def get_functional_zones(
        self,
        scenario_id: int,
        *,
        year: int,
        source: str,
        functional_zone_type_id: int | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Return GeoJSON FeatureCollection of zones filtered by year/source."""
        params: dict[str, Any] = {"year": year, "source": source}
        if functional_zone_type_id is not None:
            params["functional_zone_type_id"] = functional_zone_type_id
        resp = await self._client.get(
            f"/api/v1/scenarios/{scenario_id}/functional_zones",
            params=params,
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)

    async def get_physical_objects_with_geometry(
        self,
        scenario_id: int,
        *,
        physical_object_type_id: int | None = None,
        physical_object_function_id: int | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Return GeoJSON FeatureCollection of physical objects with geometry.

        Use ``physical_object_type_id=4`` to filter to residential buildings.
        """
        params: dict[str, Any] = {}
        if physical_object_type_id is not None:
            params["physical_object_type_id"] = physical_object_type_id
        if physical_object_function_id is not None:
            params["physical_object_function_id"] = physical_object_function_id
        resp = await self._client.get(
            f"/api/v1/scenarios/{scenario_id}/physical_objects_with_geometry",
            params=params,
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)


    @staticmethod
    def _json_or_raise(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise UrbanApiError(resp.status_code, body)
        return resp.json()
