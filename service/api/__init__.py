"""HTTP API routers grouped by concern.

Each sub-module exposes a single ``router: APIRouter`` that is mounted from
``service.app`` via ``app.include_router(...)``. Endpoints keep their
original URL paths — the split is purely organisational and surfaces in
the Swagger UI as separate tag groups.
"""
