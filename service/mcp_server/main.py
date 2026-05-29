"""MCP server entry point.

Builds the root FastMCP server, mounts domain sub-servers, exposes a
Starlette ASGI app that uvicorn (or another ASGI runner) can serve.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from .dependencies import init_dependencies, shutdown_dependencies
from .tools.scenarios import scenarios_mcp
from .tools.tasks import tasks_mcp


logger = logging.getLogger("service.mcp_server")


_INSTRUCTIONS = """PZZ Pipeline — classify cadastral / scenario objects against
Land-Use & Development Rules (ПЗЗ) and report which objects sit in the wrong
functional zone.

TWO WAYS TO CLASSIFY:
1. Scenario flow (preferred for agents): the user has a project open in
   urban_api. Pass `scenario_id` explicitly to the scenario tools; the user's
   Bearer token is taken from the Authorization header automatically (you
   never ask for the token). Use classify_scenario / classify_scenario_and_wait,
   then get_scenario_classification_report. Use get_scenario_zones_info to
   explain what may be built in a zone.
2. File flow: the user provides raw GeoJSON (submit_pzz_check_task /
   submit_classify_only_task). Only for small inline datasets.

TYPICAL SCENARIO DIALOG:
  classify_scenario(scenario_id, year, source) -> external_id
  poll get_scenario_classification_status(scenario_id, external_id) until "finished"
  get_scenario_classification_report(scenario_id, external_id) -> relay `chat_message`.

If a tool returns AUTH_TOKEN_EXPIRED, ask the frontend/user for a fresh
token and retry — do not reuse the rejected one."""


@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    init_dependencies()
    logger.info("mcp_server | startup | finished")
    try:
        yield {"started": True}
    finally:
        await shutdown_dependencies()
        logger.info("mcp_server | shutdown | finished")


main_mcp = FastMCP("PZZ Pipeline MCP", instructions=_INSTRUCTIONS, lifespan=mcp_lifespan)
main_mcp.mount(scenarios_mcp)
main_mcp.mount(tasks_mcp)


async def _health(request):  # noqa: ARG001 — Starlette signature
    return JSONResponse({"status": "ok"})


async def _root(request):  # noqa: ARG001
    return RedirectResponse(url="/mcp")


mcp_app = main_mcp.http_app()
mcp_app.routes.insert(0, Route("/health", endpoint=_health))
mcp_app.routes.insert(0, Route("/", endpoint=_root))


if __name__ == "__main__":
    import uvicorn

    from .config import load_config

    cfg = load_config()
    uvicorn.run(
        "service.mcp_server.main:mcp_app",
        host=cfg.host,
        port=cfg.port,
        workers=cfg.workers,
    )
