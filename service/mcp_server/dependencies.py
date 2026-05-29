"""Module-level singletons + getter functions for fastmcp ``Depends``.

The pattern mirrors the gMART idu_mcp project: ``init_dependencies()``
populates a dict at import time; tiny getter functions resolve the
needed instance and are wired into tools via ``fastmcp.Depends``.
"""
from __future__ import annotations

from typing import Any

from .api_client import ApiClient
from .config import McpConfig, load_config


_DEPS: dict[str, Any] = {}


def init_dependencies() -> None:
    """Build singletons. Called once at MCP server startup."""
    if _DEPS:
        return
    cfg = load_config()
    _DEPS["config"] = cfg
    _DEPS["api_client"] = ApiClient(base_url=cfg.api_base_url)


async def shutdown_dependencies() -> None:
    """Close httpx connections cleanly on shutdown."""
    api_client = _DEPS.get("api_client")
    if api_client is not None:
        await api_client.close()


def get_config() -> McpConfig:
    return _DEPS["config"]


def get_api_client() -> ApiClient:
    return _DEPS["api_client"]
