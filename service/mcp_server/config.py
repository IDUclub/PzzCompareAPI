"""MCP server config — read from env, no fancy validation."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class McpConfig:
    """Settings for the MCP server process."""

    api_base_url: str
    host: str
    port: int
    workers: int


def load_config() -> McpConfig:
    return McpConfig(
        api_base_url=os.getenv("MCP_API_BASE_URL", "http://api:8000").rstrip("/"),
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8765")),
        workers=int(os.getenv("MCP_WORKERS", "1")),
    )
