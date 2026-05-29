"""MCP (Model Context Protocol) server for the PZZ Pipeline service.

Exposes the REST API as MCP tools so AI agents (Claude Desktop, Claude
Code, Langchain, custom LLM workflows) can submit classification tasks,
monitor their status, and fetch results.

Architecture mirrors the gMART idu_mcp pattern:
- Each tool domain owns its own ``FastMCP(...)`` instance.
- The root server in ``main.py`` ``mount()``s domain servers and exposes
  an ASGI app via ``http_app()``.
- Tools call the existing FastAPI service over HTTP through an injected
  ``ApiClient`` — no shared DB/Redis with the worker.
"""
