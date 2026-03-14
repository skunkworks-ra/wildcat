"""MCP SSE client for the ms-inspect server.

ms-inspect serves over SSE transport:
  GET  /sse          — event stream (MCP session)
  POST /messages/    — send requests

Uses the official mcp Python client so the protocol is handled natively.
"""

from __future__ import annotations

import json
import logging

from mcp import ClientSession
from mcp.client.sse import sse_client

log = logging.getLogger(__name__)

# ── Phase tool groupings ───────────────────────────────────────────────────────

_PHASE1_TOOLS = [
    "ms_observation_info",
    "ms_field_list",
    "ms_scan_list",
    "ms_scan_intent_summary",
    "ms_spectral_window_list",
    "ms_correlator_config",
]

_PHASE2_TOOLS = [
    "ms_antenna_list",
    "ms_baseline_lengths",
    "ms_elevation_vs_time",
    "ms_parallactic_angle_vs_time",
    "ms_shadowing_report",
    "ms_antenna_flag_fraction",
]

_PHASE3_TOOLS = [
    "ms_rfi_channel_stats",
    "ms_flag_summary",
    "ms_refant",
    "ms_pol_cal_feasibility",
]


class MSInspectClient:
    """MCP SSE client wrapping ms-inspect.

    Opens a fresh SSE session per call group. Each context manager entry
    connects to /sse and negotiates the MCP session.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._sse_url = f"{self.base_url}/sse"

    async def list_tools(self) -> list[dict]:
        async with sse_client(self._sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [t.model_dump() for t in result.tools]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        log.debug("Calling tool %s with %s", name, arguments)
        async with sse_client(self._sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                # Extract text content and parse JSON
                if result.content:
                    text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return {"raw": text}
                return {}

    async def _run_phase(self, tool_names: list[str], ms_path: str) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for tool_name in tool_names:
            try:
                results[tool_name] = await self.call_tool(tool_name, {"params": {"ms_path": ms_path}})
            except Exception as exc:
                log.warning("Tool %s failed: %s", tool_name, exc)
                results[tool_name] = {"error": str(exc)}
        return results

    async def run_phase1(self, ms_path: str) -> dict[str, dict]:
        return await self._run_phase(_PHASE1_TOOLS, ms_path)

    async def run_phase2(self, ms_path: str) -> dict[str, dict]:
        return await self._run_phase(_PHASE2_TOOLS, ms_path)

    async def run_phase3(self, ms_path: str) -> dict[str, dict]:
        return await self._run_phase(_PHASE3_TOOLS, ms_path)

    # Keep async context manager for compatibility with main.py
    async def __aenter__(self) -> MSInspectClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass
