"""Smoke test: call one ms-inspect tool over HTTP.

Requires ms-inspect to be running:
  RADIO_MCP_TRANSPORT=http RADIO_MCP_PORT=8000 python -m ms_inspect.server

Run with:
  pytest tests/test_tools.py -v
or from the container:
  podman exec wildcat pytest tests/test_tools.py -v
"""

from __future__ import annotations

import json
import os

import pytest

from wildcat.tools import MSInspectClient

MS_INSPECT_URL = os.environ.get("MS_INSPECT_URL", "http://localhost:8000")
TEST_MS_PATH = os.environ.get("TEST_MS_PATH", "/data/test.ms")


@pytest.mark.asyncio
async def test_list_tools_returns_schemas():
    """ms-inspect should return a non-empty list of tool schemas."""
    async with MSInspectClient(MS_INSPECT_URL) as client:
        tools = await client.list_tools()

    assert isinstance(tools, list), "list_tools() should return a list"
    assert len(tools) > 0, "ms-inspect should expose at least one tool"

    # Each tool should have a name
    names = [t.get("name") for t in tools]
    assert all(names), "every tool schema should have a 'name' key"

    print(f"\nAvailable tools ({len(tools)}):")
    for name in sorted(names):
        print(f"  {name}")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.path.exists(TEST_MS_PATH),
    reason=f"TEST_MS_PATH={TEST_MS_PATH!r} does not exist — skipping live tool call",
)
async def test_ms_observation_info_returns_dict():
    """Call ms_observation_info and verify the response is a dict with content."""
    async with MSInspectClient(MS_INSPECT_URL) as client:
        result = await client.call_tool(
            "ms_observation_info", {"ms_path": TEST_MS_PATH}
        )

    assert isinstance(result, dict), "call_tool should return a dict"
    assert result, "result should not be empty"
    assert "error" not in result, f"Tool returned an error: {result}"

    print("\nms_observation_info response:")
    print(json.dumps(result, indent=2)[:500])


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.path.exists(TEST_MS_PATH),
    reason=f"TEST_MS_PATH={TEST_MS_PATH!r} does not exist — skipping phase 1 run",
)
async def test_run_phase1_returns_all_tools():
    """run_phase1 should return a dict keyed by all 6 Phase 1 tool names."""
    from wildcat.tools import _PHASE1_TOOLS

    async with MSInspectClient(MS_INSPECT_URL) as client:
        results = await client.run_phase1(TEST_MS_PATH)

    assert set(results.keys()) == set(_PHASE1_TOOLS), (
        f"Expected tools {_PHASE1_TOOLS}, got {list(results.keys())}"
    )
    for tool_name, output in results.items():
        assert isinstance(output, dict), f"{tool_name} should return a dict"
