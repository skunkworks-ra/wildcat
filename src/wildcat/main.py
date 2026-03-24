"""Entry point for wildcat.

Wires all components and drives the orchestration loop.
MS path is resolved from (in priority order):
  1. --ms CLI argument
  2. WILDCAT_MS_PATH environment variable
  3. Error — must be provided
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn

from wildcat.config import load_config
from wildcat.llm import LLMBackend
from wildcat.orchestrator import Orchestrator
from wildcat.runner import CASARunner
from wildcat.state import StateDB
from wildcat.tools import MSInspectClient
from wildcat.ui.app import build_ui_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wildcat",
        description="Agentic orchestration layer for radio interferometry reduction",
    )
    parser.add_argument(
        "--ms",
        metavar="PATH",
        help="Path to the Measurement Set (.ms directory)",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="config.toml",
        help="Path to config.toml (default: config.toml)",
    )
    parser.add_argument(
        "--workflow-id",
        type=int,
        default=None,
        help="Resume an existing workflow by ID (default: create new)",
    )
    parser.add_argument(
        "--autostart",
        action="store_true",
        default=False,
        help="Start the pipeline immediately without waiting for UI confirmation",
    )
    return parser.parse_args()


def _resolve_ms_path(args: argparse.Namespace) -> str:
    ms_path = args.ms or os.environ.get("WILDCAT_MS_PATH")
    if not ms_path:
        sys.exit(
            "error: Measurement Set path required.\n"
            "  Use --ms /path/to/data.ms  or  WILDCAT_MS_PATH=/path/to/data.ms"
        )
    return ms_path


async def main_async() -> None:
    args = _parse_args()
    ms_path = _resolve_ms_path(args)

    cfg = load_config(args.config)
    log.info("Loaded config from %s", args.config)
    log.info("LLM backend: %s", cfg.llm.backend)

    # ── Start gate ───────────────────────────────────────────────────────
    # Priority: --autostart flag > WILDCAT_AUTOSTART env var > UI button
    autostart = args.autostart or bool(os.environ.get("WILDCAT_AUTOSTART"))
    start_event = asyncio.Event()
    if autostart:
        start_event.set()
        log.info("Autostart enabled — pipeline will begin immediately")

    # ── State ────────────────────────────────────────────────────────────
    with StateDB(cfg.state.db_path) as db:
        db.init_schema()

        # ── Shared events ─────────────────────────────────────────────────
        checkpoint_event = asyncio.Event()
        stop_event = asyncio.Event()

        # ── UI (starts before pipeline so waiting screen is reachable) ────
        ui_app = build_ui_app(db, checkpoint_event, stop_event, start_event, ms_path)
        ui_config = uvicorn.Config(
            ui_app, host="0.0.0.0", port=cfg.ui.port, log_level="warning"
        )
        ui_server = uvicorn.Server(ui_config)
        ui_task = asyncio.create_task(ui_server.serve())
        log.info("UI listening on http://0.0.0.0:%d", cfg.ui.port)

        if not autostart:
            log.info("Waiting for start signal — visit http://0.0.0.0:%d/start", cfg.ui.port)
            await start_event.wait()
            log.info("Start signal received — beginning pipeline")

        # ── Workflow ──────────────────────────────────────────────────────
        if args.workflow_id is not None:
            workflow_id = args.workflow_id
            wf = db.get_workflow(workflow_id)
            log.info("Resuming workflow %d (stage=%s)", workflow_id, wf["stage"])
        else:
            workflow_id = db.create_workflow(ms_path)
            log.info("Created workflow %d for %s", workflow_id, ms_path)

        # ── LLM ──────────────────────────────────────────────────────────
        llm = LLMBackend(cfg.llm)
        await llm.start()

        # ── Tools ─────────────────────────────────────────────────────────
        async with MSInspectClient(cfg.mcp.base_url) as tools:

            # ── Runner + watcher ─────────────────────────────────────────
            runner = CASARunner(cfg.casa, db)

            # ── Orchestrator ──────────────────────────────────────────────
            orchestrator = Orchestrator(
                db=db,
                tools=tools,
                llm=llm,
                skills_path=cfg.skills.path,
                runner=runner,
                checkpoint_event=checkpoint_event,
                stop_event=stop_event,
            )

            try:
                await orchestrator.run(workflow_id)
            finally:
                ui_server.should_exit = True
                await ui_task
                await llm.stop()

    log.info("wildcat finished (workflow_id=%d)", workflow_id)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
