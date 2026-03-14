"""FastAPI checkpoint UI for wildcat.

Provides a minimal human-in-the-loop interface:
  GET  /checkpoint/{workflow_id}           — render checkpoint page
  POST /checkpoint/{workflow_id}/decide    — record human decision
  GET  /jobs/<path>                        — static file serving for plots

The checkpoint_event is shared with the Orchestrator — posting a decision
fires the event which unblocks orchestrator._handle_checkpoint().
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from wildcat.state import StateDB

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


_LOG_PATH = "/var/log/wildcat.log"


async def _tail_log(log_path: str):
    """Yield SSE-formatted lines by tailing a log file."""
    try:
        with open(log_path) as f:
            f.seek(0, 2)  # start at end of file
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    await asyncio.sleep(0.2)
    except FileNotFoundError:
        yield f"data: [log file not found: {log_path}]\n\n"


def build_ui_app(db: StateDB, checkpoint_event: asyncio.Event) -> FastAPI:
    """Construct the FastAPI application with all routes wired up."""
    app = FastAPI(title="wildcat checkpoint UI")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["tojson"] = lambda v, **kw: json.dumps(v, **kw)

    router = APIRouter()

    @router.get("/checkpoint/{workflow_id}", response_class=HTMLResponse)
    async def show_checkpoint(request: Request, workflow_id: int) -> HTMLResponse:
        """Render the checkpoint review page."""
        try:
            wf = db.get_workflow(workflow_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        checkpoint = db.get_latest_checkpoint(workflow_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"No checkpoint found for workflow {workflow_id}",
            )

        # Collect tool outputs across all phases for the metrics table
        tool_outputs: dict[str, dict] = {}
        for phase in ("phase1", "phase2", "phase3"):
            for row in db.get_tool_outputs(workflow_id, phase):
                try:
                    tool_outputs[row["tool_name"]] = json.loads(row["output_json"])
                except (json.JSONDecodeError, KeyError):
                    tool_outputs[row["tool_name"]] = {"raw": row.get("output_json", "")}

        return templates.TemplateResponse(
            "checkpoint.html",
            {
                "request":      request,
                "workflow":     wf,
                "checkpoint":   checkpoint,
                "tool_outputs": tool_outputs,
            },
        )

    @router.get("/status/{workflow_id}", response_class=HTMLResponse)
    async def status(request: Request, workflow_id: int) -> HTMLResponse:
        """Live status fragment — polled by the UI every 3s via HTMX."""
        try:
            wf = db.get_workflow(workflow_id)
            stage = wf["stage"]
            updated = wf["updated_at"]
        except KeyError:
            return HTMLResponse("<span style='color:#f87171'>Workflow not found</span>")
        cp = db.get_latest_checkpoint(workflow_id)
        color = "#86efac" if stage == "HUMAN_CHECKPOINT" else "#7dd3fc"
        cp_info = f" · checkpoint #{cp['id']}" if cp else " · awaiting checkpoint"
        reload = ""
        if cp:
            reload = "<script>if(!window._reloaded){window._reloaded=true;location.reload();}</script>"
        return HTMLResponse(
            f"<span style='color:{color}'>● {stage}</span>"
            f"<span style='color:#64748b; font-size:0.8rem; margin-left:1rem'>{updated}{cp_info}</span>"
            f"{reload}"
        )

    @router.post("/checkpoint/{workflow_id}/decide")
    async def decide(
        workflow_id: int,
        route: str = Form(...),
        notes: str = Form(""),
    ) -> dict:
        """Record the human decision and unblock the orchestrator."""
        if route not in ("imaging", "calibration"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid route {route!r}. Must be 'imaging' or 'calibration'.",
            )

        checkpoint = db.get_latest_checkpoint(workflow_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"No pending checkpoint for workflow {workflow_id}",
            )

        db.resolve_checkpoint(checkpoint["id"], human_route=route, human_notes=notes)
        log.info(
            "Checkpoint %d resolved: route=%s workflow=%d",
            checkpoint["id"], route, workflow_id,
        )

        # Unblock the orchestrator
        checkpoint_event.set()

        return {"status": "ok", "route": route}

    @router.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request) -> HTMLResponse:
        """Render the live log stream page."""
        return templates.TemplateResponse("logs.html", {"request": request})

    @router.get("/logs/stream")
    async def logs_stream() -> StreamingResponse:
        """SSE endpoint — tails /var/log/wildcat.log and pushes new lines."""
        return StreamingResponse(
            _tail_log(_LOG_PATH),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(router)

    # Serve plot PNGs from /data/jobs/
    jobs_dir = Path("/data/jobs")
    if jobs_dir.exists():
        app.mount("/jobs", StaticFiles(directory=str(jobs_dir)), name="jobs")

    return app
