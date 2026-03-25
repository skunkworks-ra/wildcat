"""CASA subprocess runner.

CASARunner.submit() spawns CASA as an asyncio subprocess, streams
stdout/stderr to SQLite in real time, and returns when CASA exits.
The orchestrator awaits submit() directly — no sentinel files or
filesystem watchers needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from wildcat.config import CASAConfig
from wildcat.state import StateDB

log = logging.getLogger(__name__)


class CASARunner:
    """Runs CASA scripts as asyncio subprocesses.

    Each job gets a unique directory under jobs_dir. stdout/stderr are
    streamed to SQLite line by line.
    """

    def __init__(self, config: CASAConfig, db: StateDB) -> None:
        self.config = config
        self.db = db
        self.jobs_dir = Path(config.jobs_dir)

    async def submit(self, job_id: int, script_path: str) -> None:
        """Spawn CASA and stream output to the database.

        Awaits CASA exit — the event loop stays idle during execution.
        """
        cmd = [self.config.executable, *self.config.args, script_path]
        log.info("Submitting job %d: %s", job_id, " ".join(cmd))

        self.db.update_job(job_id, status="running")

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def drain_stream(stream: asyncio.StreamReader, buf: list[str]) -> None:
                async for raw_line in stream:
                    line = raw_line.decode(errors="replace").rstrip()
                    buf.append(line)
                    log.debug("[job %d] %s", job_id, line)

            await asyncio.gather(
                drain_stream(proc.stdout, stdout_lines),  # type: ignore[arg-type]
                drain_stream(proc.stderr, stderr_lines),  # type: ignore[arg-type]
            )
            await proc.wait()
            returncode = proc.returncode

        except Exception as exc:
            log.exception("Failed to run CASA for job %d", job_id)
            stderr_lines.append(f"Runner error: {exc}")
            returncode = -1

        outcome = "done" if returncode == 0 else "failed"

        stdout_text = "\n".join(stdout_lines)
        self.db.update_job(
            job_id,
            status=outcome,
            stdout=stdout_text,
            stderr="\n".join(stderr_lines),
        )

        # Extract and store WILDCAT_METRICS for fast retrieval
        if outcome == "done":
            for line in stdout_lines:
                if line.startswith("WILDCAT_METRICS:"):
                    try:
                        metrics = json.loads(line[len("WILDCAT_METRICS:"):].strip())
                        self.db.update_job_metrics(job_id, json.dumps(metrics))
                    except json.JSONDecodeError:
                        log.warning("Job %d: malformed WILDCAT_METRICS line", job_id)
                    break

        log.info("Job %d finished with outcome=%s", job_id, outcome)
