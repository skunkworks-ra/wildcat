"""CASA subprocess runner with sentinel-file-based completion signalling.

Architecture:
  CASARunner.submit()  — spawns CASA as an asyncio subprocess, streams
                          stdout/stderr to SQLite in real time, writes a
                          sentinel file on completion.
  SentinelWatcher      — watchdog observer on jobs_dir; fires an
                          asyncio.Event when any sentinel appears.

The orchestrator awaits the asyncio.Event rather than polling, so the
event loop stays idle while CASA is running (which may take hours).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from wildcat.config import CASAConfig
from wildcat.state import StateDB

log = logging.getLogger(__name__)


class _SentinelHandler(FileSystemEventHandler):
    """Watchdog event handler that sets an asyncio.Event when a sentinel appears."""

    def __init__(self, loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
        self._loop = loop
        self._event = event

    def on_created(self, event: Any) -> None:
        if isinstance(event, FileCreatedEvent):
            path = Path(event.src_path)
            if path.suffix in (".done", ".failed"):
                log.info("Sentinel detected: %s", path)
                self._loop.call_soon_threadsafe(self._event.set)


class SentinelWatcher:
    """watchdog observer on jobs_dir. Fires asyncio.Event when sentinel appears.

    The event is shared with the orchestrator — when the watcher fires it,
    orchestrator's `await event.wait()` unblocks immediately.
    """

    def __init__(self, jobs_dir: Path, event: asyncio.Event) -> None:
        self.jobs_dir = jobs_dir
        self.event = event
        self._observer: Observer | None = None
        self._loop = asyncio.get_event_loop()

    def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        handler = _SentinelHandler(self._loop, self.event)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.jobs_dir), recursive=False)
        self._observer.start()
        log.info("SentinelWatcher started on %s", self.jobs_dir)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None


class CASARunner:
    """Runs CASA scripts as asyncio subprocesses.

    Each job gets a unique directory under jobs_dir. stdout/stderr are
    streamed to SQLite line by line. A sentinel file is written on exit:
      <jobs_dir>/<job_id>.done    — CASA exited 0
      <jobs_dir>/<job_id>.failed  — CASA exited non-zero
    """

    def __init__(self, config: CASAConfig, db: StateDB) -> None:
        self.config = config
        self.db = db
        self.jobs_dir = Path(config.jobs_dir)

    def _sentinel_path(self, job_id: int, outcome: str) -> Path:
        """Return path for a sentinel file. outcome: 'done' | 'failed'"""
        return self.jobs_dir / f"{job_id}.{outcome}"

    async def submit(self, job_id: int, script_path: str) -> None:
        """Spawn CASA and stream output to the database.

        This coroutine runs until CASA exits — the caller should run it
        as an asyncio.Task so the orchestrator can await the sentinel event.
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
        status  = "done" if returncode == 0 else "failed"

        self.db.update_job(
            job_id,
            status=status,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )

        # Write sentinel — this triggers the SentinelWatcher
        sentinel = self._sentinel_path(job_id, outcome)
        sentinel.touch()
        log.info("Job %d finished with outcome=%s", job_id, outcome)
