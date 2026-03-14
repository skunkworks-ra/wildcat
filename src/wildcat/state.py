"""SQLite state machine for wildcat workflow tracking.

The StateDB wraps a sqlite3.Connection and provides all read/write
operations needed by the orchestrator. Each public method is a single
logical unit — callers never write raw SQL.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from enum import Enum
from typing import Any


class Stage(str, Enum):
    IDLE             = "IDLE"
    PHASE1_RUNNING   = "PHASE1_RUNNING"
    PHASE1_COMPLETE  = "PHASE1_COMPLETE"
    PHASE2_RUNNING   = "PHASE2_RUNNING"
    PHASE2_COMPLETE  = "PHASE2_COMPLETE"
    PHASE3_RUNNING   = "PHASE3_RUNNING"
    PHASE3_COMPLETE  = "PHASE3_COMPLETE"
    HUMAN_CHECKPOINT = "HUMAN_CHECKPOINT"
    CALIBRATION_LOOP = "CALIBRATION_LOOP"
    IMAGING_PIPELINE = "IMAGING_PIPELINE"
    ERROR            = "ERROR"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ms_path     TEXT NOT NULL,
    stage       TEXT NOT NULL DEFAULT 'IDLE',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tool_outputs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id  INTEGER NOT NULL REFERENCES workflow(id),
    phase        TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    output_json  TEXT NOT NULL,
    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id  INTEGER NOT NULL REFERENCES workflow(id),
    stage        TEXT NOT NULL,
    script_path  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    stdout       TEXT,
    stderr       TEXT,
    plots        TEXT,
    queued_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at   DATETIME,
    completed_at DATETIME
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id  INTEGER NOT NULL REFERENCES workflow(id),
    stage        TEXT NOT NULL,
    llm_summary  TEXT NOT NULL,
    human_route  TEXT,
    human_notes  TEXT,
    presented_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    decided_at   DATETIME
);

CREATE TABLE IF NOT EXISTS llm_decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id  INTEGER NOT NULL REFERENCES workflow(id),
    stage        TEXT NOT NULL,
    prompt_hash  TEXT,
    decision     TEXT NOT NULL,
    model        TEXT NOT NULL,
    decided_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


class StateDB:
    """Context manager wrapping a sqlite3.Connection.

    Usage:
        db = StateDB("/data/wildcat.db")
        db.init_schema()
        wf_id = db.create_workflow("/data/test.ms")
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> StateDB:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StateDB must be used as a context manager")
        return self._conn

    def init_schema(self) -> None:
        """Create all tables if they don't already exist."""
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── Workflow ──────────────────────────────────────────────────────────

    def create_workflow(self, ms_path: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO workflow (ms_path, stage) VALUES (?, ?)",
            (ms_path, Stage.IDLE),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_workflow(self, workflow_id: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM workflow WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No workflow with id={workflow_id}")
        return dict(row)

    def transition(self, workflow_id: int, new_stage: Stage) -> None:
        self.conn.execute(
            "UPDATE workflow SET stage = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_stage, workflow_id),
        )
        self.conn.commit()

    # ── Tool outputs ──────────────────────────────────────────────────────

    def save_tool_output(
        self, workflow_id: int, phase: str, tool_name: str, output_json: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO tool_outputs (workflow_id, phase, tool_name, output_json)"
            " VALUES (?, ?, ?, ?)",
            (workflow_id, phase, tool_name, output_json),
        )
        self.conn.commit()

    def get_tool_outputs(self, workflow_id: int, phase: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tool_outputs WHERE workflow_id = ? AND phase = ?"
            " ORDER BY collected_at",
            (workflow_id, phase),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Jobs ──────────────────────────────────────────────────────────────

    def create_job(self, workflow_id: int, stage: str, script_path: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO jobs (workflow_id, stage, script_path) VALUES (?, ?, ?)",
            (workflow_id, stage, script_path),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_job(
        self,
        job_id: int,
        *,
        status: str,
        stdout: str | None = None,
        stderr: str | None = None,
        plots: str | None = None,
    ) -> None:
        ts_field = ""
        if status == "running":
            ts_field = ", started_at = CURRENT_TIMESTAMP"
        elif status in ("done", "failed"):
            ts_field = ", completed_at = CURRENT_TIMESTAMP"

        self.conn.execute(
            f"UPDATE jobs SET status = ?, stdout = ?, stderr = ?, plots = ?{ts_field}"
            " WHERE id = ?",
            (status, stdout, stderr, plots, job_id),
        )
        self.conn.commit()

    # ── Checkpoints ───────────────────────────────────────────────────────

    def create_checkpoint(
        self, workflow_id: int, stage: str, llm_summary: str
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO checkpoints (workflow_id, stage, llm_summary) VALUES (?, ?, ?)",
            (workflow_id, stage, llm_summary),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_latest_checkpoint(self, workflow_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM checkpoints WHERE workflow_id = ?"
            " ORDER BY presented_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        return dict(row) if row else None

    def resolve_checkpoint(
        self, checkpoint_id: int, human_route: str, human_notes: str
    ) -> None:
        self.conn.execute(
            "UPDATE checkpoints SET human_route = ?, human_notes = ?,"
            " decided_at = CURRENT_TIMESTAMP WHERE id = ?",
            (human_route, human_notes, checkpoint_id),
        )
        self.conn.commit()

    # ── LLM decisions ─────────────────────────────────────────────────────

    def save_llm_decision(
        self,
        workflow_id: int,
        stage: str,
        decision: str,
        model: str,
        prompt_hash: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO llm_decisions"
            " (workflow_id, stage, prompt_hash, decision, model)"
            " VALUES (?, ?, ?, ?, ?)",
            (workflow_id, stage, prompt_hash, decision, model),
        )
        self.conn.commit()
