"""LLM backend manager for wildcat.

Supports two backends that speak identical OpenAI-compatible HTTP:
  - llamacpp: spawns llama-server as a subprocess, waits for /health
  - ollama:   connects to an already-running Ollama instance

The LLMBackend.complete() method is the only call site for LLM
inference — all prompt assembly happens in the orchestrator/skills layer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Any

import httpx
from openai import AsyncOpenAI

from wildcat.config import LLMConfig

log = logging.getLogger(__name__)

_HEALTH_TIMEOUT = 60  # seconds to wait for llama-server /health
_HEALTH_POLL    = 1.0  # poll interval in seconds


class LLMBackend:
    """Manages llama-server subprocess (llamacpp mode) or connects to Ollama."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._client: AsyncOpenAI | None = None

    async def start(self) -> None:
        """Start the LLM backend.

        If backend=llamacpp: build llama-server command from config,
        spawn subprocess, wait until /health returns 200.
        If backend=ollama: just build the client (Ollama must already be running).
        """
        if self.config.backend == "llamacpp":
            cmd = self._build_llamacpp_cmd()
            log.info("Starting llama-server: %s", " ".join(cmd))
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            await self._wait_for_health(self.config.llamacpp.base_url)
            log.info("llama-server is healthy at %s", self.config.llamacpp.base_url)
        elif self.config.backend == "ollama":
            log.info("Using Ollama backend at %s", self.config.ollama.base_url)
        else:
            raise ValueError(f"Unknown LLM backend: {self.config.backend!r}")

        self._client = AsyncOpenAI(
            base_url=self.config.active_base_url,
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        )

    async def stop(self) -> None:
        """Terminate the llama-server subprocess if we own it."""
        if self._proc is not None:
            log.info("Stopping llama-server (pid=%d)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

        if self._client:
            await self._client.close()
            self._client = None

    async def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """POST to /v1/chat/completions. Returns the parsed response dict."""
        if self._client is None:
            raise RuntimeError("LLMBackend not started — call await backend.start() first")

        response = await self._client.chat.completions.create(
            model=self.config.active_model,
            messages=messages,
            temperature=self.config.llamacpp.temp if self.config.backend == "llamacpp" else 0.0,
        )
        # Return as dict for uniform handling downstream
        return response.model_dump()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_llamacpp_cmd(self) -> list[str]:
        """Construct the llama-server invocation from config."""
        cfg = self.config.llamacpp
        cmd = ["llama-server"]
        if cfg.model_path:
            cmd += ["--model", cfg.model_path]
        cmd += [
            "--port",         str(cfg.port),
            "--ctx-size",     str(cfg.ctx_size),
            "--n-gpu-layers", str(cfg.n_gpu_layers),
            "--threads",      str(cfg.threads),
            "--temp",         str(cfg.temp),
        ]
        cmd.extend(cfg.extra_args)
        return cmd

    async def _wait_for_health(self, base_url: str) -> None:
        """Poll base_url/health until 200 or timeout."""
        health_url = base_url.rstrip("/").replace("/v1", "") + "/health"
        deadline = time.monotonic() + _HEALTH_TIMEOUT

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(health_url, timeout=2.0)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(_HEALTH_POLL)

        raise TimeoutError(
            f"llama-server did not become healthy within {_HEALTH_TIMEOUT}s"
            f" ({health_url})"
        )
