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
            max_tokens=self.config.llamacpp.max_tokens if self.config.backend == "llamacpp" else 4096,
        )
        # Return as dict for uniform handling downstream
        return response.model_dump()

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_executor: Any,  # callable(name, args) -> str
        *,
        max_rounds: int = 5,
        max_result_tokens: int = 16000,
    ) -> dict[str, Any]:
        """Multi-turn tool-use loop.

        Loops until the LLM produces a final text response or budget is exhausted.

        Returns the final response dict (the one with text content, not tool calls).
        """
        import json as _json

        if self._client is None:
            raise RuntimeError("LLMBackend not started")

        conversation = list(messages)
        total_result_tokens = 0

        def _llm_kwargs() -> dict:
            return {
                "model": self.config.active_model,
                "temperature": self.config.llamacpp.temp if self.config.backend == "llamacpp" else 0.0,
                "max_tokens": self.config.llamacpp.max_tokens if self.config.backend == "llamacpp" else 4096,
            }

        def _execute_calls(calls: list) -> bool:
            """Execute OpenAI-format tool calls, append results. Returns True if budget hit."""
            nonlocal total_result_tokens
            for tc in calls:
                args = _json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                result = tool_executor(tc.function.name, args)
                tokens = len(result) // 4
                total_result_tokens += tokens
                log.info("Tool call: %s(%s) → %d tokens (total: %d/%d)", tc.function.name, args, tokens, total_result_tokens, max_result_tokens)
                conversation.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                if total_result_tokens >= max_result_tokens:
                    log.warning("Tool result token budget exhausted (%d)", total_result_tokens)
                    return True
            return False

        for _round in range(max_rounds):
            response = await self._client.chat.completions.create(
                messages=conversation, tools=tools, **_llm_kwargs()
            )
            choice = response.choices[0]

            if choice.message.tool_calls:
                conversation.append(choice.message.model_dump())
                if _execute_calls(choice.message.tool_calls):
                    break
                continue

            # No tool calls — this is the final response
            return response.model_dump()

        # Budget exhausted — explicitly tell the model to stop calling tools and answer
        log.warning("Tool rounds exhausted — sending explicit stop instruction")
        conversation.append({
            "role": "user",
            "content": "Stop calling tools. You have enough context. Return ONLY the JSON decision object now, no tool calls.",
        })
        response = await self._client.chat.completions.create(
            messages=conversation, **_llm_kwargs()
        )
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
