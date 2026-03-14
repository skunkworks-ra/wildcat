"""Typed configuration loader for wildcat.

Reads config.toml using stdlib tomllib (Python 3.11+) and exposes
typed dataclasses. All other modules import from here — no raw TOML
reading elsewhere.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LlamaCppConfig:
    base_url: str
    model_path: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    threads: int
    temp: float
    extra_args: list[str]


@dataclass
class OllamaConfig:
    base_url: str
    model: str


@dataclass
class LLMConfig:
    backend: str  # "llamacpp" | "ollama"
    llamacpp: LlamaCppConfig
    ollama: OllamaConfig

    @property
    def active_base_url(self) -> str:
        if self.backend == "llamacpp":
            return self.llamacpp.base_url
        return self.ollama.base_url

    @property
    def active_model(self) -> str:
        if self.backend == "llamacpp":
            # llama-server uses the loaded model; any string works for the API
            return "local"
        return self.ollama.model


@dataclass
class MCPConfig:
    base_url: str


@dataclass
class UIConfig:
    port: int


@dataclass
class SkillsConfig:
    path: str


@dataclass
class StateConfig:
    db_path: str


@dataclass
class CASAConfig:
    executable: str
    args: list[str]
    jobs_dir: str


@dataclass
class WildcatConfig:
    llm: LLMConfig
    mcp: MCPConfig
    ui: UIConfig
    skills: SkillsConfig
    state: StateConfig
    casa: CASAConfig


def load_config(path: str | Path = "config.toml") -> WildcatConfig:
    """Load and validate config.toml, returning a typed WildcatConfig."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    llm_raw = raw["llm"]
    llamacpp_raw = llm_raw["llamacpp"]
    ollama_raw = llm_raw["ollama"]

    return WildcatConfig(
        llm=LLMConfig(
            backend=llm_raw["backend"],
            llamacpp=LlamaCppConfig(
                base_url=llamacpp_raw["base_url"],
                model_path=llamacpp_raw["model_path"],
                port=llamacpp_raw["port"],
                ctx_size=llamacpp_raw["ctx_size"],
                n_gpu_layers=llamacpp_raw["n_gpu_layers"],
                threads=llamacpp_raw["threads"],
                temp=llamacpp_raw["temp"],
                extra_args=llamacpp_raw.get("extra_args", []),
            ),
            ollama=OllamaConfig(
                base_url=ollama_raw["base_url"],
                model=ollama_raw["model"],
            ),
        ),
        mcp=MCPConfig(base_url=raw["mcp"]["base_url"]),
        ui=UIConfig(port=raw["ui"]["port"]),
        skills=SkillsConfig(path=raw["skills"]["path"]),
        state=StateConfig(db_path=raw["state"]["db_path"]),
        casa=CASAConfig(
            executable=raw["casa"]["executable"],
            args=raw["casa"]["args"],
            jobs_dir=raw["casa"]["jobs_dir"],
        ),
    )
