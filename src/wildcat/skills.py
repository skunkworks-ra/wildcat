"""Skill document loader and system prompt assembler.

Radio interferometry skill partials live in the ms-inspect repo under
  .claude/skills/radio-interferometry/

They are concatenated in a stage-dependent order so the small LLM
receives only the knowledge relevant to the current decision point —
keeping the prompt within context budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

from wildcat.state import Stage

log = logging.getLogger(__name__)

# Partials always included regardless of stage
_ALWAYS = [
    "SKILL.md",
    "01-workflow.md",
    "01b-workflow-phase2.md",
]

# Additional partials by stage group
_STAGE_EXTRAS: dict[str, list[str]] = {
    "phase1":      ["02-orientation.md"],
    "phase2":      ["03-instrument-sanity.md"],
    "phase3":      ["04-diagnostic-reasoning.md", "05-calibrator-science.md"],
    "checkpoint":  ["04-diagnostic-reasoning.md", "06-failure-modes.md"],
}

# Map Stage enum values to stage groups
_STAGE_TO_GROUP: dict[Stage, str] = {
    Stage.PHASE1_RUNNING:   "phase1",
    Stage.PHASE1_COMPLETE:  "phase1",
    Stage.PHASE2_RUNNING:   "phase2",
    Stage.PHASE2_COMPLETE:  "phase2",
    Stage.PHASE3_RUNNING:   "phase3",
    Stage.PHASE3_COMPLETE:  "phase3",
    Stage.HUMAN_CHECKPOINT: "checkpoint",
    Stage.CALIBRATION_LOOP: "checkpoint",
    Stage.IMAGING_PIPELINE: "checkpoint",
}


def load_system_prompt(skills_path: str, stage: Stage) -> str:
    """Concatenate skill partials relevant to the current stage.

    Returns the assembled system prompt string. Missing files are logged
    as warnings (not raised) so a misconfigured mount doesn't crash the
    orchestrator — the LLM can still operate with reduced context.
    """
    base = Path(skills_path)
    group = _STAGE_TO_GROUP.get(stage, "phase1")
    partials = _ALWAYS + _STAGE_EXTRAS.get(group, [])

    sections: list[str] = []
    for filename in partials:
        path = base / filename
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
            log.debug("Loaded skill partial: %s", filename)
        else:
            log.warning("Skill partial not found (skipped): %s", path)

    if not sections:
        log.warning(
            "No skill partials loaded from %s — LLM will operate without domain context",
            skills_path,
        )
        return ""

    return "\n\n---\n\n".join(sections)
