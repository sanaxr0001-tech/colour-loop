"""Claude, called headless, explains each round and may swap one proposed mix.

The call is ``claude -p`` with structured JSON output, which uses whatever login the Claude Code
CLI has (on a Max plan: no API key, no per-call spend). This module never reads or sets
``ANTHROPIC_API_KEY``; it removes that one variable from the child process environment so the CLI
cannot silently fall back to paid API billing. If ``claude`` is missing or not logged in, the
advisor reports itself unavailable and the loop runs on the optimiser alone.

Whatever Claude returns is untrusted: an override is accepted only if it names a real proposal and
its fractions are each within 0-1 and sum to 1 (within 0.01). Anything else is ignored and logged.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from .physics import DYES

log = logging.getLogger(__name__)

SUM_TOLERANCE = 0.01

SYSTEM_PROMPT = (
    "You are the scientist in a small self-driving colour lab. A simulated liquid-handling robot "
    "mixes red, yellow and blue dye (volume fractions that sum to 1) in a 96-well plate, and a "
    "simulated plate reader measures each well's colour. A Bayesian optimiser proposes the next "
    "mixes. Your job each round: write a short note (at most two sentences, plain language) on "
    "what the results so far show and what the proposals are trying, and optionally replace at "
    "most ONE proposal with a mix you think is clearly better. Only replace when you have a "
    "concrete reason from the data. Answer with JSON only."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "description": "At most two sentences."},
        "replace": {
            "type": ["object", "null"],
            "description": "Optional. Swap one proposal for this mix.",
            "properties": {
                "index": {"type": "integer", "description": "0-based index of the proposal to replace"},
                "red": {"type": "number"},
                "yellow": {"type": "number"},
                "blue": {"type": "number"},
                "why": {"type": "string"},
            },
            "required": ["index", "red", "yellow", "blue"],
        },
    },
    "required": ["note"],
}


@dataclass
class Override:
    index: int
    fractions: List[float]
    why: str = ""


@dataclass
class Advice:
    note: str = ""
    override: Optional[Override] = None
    rejected: Optional[str] = None  # why a proposed override was ignored
    error: Optional[str] = None  # why the call itself failed
    seconds: float = 0.0
    raw: Any = field(default=None, repr=False)


def validate_override(replace: Any, n_proposals: int) -> Override:
    """Check Claude's proposed replacement. Raises ``ValueError`` with the reason if it is unsafe."""
    if not isinstance(replace, dict):
        raise ValueError("replace must be an object")
    index = replace.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"index must be an integer, got {index!r}")
    if not 0 <= index < n_proposals:
        raise ValueError(f"index {index} is not one of the {n_proposals} proposals")
    fractions = []
    for dye in DYES:
        value = replace.get(dye)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{dye} must be a number, got {value!r}")
        value = float(value)
        if value != value or not 0.0 <= value <= 1.0:  # NaN or out of range
            raise ValueError(f"{dye} fraction {value} is outside 0-1")
        fractions.append(value)
    total = sum(fractions)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"fractions sum to {total:.3f}, not 1")
    fractions = [f / total for f in fractions]
    return Override(index=index, fractions=fractions, why=str(replace.get("why", ""))[:300])


def parse_response(payload: Any, n_proposals: int) -> Advice:
    """Turn Claude's structured output into ``Advice``, dropping anything that fails validation."""
    if not isinstance(payload, dict):
        return Advice(error="response was not a JSON object", raw=payload)
    note = payload.get("note")
    advice = Advice(note=str(note)[:500] if isinstance(note, str) else "", raw=payload)
    replace = payload.get("replace")
    if replace is not None:
        try:
            advice.override = validate_override(replace, n_proposals)
        except ValueError as exc:
            advice.rejected = str(exc)
            log.warning("ignoring Claude override: %s", exc)
    return advice


def format_prompt(
    target_hex: str,
    history: Sequence[dict],
    proposals: Sequence[Sequence[float]],
    best: Optional[dict],
) -> str:
    lines = [f"Target colour: {target_hex}", ""]
    if history:
        lines.append("Results so far (fractions red/yellow/blue -> measured colour, CIEDE2000 distance):")
        for h in history:
            mix = "/".join(f"{f:.2f}" for f in h["fractions"])
            lines.append(f"  round {h['round']} {h['well']}: {mix} -> {h['hex']} dE={h['delta_e']:.2f}")
    else:
        lines.append("No results yet: this is the first round.")
    if best:
        mix = "/".join(f"{f:.2f}" for f in best["fractions"])
        lines.append(f"Best so far: {best['well']} {mix} -> {best['hex']} dE={best['delta_e']:.2f}")
    lines.append("")
    lines.append("Optimiser proposals for this round (index: red/yellow/blue):")
    for i, p in enumerate(proposals):
        lines.append(f"  {i}: " + "/".join(f"{f:.2f}" for f in p))
    lines.append("")
    lines.append(
        "Reply as JSON: {\"note\": \"...\", \"replace\": null} or with "
        "\"replace\": {\"index\": i, \"red\": r, \"yellow\": y, \"blue\": b, \"why\": \"...\"} "
        "where r + y + b = 1 and each is between 0 and 1."
    )
    return "\n".join(lines)


def _child_env() -> dict:
    # Drop the API key variable by name (its value is never looked at) so the CLI uses its login.
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


class ClaudeAdvisor:
    def __init__(self, model: str = "sonnet", timeout: float = 90.0, executable: str = "claude"):
        self.model = model
        self.timeout = timeout
        self.executable = executable
        self.unavailable_reason: Optional[str] = None
        # Run the CLI from an empty directory so no project settings, hooks or CLAUDE.md apply.
        self._workdir = tempfile.mkdtemp(prefix="colour-loop-claude-")

    def check(self) -> bool:
        """True if the CLI exists and is logged in; otherwise records why not."""
        path = shutil.which(self.executable)
        if path is None:
            self.unavailable_reason = f"`{self.executable}` is not on PATH"
            return False
        try:
            out = subprocess.run(
                [path, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=30,
                env=_child_env(),
                cwd=self._workdir,
            )
            status = json.loads(out.stdout or "{}")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            self.unavailable_reason = f"could not read `claude auth status`: {exc}"
            return False
        if not status.get("loggedIn"):
            self.unavailable_reason = "the Claude CLI is not logged in"
            return False
        return True

    def advise(
        self,
        target_hex: str,
        history: Sequence[dict],
        proposals: Sequence[Sequence[float]],
        best: Optional[dict],
    ) -> Advice:
        prompt = format_prompt(target_hex, history, proposals, best)
        cmd = [
            self.executable,
            "-p",
            "--model", self.model,
            "--effort", "low",  # a short note does not need long thinking; keeps a call to seconds
            "--output-format", "json",
            "--json-schema", json.dumps(RESPONSE_SCHEMA),
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            prompt,
        ]
        start = time.monotonic()
        try:
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_child_env(),
                cwd=self._workdir,
            )
        except subprocess.TimeoutExpired:
            return Advice(error=f"claude timed out after {self.timeout:.0f}s", seconds=time.monotonic() - start)
        except OSError as exc:
            return Advice(error=f"could not run claude: {exc}", seconds=time.monotonic() - start)
        seconds = time.monotonic() - start
        try:
            envelope = json.loads(out.stdout)
        except json.JSONDecodeError:
            return Advice(error=f"claude exited {out.returncode} without JSON: {out.stderr.strip()[:200]}", seconds=seconds)
        if envelope.get("is_error"):
            return Advice(error=f"claude reported an error: {str(envelope.get('result'))[:200]}", seconds=seconds)
        payload = envelope.get("structured_output")
        if payload is None:
            try:
                payload = json.loads(envelope.get("result", ""))
            except (TypeError, json.JSONDecodeError):
                return Advice(error="claude returned no structured output", seconds=seconds, raw=envelope.get("result"))
        advice = parse_response(payload, len(proposals))
        advice.seconds = seconds
        return advice
