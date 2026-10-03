"""Claude drives the lab: ``claude -p`` headless, with the colour-lab MCP server as its only tools.

The CLI gets an MCP config naming one server, ``colour-loop-mcp`` (stdio), and is allowed exactly
that server's six tools: no built-in tools, no other MCP servers, no user or project settings (so no
hooks), no skills, run from an empty temporary directory, with a total time budget after which the
whole process group is stopped. It uses whatever login the CLI already has (on a Max plan: no API
key, no per-call spend). This module never reads or sets ``ANTHROPIC_API_KEY`` and drops that one
variable from the child environment, as ``claude_layer.py`` does.

Where the tool calls come from: the MCP server writes every call, with its arguments and its result
or refusal, to the run record as it handles it. That is the source of truth for the live view's
tool-call feed, because it sees exactly what the lab did. Claude's ``stream-json`` output adds what
only Claude knows: the sentences it writes between tool calls and the session summary at the end,
which are appended to the same record.

``RecordFollower`` turns a run record into the live view (and replays each dispense on a second
simulated deck for the deck view), so ``colour-loop watch`` can show a session driven from Claude
Code or Claude Desktop the same way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .claude_layer import ClaudeAdvisor
from .loop import LiveState, RunConfig
from .mcp_server import SERVER_NAME, append_line
from .robot import WELL_TOTAL_UL, ColourRobot

log = logging.getLogger("colour_loop")

TOOLS = ("get_deck_state", "list_dyes", "dispense_mix", "read_plate", "suggest_next_mixes", "get_run_record")
ALLOWED_TOOLS = [f"mcp__{SERVER_NAME}__{t}" for t in TOOLS]

SYSTEM_PROMPT = (
    "You are the scientist running a small self-driving colour lab. Everything you do in the lab goes "
    "through the colour-lab tools: a simulated OT-2 liquid handler mixes red, yellow and blue dye into "
    "a 96-well plate, and a simulated plate reader measures each well's colour and its CIEDE2000 "
    "distance (delta_e) to the target. Work in rounds: dispense a few mixes into empty wells, then read "
    "them all with one read_plate call (each read of new wells is one measurement round), and use the "
    "readings to choose the next mixes. suggest_next_mixes asks a Bayesian optimiser for ideas; use it "
    "when it helps, but you decide. Aim to reach the threshold with as few wells and rounds as you can. "
    "Stop as soon as read_plate says converged is true, or when the budget is used up. The lab enforces "
    "its own limits; if a call is refused, read the reason and adjust. Before each round, write one "
    "short plain sentence on what you are trying and why. End with a two-sentence summary."
)


def user_prompt(target: str) -> str:
    return (f"The target colour for this run is {target}. Start by checking the deck and the dyes, then "
            "find a mix that matches the target.")


def server_command() -> List[str]:
    """How to start ``colour-loop-mcp``: the console script next to this Python, else ``-m``."""
    script = Path(sys.executable).parent / "colour-loop-mcp"
    if script.exists():
        return [str(script)]
    return [sys.executable, "-m", "colour_loop.mcp_server"]


@dataclass
class AgentConfig:
    target: str = "#7a4b9c"
    threshold: float = 2.0
    seed: int = 0
    max_wells: int = 32
    max_rounds: int = 8
    model: str = "sonnet"
    effort: str = "low"
    time_budget: float = 600.0  # seconds for the whole Claude session
    record_path: Path = field(default_factory=lambda: Path("runs") / f"agent-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    executable: str = "claude"

    def server_args(self) -> List[str]:
        return ["--target", self.target, "--threshold", str(self.threshold), "--seed", str(self.seed),
                "--max-wells", str(self.max_wells), "--max-rounds", str(self.max_rounds),
                "--record", str(self.record_path.resolve())]

    def mcp_config(self) -> dict:
        command = server_command()
        return {"mcpServers": {SERVER_NAME: {"type": "stdio", "command": command[0],
                                             "args": command[1:] + self.server_args()}}}

    def claude_args(self) -> List[str]:
        return [
            self.executable, "-p",
            "--model", self.model,
            "--effort", self.effort,
            "--output-format", "stream-json", "--verbose",
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",  # no built-in tools at all
            "--mcp-config", json.dumps(self.mcp_config()),
            "--strict-mcp-config",  # ... and no MCP server but this one
            "--allowedTools", ",".join(ALLOWED_TOOLS),
            "--permission-mode", "dontAsk",  # anything not allowed above is denied, never prompted
            "--setting-sources", "",  # no user/project/local settings, so no hooks
            "--disable-slash-commands",
            "--no-session-persistence",
            user_prompt(self.target),
        ]


@dataclass
class AgentResult:
    ok: bool
    error: Optional[str]
    seconds: float
    record_path: Path
    summary: dict


def child_env() -> dict:
    # Drop the API key variable by name (its value is never looked at) so the CLI uses its login.
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


def summarise_record(path: Path) -> dict:
    """Rounds, wells, best distance, refusals and the best-so-far curve of an MCP run record."""
    rounds, calls, claude = [], [], {}
    session = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "session":
                session = event
            elif kind == "round":
                rounds.append(event)
            elif kind == "call":
                calls.append(event)
            elif kind == "claude" and event.get("event") == "result":
                claude = event
    wells = sum(1 for c in calls if c["tool"] == "dispense_mix" and c["ok"])
    refusals = [c for c in calls if not c["ok"] and c.get("refused", True)]
    converged_at = next((r["round"] for r in rounds if r["converged"]), None)
    wells_at_convergence = None
    if converged_at is not None:
        wells_at_convergence = sum(len(r["wells"]) for r in rounds if r["round"] <= converged_at)
    return {
        "target": session.get("target"),
        "seed": session.get("seed"),
        "rounds": len(rounds),
        "wells": wells,
        "converged": converged_at is not None,
        "rounds_to_converge": converged_at,
        "wells_to_converge": wells_at_convergence,
        "best_delta_e": rounds[-1]["best_so_far"]["delta_e"] if rounds else None,
        "curve": [r["best_so_far"]["delta_e"] for r in rounds],
        "tool_calls": len(calls),
        "tool_calls_by_name": {t: sum(1 for c in calls if c["tool"] == t) for t in TOOLS},
        "suggestions_used": sum(1 for r in rounds for w in r["wells"] if w.get("decided_by") == "suggestion"),
        "refusals": len(refusals),
        "refusal_reasons": [f"{c['tool']}: {c.get('reason', '')}" for c in refusals],
        "claude": {k: claude.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms")} if claude else None,
    }


class RecordFollower:
    """Follows a run record as it grows and mirrors it into ``LiveState``.

    Each successful ``dispense_mix`` is replayed on ``robot`` (a second simulated OT-2 that only
    the deck view looks at) so the deck shows the same tips and wells the lab server used.
    """

    def __init__(self, path: Path, live: LiveState, robot: Optional[ColourRobot] = None, poll: float = 0.25):
        self.path = path
        self.live = live
        self.robot = robot
        self.poll = poll
        self._offset = 0
        self._buffer = b""
        self._wells: dict = {}
        self._round = 0
        self._replay: asyncio.Queue = asyncio.Queue()
        self.done = False

    async def run(self) -> None:
        replayer = asyncio.create_task(self._replay_loop()) if self.robot is not None else None
        try:
            while not self.done:
                await self.drain()
                await asyncio.sleep(self.poll)
            await self.drain()
            if replayer is not None:
                await self._replay.join()
        finally:
            if replayer is not None:
                replayer.cancel()

    async def drain(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        self._offset += len(chunk)
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            if line.strip():
                try:
                    self.apply(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    log.debug("skipping record line: %s", exc)

    def apply(self, event: dict) -> None:
        live, kind = self.live, event.get("type")
        if kind == "session":
            live.update(target=event["target"], threshold=event["threshold"],
                        max_rounds=event["limits"]["max_rounds"], status="Claude starting")
        elif kind == "call":
            item = {k: event.get(k) for k in ("seq", "time", "tool", "args", "ok", "refused", "reason")}
            item["brief"] = _brief(event)
            live.append("feed", item)
            if event["ok"]:
                live.update(status=f"Claude called {event['tool']}", activity=item["brief"])
            else:
                live.update(status=f"{event['tool']} refused", activity=event.get("reason", ""))
            if event["ok"] and event["tool"] == "dispense_mix":
                r = event["result"]
                self._wells[r["well"]] = {"round": None, "well": r["well"], "decided_by": "claude",
                                          "fractions": [r["fractions"][d] for d in ("red", "yellow", "blue")],
                                          "dispensed_ul": r["dispensed_ul"], "hex": None, "delta_e": None}
                live.update(wells=list(self._wells.values()))
                if self.robot is not None:
                    self._replay.put_nowait(r)
        elif kind == "round":
            self._round = event["round"]
            for w in event["wells"]:
                self._wells[w["well"]] = {"round": w["round"], "well": w["well"], "decided_by": w["decided_by"],
                                          "fractions": [w["fractions"][d] for d in ("red", "yellow", "blue")],
                                          "dispensed_ul": w["dispensed_ul"], "rgb": w["rgb"], "hex": w["hex"],
                                          "delta_e": w["delta_e"]}
            best = event["best_so_far"]
            best = dict(best, fractions=[best["fractions"][d] for d in ("red", "yellow", "blue")])
            live.update(wells=list(self._wells.values()), round=event["round"], best=best)
            live.append("rounds", {"round": event["round"], "round_best": event["round_best_delta_e"],
                                   "best": best["delta_e"]})
        elif kind == "claude":
            if event.get("event") == "text" and event.get("text", "").strip():
                live.append("notes", {"round": self._round, "note": event["text"].strip(), "override": False, "why": ""})
            elif event.get("event") == "result":
                live.update(status="Claude finished", activity=f"Claude finished: {event.get('subtype')}, "
                                                                 f"{event.get('num_turns')} turns")

    async def _replay_loop(self) -> None:
        while True:
            r = await self._replay.get()
            try:
                total = sum(r["dispensed_ul"].values())
                mix = [r["dispensed_ul"][d] / total for d in ("red", "yellow", "blue")]
                await self.robot.mix_wells([mix], wells=[r["well"]], total_ul=total or WELL_TOTAL_UL)
            except Exception as exc:  # the replay is cosmetic; never let it stop the view
                log.debug("deck replay failed: %s", exc)
            finally:
                self._replay.task_done()


def _brief(event: dict) -> str:
    """One line for the feed: what a call did."""
    if not event["ok"]:
        return event.get("reason", "")
    r, tool = event.get("result") or {}, event["tool"]
    if tool == "dispense_mix":
        vols = ", ".join(f"{d} {v:g}" for d, v in r["dispensed_ul"].items() if v)
        return f"{r['well']}: {vols} uL"
    if tool == "read_plate":
        new = [x for x in r.get("readings", []) if x.get("new")]
        parts = [f"{x['well']} {x['hex']} dE {x['delta_e']:.2f}" for x in (new or r.get("readings", []))]
        tail = " - converged" if r.get("converged") else ""
        return ("round %s: " % r["round"] if r.get("round") else "re-read: ") + ", ".join(parts) + tail
    if tool == "suggest_next_mixes":
        return "; ".join("/".join(f"{v:.2f}" for v in s.values()) for s in r.get("suggestions", [])) + " (R/Y/B)"
    if tool in ("get_deck_state", "get_run_record"):
        b = r.get("budget") or {}
        return f"{b.get('wells_used')}/{b.get('max_wells')} wells, {b.get('rounds_done')}/{b.get('max_rounds')} rounds used"
    if tool == "list_dyes":
        return ", ".join(r.get("dyes", []))
    return ""


async def _read_stream(proc: asyncio.subprocess.Process, record: Path) -> dict:
    """Copy Claude's narration and session summary from stream-json into the record."""
    result: dict = {}
    assert proc.stdout is not None
    async for raw in proc.stdout:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = msg.get("type")
        if kind == "system" and msg.get("subtype") == "init":
            append_line(record, {"type": "claude", "event": "init", "model": msg.get("model"),
                                 "tools": msg.get("tools"), "mcp_servers": msg.get("mcp_servers"),
                                 "api_key_source": msg.get("apiKeySource"), "permission_mode": msg.get("permissionMode")})
        elif kind == "assistant":
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    append_line(record, {"type": "claude", "event": "text", "time": round(time.time(), 3),
                                         "text": block["text"]})
        elif kind == "result":
            result = {k: msg.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "result")}
            append_line(record, {"type": "claude", "event": "result", **result})
    return result


async def run_agent(config: AgentConfig, live: Optional[LiveState] = None,
                    deck_robot: Optional[ColourRobot] = None) -> AgentResult:
    """One Claude-driven run. With ``live``, the run record is mirrored into the live view as it grows."""
    started = time.monotonic()
    record = config.record_path
    if record.exists():
        raise ValueError(f"run record {record} already exists; pick a new --record path")
    record.parent.mkdir(parents=True, exist_ok=True)

    advisor = ClaudeAdvisor(executable=config.executable)
    if not advisor.check():
        return AgentResult(False, f"Claude is unavailable: {advisor.unavailable_reason}", 0.0, record, {})
    executable = shutil.which(config.executable) or config.executable

    follower = None
    follow_task = None
    if live is not None:
        follower = RecordFollower(record, live, deck_robot)
        follow_task = asyncio.create_task(follower.run())

    error = None
    with tempfile.TemporaryDirectory(prefix="colour-loop-agent-") as workdir, \
            tempfile.TemporaryFile() as stderr:
        args = config.claude_args()
        args[0] = executable
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=workdir, env=child_env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=stderr,
            start_new_session=True,  # its own process group, so the time budget can stop the MCP server too
            limit=16 * 1024 * 1024)
        try:
            result = await asyncio.wait_for(_read_stream(proc, record), timeout=config.time_budget)
            await asyncio.wait_for(proc.wait(), timeout=30)
            if not result:
                stderr.seek(0)
                error = f"claude exited {proc.returncode} without a result: {stderr.read().decode(errors='replace').strip()[:300]}"
            elif result.get("is_error"):
                error = f"claude reported an error: {result.get('subtype')}: {str(result.get('result'))[:300]}"
        except asyncio.TimeoutError:
            error = f"time budget of {config.time_budget:.0f}s used up; Claude was stopped"
            append_line(record, {"type": "claude", "event": "timeout", "seconds": config.time_budget})
        finally:
            if proc.returncode is None:
                _stop_group(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    _stop_group(proc, signal.SIGKILL)
                    await proc.wait()

    if follower is not None:
        follower.done = True
        await follow_task
    summary = summarise_record(record)
    seconds = time.monotonic() - started
    summary["seconds"] = round(seconds, 1)
    if live is not None:
        verdict = "converged" if summary["converged"] else ("stopped: " + error if error else "did not converge")
        live.update(status=f"{verdict} after {summary['rounds']} round{'s' if summary['rounds'] != 1 else ''}, "
                           f"{summary['wells']} wells, {summary['refusals']} refused calls", finished=True)
    return AgentResult(error is None, error, seconds, record, summary)


def _stop_group(proc: asyncio.subprocess.Process, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def agent_live_state(config: AgentConfig) -> LiveState:
    live = LiveState(RunConfig(target=config.target, threshold=config.threshold, max_rounds=config.max_rounds,
                               use_claude=True, claude_model=config.model), mode="agent")
    live.update(status="starting Claude")
    return live
