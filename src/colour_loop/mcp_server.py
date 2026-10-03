"""MCP tool server: an agent drives the simulated colour lab through six tools.

    colour-loop-mcp --target "#7a4b9c" [--seed 0] [--record runs/agent.jsonl]

One server process is one run. The target colour is fixed by the start-up argument, so the agent
cannot move the goal posts. The transport is stdio, so Claude Code, Claude Desktop or any other MCP
client starts the server itself.

Tools: ``get_deck_state``, ``list_dyes``, ``dispense_mix``, ``read_plate``, ``suggest_next_mixes``
and ``get_run_record``. They drive the same simulated OT-2 (``robot.py``), plate reader
(``physics.py``) and BayBE campaign (``optimiser.py``) as the closed loop.

Guardrails are enforced here, in code, never in a prompt. A refused call returns a tool error whose
text says ``Refused:`` and why, and is appended to the run record like every other call. The run
record is JSONL, one object per line: ``session`` (written at start-up), ``call`` (every tool call,
with its arguments and its result or refusal) and ``round`` (every plate read that measured new
wells). ``colour-loop agent`` and ``colour-loop watch`` follow this file to draw the live view.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from .claude_layer import SUM_TOLERANCE
from .colour import colour_distance, hex_to_rgb, rgb_to_hex
from .physics import DEFAULT_NOISE_SD, DYES, PlateReader
from .robot import PIPETTE_MAX_UL, PIPETTE_MIN_UL, WELL_NAMES, WELL_TOTAL_UL, ColourRobot, fractions_to_volumes

log = logging.getLogger("colour_loop.mcp")

SERVER_NAME = "colour-lab"
MAX_WELLS_BY_TIPS = 96 // len(DYES)  # each dispense takes one fresh tip per dye from one 96-tip rack
MIN_FRACTION = PIPETTE_MIN_UL / WELL_TOTAL_UL  # smallest non-zero dye fraction of a standard well
MAX_SUGGESTIONS = 8

DYE_LOOKS = {
    "red": "a pinkish red: absorbs green strongly and blue partly",
    "yellow": "yellow: absorbs blue strongly and a little green",
    "blue": "blue: absorbs red strongly and a little green",
}

INSTRUCTIONS = (
    "A simulated colour lab: an OT-2 liquid handler mixes red, yellow and blue dye into a 96-well "
    "plate and a plate reader measures each well's colour and its CIEDE2000 distance to the run's "
    "target. Dispense mixes into empty wells, read them, learn, repeat until a well is within the "
    "threshold. suggest_next_mixes asks a Bayesian optimiser (BayBE) for ideas; using them is "
    "optional. The server enforces lab limits and refuses unsafe or impossible requests."
)


class Refusal(ToolError):
    """A guardrail refused the call. The message says which limit and why."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Refused: {reason}")


@dataclass
class LabConfig:
    target: str = "#7a4b9c"
    threshold: float = 2.0  # a well within this CIEDE2000 distance of the target ends the run
    seed: Optional[int] = 0  # plate-reader noise and optimiser seed
    noise_sd: float = DEFAULT_NOISE_SD
    max_wells: int = 32  # per-run limit on dispenses (one tip per dye each: 32 x 3 = one tip rack)
    max_rounds: int = 8  # per-run limit on measurement rounds (plate reads that measure new wells)
    record_path: Path = field(
        default_factory=lambda: Path.home() / "colour-loop-runs" / f"mcp-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")

    def validate(self) -> None:
        hex_to_rgb(self.target)
        if not 1 <= self.max_wells <= MAX_WELLS_BY_TIPS:
            raise ValueError(f"max wells must be 1-{MAX_WELLS_BY_TIPS} (one tip rack)")
        if self.max_rounds < 1:
            raise ValueError("max rounds must be at least 1")

    def limits(self) -> dict:
        return {
            "max_wells": self.max_wells,
            "max_rounds": self.max_rounds,
            "pipette_ul": [PIPETTE_MIN_UL, PIPETTE_MAX_UL],
            "standard_well_ul": WELL_TOTAL_UL,
            "fraction_sum_tolerance": SUM_TOLERANCE,
        }


def append_line(path: Path, obj: dict) -> None:
    """Append one JSON line with a single write, so two processes can share the file."""
    data = (json.dumps(obj) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _fractions_dict(fractions) -> dict:
    return {d: round(float(f), 4) for d, f in zip(DYES, fractions)}


class LabSession:
    """The state of one run and the guardrails around it. The MCP tools are thin wrappers."""

    def __init__(self, config: LabConfig):
        config.validate()
        self.config = config
        self.target_rgb = hex_to_rgb(config.target)
        self.reader = PlateReader(noise_sd=config.noise_sd, seed=config.seed)
        self.robot: Optional[ColourRobot] = None
        self.optimiser = None  # BayBE is imported on first use: it pulls in torch, which is slow
        self.wells: Dict[str, dict] = {}  # well name -> what went in and, once read, what came out
        self.rounds: List[dict] = []
        self.refusals: List[dict] = []
        self.best: Optional[dict] = None
        self.calls = 0
        self.lock = asyncio.Lock()
        self._suggested: List[List[float]] = []
        self._fed: set = set()
        config.record_path.parent.mkdir(parents=True, exist_ok=True)
        self._record({"type": "session", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "target": config.target,
                      "threshold": config.threshold, "seed": config.seed, "limits": config.limits()})

    # --- record -------------------------------------------------------------------------------

    def _record(self, obj: dict) -> None:
        append_line(self.config.record_path, obj)

    def log_call(self, tool: str, args: Any, ok: bool, result: Any = None, reason: str = "",
                 refused: bool = True) -> None:
        self.calls += 1
        entry = {"type": "call", "seq": self.calls, "time": round(time.time(), 3), "tool": tool, "args": args, "ok": ok}
        if ok:
            entry["result"] = _summarise(tool, result)
        else:
            entry["refused"] = refused
            entry["reason"] = reason
            if refused:
                self.refusals.append({"seq": self.calls, "tool": tool, "args": args, "reason": reason})
        self._record(entry)

    # --- helpers ------------------------------------------------------------------------------

    async def _robot(self) -> ColourRobot:
        if self.robot is None:
            self.robot = ColourRobot()
            await self.robot.setup()
        return self.robot

    @property
    def converged(self) -> bool:
        return self.best is not None and self.best["delta_e"] < self.config.threshold

    def budget(self) -> dict:
        return {"wells_used": len(self.wells), "max_wells": self.config.max_wells,
                "rounds_done": len(self.rounds), "max_rounds": self.config.max_rounds}

    @staticmethod
    def _well_name(value: Any) -> str:
        name = value.strip().upper() if isinstance(value, str) else None
        if name not in WELL_NAMES_SET:
            raise Refusal(f"unknown well {value!r}: the plate has wells A1-H12 (rows A-H, columns 1-12)")
        return name

    @staticmethod
    def _fractions(value: Any) -> List[float]:
        if isinstance(value, dict):
            unknown = set(value) - set(DYES)
            if unknown:
                raise Refusal(f"unknown dye(s) {sorted(unknown)}: the dyes are {', '.join(DYES)}")
            raw = [value.get(d, 0.0) for d in DYES]
        elif isinstance(value, (list, tuple)):
            if len(value) != len(DYES):
                raise Refusal(f"give {len(DYES)} fractions in the order {', '.join(DYES)}, got {len(value)}")
            raw = list(value)
        else:
            raise Refusal("fractions must be an object like {\"red\": 0.5, \"yellow\": 0.2, \"blue\": 0.3}")
        fractions = []
        for dye, f in zip(DYES, raw):
            if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f):
                raise Refusal(f"{dye} fraction must be a number, got {f!r}")
            if not 0.0 <= f <= 1.0:
                raise Refusal(f"{dye} fraction {f} is outside 0-1")
            fractions.append(float(f))
        total = sum(fractions)
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise Refusal(f"fractions sum to {total:.3f}; they must sum to 1 (within {SUM_TOLERANCE})")
        return [f / total for f in fractions]

    def _check_round_budget(self) -> None:
        if len(self.rounds) >= self.config.max_rounds:
            raise Refusal(f"the round limit for this run is used up ({self.config.max_rounds} measurement rounds)")

    # --- tools --------------------------------------------------------------------------------

    async def get_deck_state(self) -> dict:
        robot = await self._robot()
        return {
            "target": self.config.target,
            "threshold": self.config.threshold,
            "plate": {
                "labware": "NEST 96 well plate 200 uL flat, deck slot 5",
                "well_capacity_ul": robot.well_capacity_ul("A1"),
                "filled_wells": {name: {"fractions": w["fractions"], "read": w["reading"] is not None}
                                 for name, w in self.wells.items()},
                "empty_wells": len(WELL_NAMES) - len(self.wells),
            },
            "pipette": {"model": "P300 single-channel GEN2 (simulated)", "range_ul": [PIPETTE_MIN_UL, PIPETTE_MAX_UL],
                        "tips_left": robot.tips_left(), "tips_per_dispense": len(DYES)},
            "dye_stock_ul": robot.dye_left_ul(),
            "budget": self.budget(),
            "best_so_far": self.best,
            "converged": self.converged,
        }

    async def list_dyes(self) -> dict:
        robot = await self._robot()
        stock = robot.dye_left_ul()
        return {
            "dyes": [{"name": d, "looks": DYE_LOOKS[d], "stock_ul": stock[d]} for d in DYES],
            "fractions": "volume fractions of a well, one per dye, each 0-1, summing to 1",
            "standard_well_ul": WELL_TOTAL_UL,
            "pipette_range_ul": [PIPETTE_MIN_UL, PIPETTE_MAX_UL],
            "smallest_nonzero_fraction": round(MIN_FRACTION, 4),
        }

    async def dispense_mix(self, well: Any, fractions: Any, total_volume_ul: Any = WELL_TOTAL_UL) -> dict:
        name = self._well_name(well)
        if name in self.wells:
            raise Refusal(f"well {name} already holds a mix from this run; dispense into an empty well")
        if len(self.wells) >= self.config.max_wells:
            raise Refusal(f"the well limit for this run is used up ({self.config.max_wells} wells)")
        self._check_round_budget()
        mix = self._fractions(fractions)
        if isinstance(total_volume_ul, bool) or not isinstance(total_volume_ul, (int, float)) \
                or not math.isfinite(total_volume_ul) or total_volume_ul <= 0:
            raise Refusal(f"total_volume_ul must be a positive number, got {total_volume_ul!r}")
        total = float(total_volume_ul)
        volumes = fractions_to_volumes(mix, total)
        for dye, volume in zip(DYES, volumes):
            if volume and not PIPETTE_MIN_UL <= volume <= PIPETTE_MAX_UL:
                raise Refusal(
                    f"{dye} would be {volume:g} uL, outside the pipette's {PIPETTE_MIN_UL:g}-{PIPETTE_MAX_UL:g} uL "
                    f"range (a dye is either left out or at least {PIPETTE_MIN_UL:g} uL, which is a fraction of "
                    f"{MIN_FRACTION:.3f} of a {WELL_TOTAL_UL:g} uL well)")
        robot = await self._robot()
        capacity, current = robot.well_capacity_ul(name), robot.well_volume_ul(name)
        if current + total > capacity:
            raise Refusal(f"{total:g} uL would overfill well {name} (holds {capacity:g} uL, has {current:g} uL)")

        (fill,) = await robot.mix_wells([mix], wells=[name], total_ul=total)
        suggested = any(max(abs(a - b) for a, b in zip(mix, s)) < 0.005 for s in self._suggested)
        self.wells[name] = {
            "well": name,
            "order": len(self.wells) + 1,
            "decided_by": "suggestion" if suggested else "claude",
            "fractions": [round(fill.fractions[d], 4) for d in DYES],
            "dispensed_ul": fill.volumes_ul,
            "reading": None,
        }
        return {"well": name, "dispensed_ul": fill.volumes_ul, "fractions": _fractions_dict(self.wells[name]["fractions"]),
                "budget": self.budget()}

    async def read_plate(self, wells: Any) -> dict:
        if isinstance(wells, str):
            wells = [wells]
        if not isinstance(wells, list) or not wells:
            raise Refusal("wells must be a non-empty list of well names, like [\"A1\", \"B1\"]")
        names = list(dict.fromkeys(self._well_name(w) for w in wells))
        empty = [n for n in names if n not in self.wells]
        if empty:
            raise Refusal(f"well(s) {', '.join(empty)} are empty: nothing to read; dispense into them first")
        new = [n for n in names if self.wells[n]["reading"] is None]
        round_no = None
        if new:
            self._check_round_budget()
            round_no = len(self.rounds) + 1
            measured = []
            for n in new:
                w = self.wells[n]
                rgb = self.reader.read(w["dispensed_ul"])
                w["reading"] = {"round": round_no, "rgb": list(rgb), "hex": rgb_to_hex(rgb),
                                "delta_e": round(colour_distance(rgb, self.target_rgb), 3)}
                measured.append(self._well_view(n))
            round_best = min(measured, key=lambda m: m["delta_e"])
            if self.best is None or round_best["delta_e"] < self.best["delta_e"]:
                self.best = round_best
            self.rounds.append({"round": round_no, "wells": new, "round_best": round_best["delta_e"],
                                "best": self.best["delta_e"]})
            self._record({"type": "round", "round": round_no, "wells": measured,
                          "round_best_delta_e": round_best["delta_e"], "best_so_far": self.best,
                          "converged": self.converged})
        return {
            "readings": [dict(self._well_view(n), new=n in new) for n in names],
            "round": round_no,
            "best_so_far": self.best,
            "converged": self.converged,
            "threshold": self.config.threshold,
            "budget": self.budget(),
        }

    async def suggest_next_mixes(self, n: Any = 4) -> dict:
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_SUGGESTIONS:
            raise Refusal(f"n must be a whole number from 1 to {MAX_SUGGESTIONS}, got {n!r}")
        if self.optimiser is None:
            from .optimiser import Optimiser

            self.optimiser = await asyncio.to_thread(Optimiser, seed=self.config.seed, min_fraction=MIN_FRACTION)
        unfed = [w for name, w in self.wells.items() if w["reading"] is not None and name not in self._fed]
        if unfed:
            await asyncio.to_thread(self.optimiser.add, [w["fractions"] for w in unfed],
                                    [w["reading"]["delta_e"] for w in unfed])
            self._fed.update(w["well"] for w in unfed)
        proposals = await asyncio.to_thread(self.optimiser.recommend, n)
        self._suggested.extend(proposals)
        return {
            "suggestions": [_fractions_dict(p) for p in proposals],
            "based_on_readings": len(self._fed),
            "about": "BayBE (Bayesian optimisation) over the readings so far, minimising CIEDE2000 distance to the "
                     "target, on a 2 % grid of mixes the pipette can dispense. Suggestions are advice, not orders.",
        }

    async def get_run_record(self) -> dict:
        return {
            "target": self.config.target,
            "threshold": self.config.threshold,
            "limits": self.config.limits(),
            "budget": self.budget(),
            "wells": [self._well_view(n) for n in self.wells],
            "rounds": self.rounds,
            "best_so_far": self.best,
            "converged": self.converged,
            "refusals": self.refusals,
            "tool_calls_so_far": self.calls,
            "record_path": str(self.config.record_path),
        }

    def _well_view(self, name: str) -> dict:
        w = self.wells[name]
        view = {"well": name, "fractions": _fractions_dict(w["fractions"]), "dispensed_ul": w["dispensed_ul"],
                "decided_by": w["decided_by"]}
        if w["reading"] is not None:
            view.update(w["reading"])
        return view


WELL_NAMES_SET = frozenset(WELL_NAMES)


def _summarise(tool: str, result: Any) -> Any:
    """What the run record keeps of a successful call's result (the big read-only ones are summarised)."""
    if not isinstance(result, dict):
        return result
    if tool == "get_deck_state":
        return {"budget": result.get("budget"), "converged": result.get("converged")}
    if tool == "get_run_record":
        return {"budget": result.get("budget"), "refusals": len(result.get("refusals", [])),
                "converged": result.get("converged")}
    if tool == "list_dyes":
        return {"dyes": [d["name"] for d in result.get("dyes", [])]}
    return result


class LabServer(MCPServer):
    """``MCPServer`` that serialises tool calls and logs each one, refusals included, to the record."""

    def __init__(self, lab: LabSession):
        super().__init__(name=SERVER_NAME, instructions=INSTRUCTIONS, log_level="WARNING")
        self.lab = lab

    async def call_tool(self, name, arguments, context=None):
        async with self.lab.lock:
            try:
                result = await super().call_tool(name, arguments, context)
            except UnexpectedToolError as exc:
                self.lab.log_call(name, arguments, ok=False, reason=f"server error: {exc.__cause__!r}"[:300],
                                  refused=False)
                raise
            except ToolError as exc:
                cause = exc.__cause__
                if isinstance(cause, Refusal):
                    reason = cause.reason
                elif isinstance(cause, ValidationError):
                    reason = "invalid arguments: " + "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in cause.errors())
                else:
                    reason = str(exc)
                self.lab.log_call(name, arguments, ok=False, reason=reason)
                raise
            self.lab.log_call(name, arguments, ok=True, result=getattr(result, "structured_content", None))
            return result


def build_server(lab: LabSession) -> LabServer:
    server = LabServer(lab)

    @server.tool()
    async def get_deck_state() -> dict[str, Any]:
        """The deck right now: the target colour, which plate wells hold which mix and whether they have
        been read, tips and dye left, the run's budget (wells and measurement rounds used and allowed),
        the best well so far and whether the run has converged."""
        return await lab.get_deck_state()

    @server.tool()
    async def list_dyes() -> dict[str, Any]:
        """The three dye stocks (red, yellow, blue), what each looks like, and the volume rules for mixing them."""
        return await lab.list_dyes()

    @server.tool()
    async def dispense_mix(well: str, fractions: dict[str, Any], total_volume_ul: float = WELL_TOTAL_UL) -> dict[str, Any]:
        """Mix dyes into one empty plate well and return the volumes dispensed.

        well: a plate well, A1-H12, that is still empty.
        fractions: the volume fraction of each dye, e.g. {"red": 0.48, "yellow": 0.12, "blue": 0.40};
            each 0-1, summing to 1 (within 0.01). A dye left out counts as 0.
        total_volume_ul: how much to make up in the well (default 180 uL; a well holds 200 uL).

        Each dye's volume must be 0 or within the pipette's 20-300 uL range, so at the default volume a
        dye is either left out or at least 0.12 of the mix. Takes one fresh tip per dye. Counts against
        the run's well limit. Refused, with the reason, if any rule is broken."""
        return await lab.dispense_mix(well, fractions, total_volume_ul)

    @server.tool()
    async def read_plate(wells: list[str]) -> dict[str, Any]:
        """Measure filled wells with the plate reader: RGB, hex colour and CIEDE2000 distance (delta_e) to
        the target for each (about 1 is just visible; the run converges below the threshold). Reading
        wells not read before is one measurement round and counts against the round limit; a well is
        measured once, and reading it again returns the same measurement."""
        return await lab.read_plate(wells)

    @server.tool()
    async def suggest_next_mixes(n: int = 4) -> dict[str, Any]:
        """Ask the Bayesian optimiser (BayBE) for n (1-8) promising mixes given every reading so far.
        Optional: you decide what to dispense. Does not use any budget."""
        return await lab.suggest_next_mixes(n)

    @server.tool()
    async def get_run_record() -> dict[str, Any]:
        """Everything so far: every well with its mix, volumes and reading, the per-round best distance,
        the best well, the refused calls and why, and the budget left."""
        return await lab.get_run_record()

    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colour-loop-mcp", description="MCP tool server (stdio) for the simulated colour lab.")
    parser.add_argument("--target", default="#7a4b9c", help="target colour as #rrggbb, fixed for the run (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=2.0, help="converged below this CIEDE2000 distance (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="seed for reader noise and the optimiser (default: %(default)s)")
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE_SD, help="reader noise, SD in 0-255 counts (default: %(default)s)")
    parser.add_argument("--max-wells", type=int, default=32, help="per-run well limit (default: %(default)s, at most 32)")
    parser.add_argument("--max-rounds", type=int, default=8, help="per-run measurement-round limit (default: %(default)s)")
    parser.add_argument("--record", type=Path, default=None,
                        help="run record JSONL (default: ~/colour-loop-runs/mcp-<time>.jsonl)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # stdout carries the MCP protocol; everything else goes to stderr
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="colour-lab: %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    config = LabConfig(target=args.target, threshold=args.threshold, seed=args.seed, noise_sd=args.noise,
                       max_wells=args.max_wells, max_rounds=args.max_rounds)
    if args.record is not None:
        config.record_path = args.record.expanduser().resolve()
    try:
        lab = LabSession(config)
    except ValueError as exc:
        print(f"colour-loop-mcp: {exc}", file=sys.stderr)
        return 2
    build_server(lab).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
