"""The closed loop: propose -> (Claude reviews) -> robot mixes -> reader measures -> learn -> repeat."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .claude_layer import Advice, ClaudeAdvisor
from .colour import colour_distance, hex_to_rgb, rgb_to_hex
from .optimiser import Optimiser, simplex_grid, snap_to_grid
from .physics import DEFAULT_NOISE_SD, DYES, PlateReader, ideal_rgb_from_fractions
from .robot import ColourRobot

log = logging.getLogger("colour_loop")


@dataclass
class RunConfig:
    target: str = "#7a4b9c"
    batch_size: int = 4
    max_rounds: int = 10
    threshold: float = 2.0  # stop once a well is within this CIEDE2000 distance of the target
    seed: Optional[int] = 0
    noise_sd: float = DEFAULT_NOISE_SD
    use_claude: bool = True
    claude_model: str = "sonnet"
    claude_timeout: float = 90.0
    record_path: Path = field(default_factory=lambda: Path("runs") / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    step_delay: float = 0.0

    def validate(self) -> None:
        hex_to_rgb(self.target)
        if not 1 <= self.batch_size <= 8:
            raise ValueError("batch size must be 1-8")
        if self.max_rounds < 1:
            raise ValueError("max rounds must be at least 1")
        if self.batch_size * self.max_rounds > 96:
            raise ValueError("batch size x max rounds must fit on one 96-well plate")
        if self.max_rounds * len(DYES) > 96:
            raise ValueError("not enough tips for that many rounds")


class LiveState:
    """Thread-safe snapshot of the run for the live web view."""

    def __init__(self, config: RunConfig):
        self._lock = threading.Lock()
        self._state = {
            "target": config.target,
            "threshold": config.threshold,
            "max_rounds": config.max_rounds,
            "batch_size": config.batch_size,
            "status": "starting",
            "round": 0,
            "activity": "",
            "wells": [],
            "rounds": [],
            "best": None,
            "notes": [],
            "claude": {"enabled": config.use_claude, "reason": ""},
            "finished": False,
            "deck_url": None,
        }

    def update(self, **changes) -> None:
        with self._lock:
            self._state.update(changes)

    def append(self, key: str, item) -> None:
        with self._lock:
            self._state[key] = [*self._state[key], item]

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._state))


@dataclass
class RunResult:
    rounds: int
    converged: bool
    best: dict
    record_path: Path
    seconds: float
    claude_ran: bool


def closest_reachable(target_rgb) -> float:
    """Smallest noise-free distance any grid mix can reach. Only a simulator can know this."""
    grid = simplex_grid()
    return min(colour_distance(ideal_rgb_from_fractions(row), target_rgb) for row in grid.to_numpy())


def _best(wells: List[dict]) -> dict:
    return min(wells, key=lambda w: w["delta_e"])


async def run(config: RunConfig, live: Optional[LiveState] = None, robot: Optional[ColourRobot] = None) -> RunResult:
    config.validate()
    live = live or LiveState(config)
    target_rgb = hex_to_rgb(config.target)
    started = time.monotonic()
    floor = closest_reachable(target_rgb)
    if floor >= config.threshold:
        log.warning("Simulator check: the closest colour these three dyes can make is dE %.2f from %s, "
                    "so the loop cannot get below the %.1f threshold; it will stop at max rounds.",
                    floor, config.target, config.threshold)

    advisor: Optional[ClaudeAdvisor] = None
    if config.use_claude:
        advisor = ClaudeAdvisor(model=config.claude_model, timeout=config.claude_timeout)
        if not advisor.check():
            log.warning("Claude layer off: %s. Continuing optimiser-only.", advisor.unavailable_reason)
            live.update(claude={"enabled": False, "reason": advisor.unavailable_reason})
            advisor = None
    else:
        live.update(claude={"enabled": False, "reason": "--no-claude"})

    async def on_robot_event(event: dict) -> None:
        if event["kind"] == "dispense":
            live.update(activity=f"dispensing {event['volume_ul']:.1f} uL {event['dye']} into {event['well']}")
        elif event["kind"] == "tip":
            verb = "picking up a fresh tip for" if event["action"] == "pick_up" else "discarding the tip used for"
            live.update(activity=f"{verb} {event['dye']}")

    if robot is None:
        robot = ColourRobot(step_delay=config.step_delay)
        await robot.setup()
    robot.on_event = on_robot_event
    reader = PlateReader(noise_sd=config.noise_sd, seed=config.seed)
    optimiser = Optimiser(seed=config.seed)

    config.record_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    converged = False
    claude_ran = False
    round_no = 0

    with config.record_path.open("a") as record:
        for round_no in range(1, config.max_rounds + 1):
            live.update(round=round_no, status="optimiser proposing", activity="BayBE is choosing the next mixes")
            proposals = optimiser.recommend(config.batch_size)

            advice = Advice()
            mixes = [list(p) for p in proposals]
            decided_by = ["optimiser"] * len(mixes)
            if advisor is not None:
                live.update(status="Claude reviewing", activity=f"asking Claude ({config.claude_model}) about the proposals")
                advice = advisor.advise(config.target, history, proposals, _best(history) if history else None)
                if advice.error:
                    log.warning("round %d: Claude call failed (%s); using the optimiser's proposals", round_no, advice.error)
                    live.append("notes", {"round": round_no, "note": f"(no note: {advice.error})",
                                          "override": False, "why": ""})
                else:
                    claude_ran = True
                if advice.override is not None:
                    i = advice.override.index
                    mixes[i] = snap_to_grid(advice.override.fractions)
                    decided_by[i] = "claude"
                if advice.note:
                    live.append("notes", {"round": round_no, "note": advice.note,
                                          "override": decided_by.count("claude") > 0,
                                          "why": advice.override.why if advice.override else ""})

            live.update(status="robot mixing")
            fills = await robot.mix_wells(mixes)

            live.update(status="plate reader measuring", activity="reading the new wells")
            wells = []
            for fill, proposed, who in zip(fills, proposals, decided_by):
                rgb = reader.read(fill.volumes_ul)
                distance = colour_distance(rgb, target_rgb)
                well = {
                    "round": round_no,
                    "well": fill.well,
                    "decided_by": who,
                    "proposed_fractions": [round(f, 4) for f in proposed],
                    "fractions": [round(fill.fractions[d], 4) for d in DYES],
                    "dispensed_ul": fill.volumes_ul,
                    "rgb": list(rgb),
                    "hex": rgb_to_hex(rgb),
                    "delta_e": round(distance, 3),
                }
                wells.append(well)
                live.append("wells", well)
            optimiser.add([w["fractions"] for w in wells], [w["delta_e"] for w in wells])
            history.extend(wells)

            best = _best(history)
            round_best = _best(wells)
            converged = best["delta_e"] < config.threshold
            live.append("rounds", {"round": round_no, "round_best": round_best["delta_e"], "best": best["delta_e"]})
            live.update(best=best)

            record.write(json.dumps({
                "round": round_no,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "target": config.target,
                "proposals": [[round(f, 4) for f in p] for p in proposals],
                "claude": {
                    "asked": advisor is not None,
                    "note": advice.note,
                    "override": asdict(advice.override) if advice.override else None,
                    "rejected_override": advice.rejected,
                    "error": advice.error,
                    "seconds": round(advice.seconds, 2),
                },
                "wells": wells,
                "round_best_delta_e": round_best["delta_e"],
                "best_so_far": best,
                "converged": converged,
            }) + "\n")
            record.flush()

            who = " (1 mix chosen by Claude)" if "claude" in decided_by else ""
            log.info("round %d: best this round dE=%.2f at %s %s, best so far dE=%.2f%s",
                     round_no, round_best["delta_e"], round_best["well"], round_best["hex"], best["delta_e"], who)
            if advice.note:
                log.info("  Claude: %s", advice.note)
            if converged:
                break

    seconds = time.monotonic() - started
    best = _best(history)
    verdict = "converged" if converged else "stopped at max rounds"
    live.update(status=f"{verdict} after {round_no} rounds", activity=f"best well {best['well']} at dE {best['delta_e']:.2f}", finished=True)
    return RunResult(rounds=round_no, converged=converged, best=best, record_path=config.record_path,
                     seconds=seconds, claude_ran=claude_ran)
