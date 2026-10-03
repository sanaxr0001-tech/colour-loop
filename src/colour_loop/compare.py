"""Claude-driven vs optimiser-only, on the same targets, seeds and lab limits.

Optimiser-only is the closed loop from ``loop.py`` with Claude off: BayBE proposes 4 mixes a round.
It searches only mixes the pipette can dispense (each dye left out or at least 20 uL), and gets the
same budget of wells and rounds as the MCP server gives Claude. Claude-driven is ``agent.py``: Claude
decides everything through the MCP tools and may ask BayBE for suggestions. Both use the same seed
for the simulated plate reader's noise, so the same mix reads the same.

Writes ``comparison.json`` (every run plus a per-mode summary) and ``comparison.png`` (distance to
target by round for every run, one panel per target).
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .loop import RunConfig, run
from .mcp_server import MIN_FRACTION

OPTIMISER, CLAUDE = "optimiser-only", "claude-driven"
BATCH = 4


@dataclass
class CompareConfig:
    targets: List[str] = field(default_factory=lambda: ["#7a4b9c", "#a84378", "#5371ae"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    threshold: float = 2.0
    max_wells: int = 32
    max_rounds: int = 8
    model: str = "sonnet"
    effort: str = "low"
    time_budget: float = 480.0
    parallel: int = 3
    with_claude: bool = True
    out_dir: Path = Path("docs")
    runs_dir: Path = field(default_factory=lambda: Path("runs") / f"compare-{time.strftime('%Y%m%d-%H%M%S')}")


def summarise_loop_record(path: Path, seconds: float) -> dict:
    """The same fields as ``agent.summarise_record``, for a closed-loop (optimiser-only) record."""
    rounds = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    converged_at = next((r["round"] for r in rounds if r["converged"]), None)
    return {
        "target": rounds[0]["target"] if rounds else None,
        "rounds": len(rounds),
        "wells": sum(len(r["wells"]) for r in rounds),
        "converged": converged_at is not None,
        "rounds_to_converge": converged_at,
        "wells_to_converge": sum(len(r["wells"]) for r in rounds[:converged_at]) if converged_at else None,
        "best_delta_e": rounds[-1]["best_so_far"]["delta_e"] if rounds else None,
        "curve": [r["best_so_far"]["delta_e"] for r in rounds],
        "refusals": 0,  # the loop only proposes mixes inside the limits; there is nothing to refuse
        "seconds": round(seconds, 1),
    }


async def optimiser_run(target: str, seed: int, config: CompareConfig) -> dict:
    record = config.runs_dir / f"optimiser-{target.lstrip('#')}-seed{seed}.jsonl"
    record.unlink(missing_ok=True)
    rounds = min(config.max_rounds, config.max_wells // BATCH)
    loop_config = RunConfig(target=target, batch_size=BATCH, max_rounds=rounds, threshold=config.threshold,
                            seed=seed, use_claude=False, record_path=record, min_fraction=MIN_FRACTION)
    started = time.monotonic()
    await run(loop_config)
    summary = summarise_loop_record(record, time.monotonic() - started)
    return {"mode": OPTIMISER, "target": target, "seed": seed, "record": str(record), **summary}


async def claude_run(target: str, seed: int, config: CompareConfig, gate: asyncio.Semaphore) -> dict:
    from .agent import AgentConfig, run_agent

    record = config.runs_dir / f"claude-{target.lstrip('#')}-seed{seed}.jsonl"
    record.unlink(missing_ok=True)
    agent_config = AgentConfig(target=target, threshold=config.threshold, seed=seed, max_wells=config.max_wells,
                               max_rounds=config.max_rounds, model=config.model, effort=config.effort,
                               time_budget=config.time_budget, record_path=record)
    async with gate:
        result = await run_agent(agent_config)
    summary = dict(result.summary)
    summary.setdefault("seconds", round(result.seconds, 1))
    return {"mode": CLAUDE, "target": target, "seed": seed, "record": str(record), "error": result.error, **summary}


def _aggregate(runs: List[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 1) if values else None

    return {
        "runs": len(runs),
        "converged": sum(1 for r in runs if r.get("converged")),
        "mean_rounds": mean(r.get("rounds") for r in runs),
        "mean_wells": mean(r.get("wells") for r in runs),
        "mean_rounds_to_converge": mean(r.get("rounds_to_converge") for r in runs),
        "mean_wells_to_converge": mean(r.get("wells_to_converge") for r in runs),
        "mean_best_delta_e": mean(r.get("best_delta_e") for r in runs),
        "mean_seconds": mean(r.get("seconds") for r in runs),
        "refusals": sum(r.get("refusals") or 0 for r in runs),
    }


async def run_compare(config: CompareConfig) -> dict:
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[dict] = []
    # The optimiser runs share this process (BayBE is CPU-bound), so they go first, one at a time;
    # the Claude runs are separate processes and can overlap.
    for target in config.targets:
        for seed in config.seeds:
            runs.append(await optimiser_run(target, seed, config))
    if config.with_claude:
        gate = asyncio.Semaphore(max(1, config.parallel))
        runs += await asyncio.gather(*(claude_run(t, s, config, gate) for t in config.targets for s in config.seeds))

    modes = [OPTIMISER] + ([CLAUDE] if config.with_claude else [])
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {"targets": config.targets, "seeds": config.seeds, "threshold": config.threshold,
                     "max_wells": config.max_wells, "max_rounds": config.max_rounds,
                     "optimiser_wells_per_round": BATCH, "claude_model": config.model if config.with_claude else None,
                     "claude_effort": config.effort if config.with_claude else None,
                     "min_dye_fraction": round(MIN_FRACTION, 4)},
        "summary": {m: _aggregate([r for r in runs if r["mode"] == m]) for m in modes},
        "runs": runs,
    }
    (config.out_dir / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")
    chart(report, config.out_dir / "comparison.png")
    return report


def chart(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = report["settings"]["targets"]
    threshold = report["settings"]["threshold"]
    colours = {OPTIMISER: "#3b5bdb", CLAUDE: "#c2410c"}
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.6), sharey=True, squeeze=False)
    for ax, target in zip(axes[0], targets):
        for mode in colours:
            runs = [r for r in report["runs"] if r["target"] == target and r["mode"] == mode and r.get("curve")]
            if not runs:
                continue
            agg = _aggregate(runs)
            label = (f"{mode}: {agg['converged']}/{agg['runs']} converged, "
                     f"mean {agg['mean_rounds']:g} rounds, {agg['mean_wells']:g} wells")
            if agg["refusals"]:
                label += f", {agg['refusals']} refused"
            for i, r in enumerate(runs):
                curve = r["curve"]
                ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=4, linewidth=2,
                        color=colours[mode], alpha=0.75, label=label if i == 0 else None)
        ax.axhline(threshold, color="#2f7d32", linestyle="--", linewidth=1)
        ax.text(0.98, threshold, f"stop at {threshold:g}", color="#2f7d32", fontsize=8, ha="right", va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_title(target, fontsize=11)
        ax.add_patch(plt.Rectangle((0.0, 1.02), 0.1, 0.07, transform=ax.transAxes, color=target, clip_on=False))
        ax.set_xlabel("measurement round")
        ax.set_xlim(0.5, report["settings"]["max_rounds"] + 0.5)
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#ece8e1", linewidth=0.8)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    axes[0][0].set_ylabel("best CIEDE2000 distance so far")
    model = report["settings"].get("claude_model")
    seeds = ", ".join(str(s) for s in report["settings"]["seeds"])
    fig.suptitle("Claude driving the lab through MCP tools vs the optimiser alone"
                 + (f" (Claude: {model})" if model else "") + f"; one line per run, seeds {seeds}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
