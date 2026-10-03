"""Command line.

    colour-loop run --target "#7a4b9c" --visual        the optimiser closes the loop (Claude reviews)
    colour-loop agent --target "#7a4b9c" --visual      Claude drives the lab through the MCP tools
    colour-loop compare                                both modes on the same targets and seeds
    colour-loop watch runs/agent.jsonl                 live view of any MCP run record
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings
import webbrowser
from pathlib import Path

from websockets.exceptions import ConnectionClosed

from .loop import LiveState, RunConfig, run
from .robot import ColourRobot
from .web import LiveServer


def _add_view_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=8765, help="port for the live view (default: %(default)s)")
    p.add_argument("--no-browser", action="store_true", help="with --visual: serve the view but do not open a browser")
    p.add_argument("--start-delay", type=float, default=2.0,
                   help="with --visual: seconds to wait for the page to connect before the run starts (default: %(default)s)")
    p.add_argument("--exit-when-done", action="store_true",
                   help="with --visual: exit when the run ends instead of keeping the view up")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colour-loop", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the closed loop once (the optimiser proposes, Claude reviews)")
    r.add_argument("--target", default="#7a4b9c", help="target colour as #rrggbb (default: %(default)s)")
    r.add_argument("--batch", type=int, default=4, help="wells per round (default: %(default)s)")
    r.add_argument("--max-rounds", type=int, default=10, help="stop after this many rounds (default: %(default)s)")
    r.add_argument("--threshold", type=float, default=2.0,
                   help="stop when a well is within this CIEDE2000 distance (default: %(default)s)")
    r.add_argument("--seed", type=int, default=0, help="seed for the optimiser and reader noise (default: %(default)s)")
    r.add_argument("--noise", type=float, default=1.5, help="reader noise, SD in 0-255 counts (default: %(default)s)")
    r.add_argument("--no-claude", action="store_true", help="optimiser only; do not call Claude")
    r.add_argument("--model", default="sonnet", help="Claude model for the headless call, e.g. sonnet or haiku (default: %(default)s)")
    r.add_argument("--claude-timeout", type=float, default=90.0, help="seconds per Claude call (default: %(default)s)")
    r.add_argument("--record", type=Path, default=None, help="JSONL run record (default: runs/run-<time>.jsonl)")
    r.add_argument("--visual", action="store_true",
                   help="open the live view: deck simulation plus results panel, in the browser")
    r.add_argument("--step-delay", type=float, default=None,
                   help="seconds to pause after each robot step so it can be watched (default: 0.25 with --visual, else 0)")
    _add_view_args(r)

    a = sub.add_parser("agent", help="Claude drives the lab through the colour-lab MCP tools (headless claude -p)")
    a.add_argument("--target", default="#7a4b9c", help="target colour as #rrggbb (default: %(default)s)")
    a.add_argument("--threshold", type=float, default=2.0, help="converged below this CIEDE2000 distance (default: %(default)s)")
    a.add_argument("--seed", type=int, default=0, help="seed for reader noise and the optimiser tool (default: %(default)s)")
    a.add_argument("--max-wells", type=int, default=32, help="per-run well limit the server enforces (default: %(default)s)")
    a.add_argument("--max-rounds", type=int, default=8, help="per-run round limit the server enforces (default: %(default)s)")
    a.add_argument("--model", default="sonnet", help="Claude model, e.g. sonnet or haiku (default: %(default)s)")
    a.add_argument("--effort", default="low", help="Claude effort level (default: %(default)s)")
    a.add_argument("--time-budget", type=float, default=600.0,
                   help="seconds for the whole Claude session before it is stopped (default: %(default)s)")
    a.add_argument("--record", type=Path, default=None, help="JSONL run record (default: runs/agent-<time>.jsonl)")
    a.add_argument("--visual", action="store_true",
                   help="open the live view: deck, results and Claude's tool calls as they happen")
    a.add_argument("--step-delay", type=float, default=0.15,
                   help="with --visual: seconds per robot step in the deck replay (default: %(default)s)")
    _add_view_args(a)

    c = sub.add_parser("compare", help="Claude-driven vs optimiser-only on the same targets and seeds")
    c.add_argument("--targets", default="#7a4b9c,#a84378,#5371ae",
                   help="comma-separated target colours (default: %(default)s)")
    c.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (default: %(default)s)")
    c.add_argument("--threshold", type=float, default=2.0, help="(default: %(default)s)")
    c.add_argument("--max-wells", type=int, default=32, help="well budget for both modes (default: %(default)s)")
    c.add_argument("--max-rounds", type=int, default=8, help="round budget for both modes (default: %(default)s)")
    c.add_argument("--model", default="sonnet", help="Claude model (default: %(default)s)")
    c.add_argument("--effort", default="low", help="Claude effort level (default: %(default)s)")
    c.add_argument("--time-budget", type=float, default=480.0, help="seconds per Claude run (default: %(default)s)")
    c.add_argument("--parallel", type=int, default=3, help="Claude runs at once (default: %(default)s)")
    c.add_argument("--no-claude", action="store_true", help="optimiser-only half only (no Claude)")
    c.add_argument("--out", type=Path, default=Path("docs"), help="where comparison.json and comparison.png go (default: %(default)s)")
    c.add_argument("--runs", type=Path, default=None, help="where the run records go (default: runs/compare-<time>/)")

    w = sub.add_parser("watch", help="live view of an MCP run record, e.g. one Claude Desktop or Claude Code is writing")
    w.add_argument("record", type=Path, help="the run record colour-loop-mcp writes (its --record path)")
    w.add_argument("--step-delay", type=float, default=0.15, help="seconds per robot step in the deck replay (default: %(default)s)")
    _add_view_args(w)
    return parser


def _quiet_closed_viewers(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # The Visualizer pushes deck updates to every open tab; a tab that was closed leaves a failed
    # send behind. That is harmless, so do not print a traceback for it.
    exc = context.get("exception")
    if exc is not None and type(exc).__module__.startswith("websockets"):
        return
    loop.default_exception_handler(context)


class View:
    """The live view: PyLabRobot's deck Visualizer for ``robot`` plus the results page."""

    def __init__(self, live: LiveState, robot: ColourRobot, args: argparse.Namespace):
        self.live, self.robot, self.args = live, robot, args
        self.server = self.visualizer = None

    async def open(self) -> None:
        from pylabrobot.visualizer import Visualizer

        self.visualizer = Visualizer(self.robot.lh, open_browser=False, name="colour-loop: simulated OT-2",
                                     show_machine_tools_at_start=False)
        await self.visualizer.setup()
        self.live.update(deck_url=f"http://{self.visualizer.host}:{self.visualizer.fs_port}/")
        self.server = LiveServer(self.live, port=self.args.port).start()
        print(f"Live view: {self.server.url}", flush=True)
        if not self.args.no_browser:
            webbrowser.open(self.server.url)
        await asyncio.sleep(self.args.start_delay)  # let the page connect before anything moves

    async def hold_and_close(self) -> None:
        if not self.args.exit_when_done:
            print("The live view stays up; press Ctrl-C to exit.", flush=True)
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass
        if self.visualizer is not None:
            try:
                await self.visualizer.stop()
            except ConnectionClosed:  # the browser tab is already gone
                pass
        if self.server is not None:
            self.server.stop()


async def _run(args: argparse.Namespace) -> int:
    asyncio.get_running_loop().set_exception_handler(_quiet_closed_viewers)
    config = RunConfig(
        target=args.target,
        batch_size=args.batch,
        max_rounds=args.max_rounds,
        threshold=args.threshold,
        seed=args.seed,
        noise_sd=args.noise,
        use_claude=not args.no_claude,
        claude_model=args.model,
        claude_timeout=args.claude_timeout,
        step_delay=args.step_delay if args.step_delay is not None else (0.25 if args.visual else 0.0),
    )
    if args.record is not None:
        config.record_path = args.record
    config.validate()
    live = LiveState(config)

    robot = ColourRobot(step_delay=config.step_delay)
    await robot.setup()
    view = None
    if args.visual:
        view = View(live, robot, args)
        await view.open()

    result = await run(config, live=live, robot=robot)

    state = "converged" if result.converged else "did not converge"
    print(
        f"\n{state} in {result.rounds} round{'s' if result.rounds != 1 else ''} ({result.seconds:.0f} s). Best well {result.best['well']}: "
        f"{result.best['hex']} vs target {config.target}, CIEDE2000 {result.best['delta_e']:.2f}.\n"
        f"Claude layer: {_claude_summary(config, result)}. Record: {result.record_path}",
        flush=True,
    )
    if view is not None:
        await view.hold_and_close()
    await robot.stop()
    return 0 if result.converged else 1


def _claude_summary(config: RunConfig, result) -> str:
    if not config.use_claude:
        return "off (--no-claude)"
    return "ran headless" if result.claude_ran else "did not run (see the log above)"


async def _agent(args: argparse.Namespace) -> int:
    from .agent import AgentConfig, agent_live_state, run_agent
    from .mcp_server import LabConfig

    asyncio.get_running_loop().set_exception_handler(_quiet_closed_viewers)
    config = AgentConfig(target=args.target, threshold=args.threshold, seed=args.seed, max_wells=args.max_wells,
                         max_rounds=args.max_rounds, model=args.model, effort=args.effort,
                         time_budget=args.time_budget)
    if args.record is not None:
        config.record_path = args.record
    LabConfig(target=config.target, max_wells=config.max_wells, max_rounds=config.max_rounds).validate()

    live = robot = view = None
    if args.visual:
        live = agent_live_state(config)
        robot = ColourRobot(step_delay=args.step_delay)  # mirrors the lab server's deck for the view
        await robot.setup()
        view = View(live, robot, args)
        await view.open()

    print(f"Claude ({config.model}) is driving the lab through colour-lab MCP tools; record: {config.record_path}",
          flush=True)
    result = await run_agent(config, live=live, deck_robot=robot)
    _print_agent_result(config, result)
    if view is not None:
        await view.hold_and_close()
        await robot.stop()
    if result.error and not result.summary:
        return 2
    return 0 if result.summary.get("converged") else 1


def _print_agent_result(config, result) -> None:
    s = result.summary
    if result.error:
        print(f"\n{result.error}", flush=True)
    if not s:
        return
    state = "converged" if s["converged"] else "did not converge"
    best = f"{s['best_delta_e']:.2f}" if s["best_delta_e"] is not None else "-"
    print(f"\n{state}: {s['rounds']} round{'s' if s['rounds'] != 1 else ''}, {s['wells']} wells, best CIEDE2000 {best} "
          f"vs target {config.target} ({result.seconds:.0f} s).\n"
          f"{s['tool_calls']} tool calls, {s['refusals']} refused, {s['suggestions_used']} wells from the optimiser's "
          f"suggestions. Record: {result.record_path}", flush=True)
    for reason in s["refusal_reasons"]:
        print(f"  refused - {reason}", flush=True)


async def _compare(args: argparse.Namespace) -> int:
    from .compare import CompareConfig, run_compare

    config = CompareConfig(
        targets=[t.strip() for t in args.targets.split(",") if t.strip()],
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        threshold=args.threshold, max_wells=args.max_wells, max_rounds=args.max_rounds, model=args.model,
        effort=args.effort, time_budget=args.time_budget, parallel=args.parallel, with_claude=not args.no_claude,
        out_dir=args.out)
    if args.runs is not None:
        config.runs_dir = args.runs
    report = await run_compare(config)
    for mode, agg in report["summary"].items():
        print(f"{mode:>15}: converged {agg['converged']}/{agg['runs']}, mean rounds {agg['mean_rounds']}, "
              f"mean wells {agg['mean_wells']}, mean seconds {agg['mean_seconds']}, refusals {agg['refusals']}", flush=True)
    print(f"wrote {config.out_dir / 'comparison.json'} and {config.out_dir / 'comparison.png'}", flush=True)
    return 0


async def _watch(args: argparse.Namespace) -> int:
    from .agent import RecordFollower

    asyncio.get_running_loop().set_exception_handler(_quiet_closed_viewers)
    live = LiveState(RunConfig(), mode="agent")
    live.update(status=f"watching {args.record}")
    robot = ColourRobot(step_delay=args.step_delay)
    await robot.setup()
    view = View(live, robot, argparse.Namespace(**{**vars(args), "exit_when_done": True}))
    await view.open()
    follower = RecordFollower(args.record, live, robot)
    print("Following the record; press Ctrl-C to exit.", flush=True)
    try:
        await follower.run()
    except asyncio.CancelledError:
        pass
    await view.hold_and_close()
    await robot.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # torch and BayBE announce API deprecations on import; they are not actionable here
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for noisy in ("pylabrobot", "websockets", "baybe", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    command = {"run": _run, "agent": _agent, "compare": _compare, "watch": _watch}[args.command]
    try:
        return asyncio.run(command(args))
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
