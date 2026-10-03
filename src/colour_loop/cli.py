"""Command line: ``colour-loop run --target "#7a4b9c" --visual``."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colour-loop", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the closed loop once")
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
    r.add_argument("--port", type=int, default=8765, help="port for the live view (default: %(default)s)")
    r.add_argument("--no-browser", action="store_true", help="with --visual: serve the view but do not open a browser")
    r.add_argument("--step-delay", type=float, default=None,
                   help="seconds to pause after each robot step so it can be watched (default: 0.25 with --visual, else 0)")
    r.add_argument("--start-delay", type=float, default=2.0,
                   help="with --visual: seconds to wait for the page to connect before the robot starts (default: %(default)s)")
    r.add_argument("--exit-when-done", action="store_true",
                   help="with --visual: exit when the run ends instead of keeping the view up")
    return parser


def _quiet_closed_viewers(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # The Visualizer pushes deck updates to every open tab; a tab that was closed leaves a failed
    # send behind. That is harmless, so do not print a traceback for it.
    exc = context.get("exception")
    if exc is not None and type(exc).__module__.startswith("websockets"):
        return
    loop.default_exception_handler(context)


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
    server = visualizer = None
    if args.visual:
        from pylabrobot.visualizer import Visualizer

        visualizer = Visualizer(robot.lh, open_browser=False, name="colour-loop: simulated OT-2",
                                show_machine_tools_at_start=False)
        await visualizer.setup()
        live.update(deck_url=f"http://{visualizer.host}:{visualizer.fs_port}/")
        server = LiveServer(live, port=args.port).start()
        print(f"Live view: {server.url}", flush=True)
        if not args.no_browser:
            webbrowser.open(server.url)
        await asyncio.sleep(args.start_delay)  # let the page connect before the robot starts moving

    result = await run(config, live=live, robot=robot)

    state = "converged" if result.converged else "did not converge"
    print(
        f"\n{state} in {result.rounds} round{'s' if result.rounds != 1 else ''} ({result.seconds:.0f} s). Best well {result.best['well']}: "
        f"{result.best['hex']} vs target {config.target}, CIEDE2000 {result.best['delta_e']:.2f}.\n"
        f"Claude layer: {_claude_summary(config, result)}. Record: {result.record_path}",
        flush=True,
    )

    if args.visual and not args.exit_when_done:
        print("The live view stays up; press Ctrl-C to exit.", flush=True)
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
    if visualizer is not None:
        try:
            await visualizer.stop()
        except ConnectionClosed:  # the browser tab is already gone
            pass
    if server is not None:
        server.stop()
    await robot.stop()
    return 0 if result.converged else 1


def _claude_summary(config: RunConfig, result) -> str:
    if not config.use_claude:
        return "off (--no-claude)"
    return "ran headless" if result.claude_ran else "did not run (see the log above)"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # torch and BayBE announce API deprecations on import; they are not actionable here
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for noisy in ("pylabrobot", "websockets", "baybe", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
