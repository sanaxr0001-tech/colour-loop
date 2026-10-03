"""Record the live view of a real run to docs/demo.mp4 and docs/demo.gif.

Starts ``colour-loop run --visual`` (or ``colour-loop agent --visual``), opens the live view in
headless Chromium with Playwright's video recording on, waits for the run to finish, then converts
the recording with ffmpeg.

    python scripts/record_demo.py                     # with Claude, if the CLI is logged in
    python scripts/record_demo.py --no-claude         # optimiser only
    python scripts/record_demo.py --agent             # Claude drives through the MCP tools:
                                                      #   docs/agent-demo.mp4, .gif, -run.jsonl
    python scripts/record_demo.py --chromium /path/to/chrome

Needs: ``pip install -e '.[video]'``, ``playwright install chromium`` (or ``--chromium``), ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT = 1600, 900

# Zoom the Visualizer onto deck slots 1, 4 and 5 (tips, dyes, plate). Deck coordinates are in mm;
# the stage transform follows the convention of the Visualizer's own fitToViewport().
FOCUS_ON_LABWARE = """() => {
  const box = {x0: 105, x1: 386, y0: 58, y1: 254};
  const w = stage.width(), h = stage.height(), pad = 24;
  const s = Math.min((w - 2 * pad) / (box.x1 - box.x0), (h - 2 * pad) / (box.y1 - box.y0));
  stage.scaleX(s); stage.scaleY(-s);
  stage.x(w / 2 - (box.x0 + box.x1) / 2 * s);
  stage.y(h / 2 + (box.y0 + box.y1) / 2 * s - h * s);
  refreshScaleOverlays();
}"""


def wait_for_url(proc: subprocess.Popen, timeout: float = 120) -> str:
    deadline = time.monotonic() + timeout
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"colour-loop exited early with code {proc.returncode}")
            continue
        print("  run:", line.rstrip())
        match = re.search(r"Live view: (\S+)", line)
        if match:
            return match.group(1)
    raise TimeoutError("the live view never came up")


def state(url: str) -> dict:
    with urllib.request.urlopen(url + "state", timeout=5) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="#7a4b9c")
    parser.add_argument("--no-claude", action="store_true")
    parser.add_argument("--agent", action="store_true", help="record `colour-loop agent` (Claude drives via MCP)")
    parser.add_argument("--model", default="sonnet", help="with --agent: Claude model (default: %(default)s)")
    parser.add_argument("--gif-speed", type=float, default=None,
                        help="speed-up for the GIF only (default: 1)")
    parser.add_argument("--chromium", default=None, help="path to a Chromium/Chrome executable")
    parser.add_argument("--out", type=Path, default=ROOT / "docs")
    parser.add_argument("--record", type=Path, default=None,
                        help="run record (default: docs/demo-run.jsonl, or docs/agent-demo-run.jsonl with --agent)")
    args = parser.parse_args()
    name = "agent-demo" if args.agent else "demo"
    if args.record is None:
        args.record = args.out / f"{name}-run.jsonl"
    speed = args.gif_speed or 1.0

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is needed to convert the recording")
    args.out.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.unlink(missing_ok=True)

    cmd = [sys.executable, "-u", "-m", "colour_loop.cli", "agent" if args.agent else "run", "--visual",
           "--no-browser", "--target", args.target, "--start-delay", "6", "--record", str(args.record)]
    if args.agent:
        cmd += ["--model", args.model]
    elif args.no_claude:
        cmd.append("--no-claude")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    workdir = Path(tempfile.mkdtemp(prefix="colour-loop-video-"))
    try:
        url = wait_for_url(proc)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chromium)
            context = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT},
                                          record_video_dir=str(workdir),
                                          record_video_size={"width": WIDTH, "height": HEIGHT})
            page = context.new_page()
            page.goto(url)
            # Give the deck the whole iframe: collapse the Visualizer's resource tree.
            deck = page.frame_locator("iframe")
            deck.locator("#toolbar-right-toggle").wait_for(state="attached", timeout=15_000)
            deck_frame = next(f for f in page.frames if f != page.main_frame)
            deck_frame.wait_for_function("() => typeof stage !== 'undefined' && stage.width() > 0", timeout=15_000)
            page.wait_for_timeout(1000)  # let the Visualizer finish wiring up its buttons
            for _ in range(3):
                deck_frame.evaluate("() => document.getElementById('toolbar-right-toggle').click()")
                page.wait_for_timeout(500)
                if deck_frame.evaluate("() => document.getElementById('sidepanel').classList.contains('collapsed')"):
                    break
            deck_frame.evaluate(FOCUS_ON_LABWARE)
            started = time.monotonic()
            while not state(url)["finished"]:
                if time.monotonic() - started > 900:
                    raise TimeoutError("run did not finish within 15 minutes")
                time.sleep(0.5)
            page.wait_for_timeout(5000)  # hold the final frame
            video = page.video
            context.close()
            browser.close()
            webm = Path(video.path())
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=20)
            for line in (out or "").splitlines():
                print("  run:", line)
        except subprocess.TimeoutExpired:
            proc.kill()

    mp4 = args.out / f"{name}.mp4"
    gif = args.out / f"{name}.gif"
    # Drop the first second (blank page while loading); H.264 at modest quality keeps it small.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(webm),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(mp4)], check=True)
    palette = workdir / "palette.png"
    scale = f"setpts=PTS/{speed:g},fps=6,scale=1000:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(webm),
                    "-vf", f"{scale},palettegen=max_colors=96", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(webm), "-i", str(palette),
                    "-lavfi", f"{scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4", str(gif)],
                   check=True)
    shutil.rmtree(workdir, ignore_errors=True)
    for f in (mp4, gif):
        print(f"wrote {f.relative_to(ROOT)} ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
