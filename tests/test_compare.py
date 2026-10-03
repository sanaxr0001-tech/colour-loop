"""The optimiser-only half of the comparison, with a fixed seed (Claude is never called here)."""

import asyncio
import json

from colour_loop.compare import OPTIMISER, CompareConfig, run_compare
from colour_loop.robot import PIPETTE_MIN_UL


def compare(tmp_path, name):
    config = CompareConfig(targets=["#7a4b9c", "#5371ae"], seeds=[0], with_claude=False,
                           out_dir=tmp_path / name, runs_dir=tmp_path / name / "runs")
    return asyncio.run(run_compare(config)), config


def test_optimiser_only_comparison(tmp_path):
    report, config = compare(tmp_path, "a")
    assert (config.out_dir / "comparison.png").stat().st_size > 10_000
    assert json.loads((config.out_dir / "comparison.json").read_text()) == report

    summary = report["summary"][OPTIMISER]
    assert summary["runs"] == 2 and summary["converged"] == 2 and summary["refusals"] == 0
    for run in report["runs"]:
        assert run["mode"] == OPTIMISER and run["converged"]
        assert run["rounds"] <= config.max_rounds and run["wells"] == 4 * run["rounds"]
        assert run["curve"] == sorted(run["curve"], reverse=True)  # best so far never gets worse
        # the optimiser works under the same pipette range the MCP server enforces
        for line in open(run["record"]):
            for well in json.loads(line)["wells"]:
                assert all(v == 0 or v >= PIPETTE_MIN_UL for v in well["dispensed_ul"].values())


def test_same_seed_same_comparison(tmp_path):
    a, _ = compare(tmp_path, "a")
    b, _ = compare(tmp_path, "b")
    assert [r["curve"] for r in a["runs"]] == [r["curve"] for r in b["runs"]]
