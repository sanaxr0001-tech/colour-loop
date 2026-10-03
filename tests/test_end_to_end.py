"""A whole optimiser-only run: simulated robot, simulated reader, BayBE, fixed seed, no Claude."""

import asyncio
import json

from colour_loop.loop import RunConfig, run


def run_once(tmp_path, name, **overrides):
    config = RunConfig(use_claude=False, seed=0, record_path=tmp_path / f"{name}.jsonl", **overrides)
    result = asyncio.run(run(config))
    records = [json.loads(line) for line in config.record_path.read_text().splitlines()]
    return result, records


def test_optimiser_only_run_converges(tmp_path):
    result, records = run_once(tmp_path, "a")
    assert result.converged
    assert result.best["delta_e"] < 2.0
    assert result.rounds <= 8
    assert not result.claude_ran
    assert len(records) == result.rounds

    first = records[0]
    assert first["round"] == 1 and len(first["proposals"]) == 4 and len(first["wells"]) == 4
    assert first["claude"]["asked"] is False
    well = first["wells"][0]
    assert set(well) >= {"well", "decided_by", "dispensed_ul", "rgb", "hex", "delta_e", "fractions"}
    assert well["decided_by"] == "optimiser"
    assert abs(sum(well["dispensed_ul"].values()) - 180.0) < 1e-6
    assert records[-1]["converged"] is True
    # best-so-far never gets worse
    bests = [r["best_so_far"]["delta_e"] for r in records]
    assert bests == sorted(bests, reverse=True)


def test_same_seed_same_run(tmp_path):
    _, a = run_once(tmp_path, "a", max_rounds=3, threshold=0.0)
    _, b = run_once(tmp_path, "b", max_rounds=3, threshold=0.0)
    strip = lambda recs: [[(w["well"], w["fractions"], w["rgb"]) for w in r["wells"]] for r in recs]
    assert strip(a) == strip(b)
