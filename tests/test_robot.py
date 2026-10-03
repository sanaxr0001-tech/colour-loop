import asyncio

import pytest

from colour_loop.optimiser import simplex_grid, snap_to_grid
from colour_loop.robot import WELL_TOTAL_UL, ColourRobot, fractions_to_volumes


@pytest.mark.parametrize("fractions", [(1, 0, 0), (1 / 3, 1 / 3, 1 / 3), (0.123, 0.456, 0.421), (0.02, 0.0, 0.98)])
def test_volumes_always_sum_to_the_well_total(fractions):
    volumes = fractions_to_volumes(fractions)
    assert sum(volumes) == pytest.approx(WELL_TOTAL_UL)
    assert all(v >= 0 for v in volumes)


@pytest.mark.parametrize("fractions", [(0.334, 0.333, 0.333), (0.011, 0.5, 0.489), (0.7, 0.3, 0.0)])
def test_snap_to_grid_sums_to_one_and_stays_close(fractions):
    snapped = snap_to_grid(fractions)
    assert sum(snapped) == pytest.approx(1.0)
    assert max(abs(a - b) for a, b in zip(snapped, fractions)) <= 0.02


def test_grid_is_the_whole_simplex():
    grid = simplex_grid(50)
    assert len(grid) == 51 * 52 // 2
    assert (grid.sum(axis=1) - 1).abs().max() < 1e-9


def test_robot_dispenses_and_pylabrobot_tracks_it():
    async def go():
        robot = ColourRobot()
        await robot.setup()
        fills = await robot.mix_wells([(0.5, 0.1, 0.4), (1, 0, 0), (0.2, 0.2, 0.6), (1 / 3, 1 / 3, 1 / 3)])
        more = await robot.mix_wells([(0, 1, 0)])
        return robot, fills + more

    robot, fills = asyncio.run(go())
    assert [f.well for f in fills] == ["A1", "B1", "C1", "D1", "E1"]
    assert fills[0].volumes_ul == {"red": 90.0, "yellow": 18.0, "blue": 72.0}
    for fill in fills:
        # PyLabRobot's own volume tracker agrees with our per-dye ledger
        assert robot.well_volume_ul(fill.well) == pytest.approx(sum(fill.volumes_ul.values()))
    used = {dye: 12_000 - tube.tracker.get_used_volume() for dye, tube in robot.layout.dye_tubes.items()}
    assert used["red"] == pytest.approx(90 + 180 + 36 + 60)
    tips_used = sum(not spot.has_tip() for spot in robot.layout.tips.get_all_items())
    assert tips_used == 6  # one fresh tip per dye per call
