"""BayBE proposes the next mixes.

The search space is every 3-dye mix on a 2 % grid of the simplex (red + yellow + blue = 1), 1326
candidates. A discrete grid keeps BayBE fast and makes every proposal directly pipettable; at a
180 uL well a 2 % step is 3.6 uL. The objective is the CIEDE2000 distance to the target, minimised.
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd
from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.settings import Settings
from baybe.targets import NumericalTarget

from .physics import DYES

GRID_STEPS = 50  # 1 / 50 = 2 % fraction resolution
TARGET = "delta_e"


def simplex_grid(steps: int = GRID_STEPS) -> pd.DataFrame:
    rows = [
        (i / steps, j / steps, (steps - i - j) / steps)
        for i in range(steps + 1)
        for j in range(steps + 1 - i)
    ]
    return pd.DataFrame(rows, columns=list(DYES))


def snap_to_grid(fractions: Sequence[float], steps: int = GRID_STEPS) -> List[float]:
    """Round a mix to the nearest grid point that still sums to exactly 1."""
    units = [round(f * steps) for f in fractions]
    # put the rounding error on the component that was rounded the furthest
    error = steps - sum(units)
    if error:
        residuals = [f * steps - u for f, u in zip(fractions, units)]
        order = sorted(range(len(units)), key=lambda i: residuals[i], reverse=error > 0)
        for i in order[: abs(error)]:
            units[i] += 1 if error > 0 else -1
    return [u / steps for u in units]


class Optimiser:
    """Thin wrapper around a BayBE campaign over dye fractions."""

    def __init__(self, seed: int | None = 0, steps: int = GRID_STEPS):
        if seed is not None:
            Settings(random_seed=seed).activate()
        grid = simplex_grid(steps)
        parameters = [
            NumericalDiscreteParameter(name=d, values=sorted(grid[d].unique()), tolerance=1e-6)
            for d in DYES
        ]
        searchspace = SearchSpace(discrete=SubspaceDiscrete.from_dataframe(grid, parameters=parameters))
        objective = SingleTargetObjective(NumericalTarget(TARGET, minimize=True))
        self.campaign = Campaign(searchspace, objective)
        self.steps = steps

    def recommend(self, batch_size: int) -> List[List[float]]:
        frame = self.campaign.recommend(batch_size=batch_size)
        return [[float(row[d]) for d in DYES] for _, row in frame.iterrows()]

    def add(self, mixes: Sequence[Sequence[float]], distances: Sequence[float]) -> None:
        rows = [dict(zip(DYES, snap_to_grid(m, self.steps)), **{TARGET: d}) for m, d in zip(mixes, distances)]
        self.campaign.add_measurements(pd.DataFrame(rows))
