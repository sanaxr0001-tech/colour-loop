"""BayBE proposes the next mixes.

The search space is every 3-dye mix on a 2 % grid of the simplex (red + yellow + blue = 1), 1326
candidates. A discrete grid keeps BayBE fast and makes every proposal directly pipettable; at a
180 uL well a 2 % step is 3.6 uL. The objective is the CIEDE2000 distance to the target, minimised.

With ``min_fraction`` set, the grid keeps only mixes whose every dye is either absent or at least
that fraction of the well, so that each dye volume is inside the pipette's range (the MCP server
and the comparison use this: 20 uL of a 180 uL well leaves 681 of the 1326 mixes).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
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


def simplex_grid(steps: int = GRID_STEPS, min_fraction: float = 0.0) -> pd.DataFrame:
    rows = [
        (i / steps, j / steps, (steps - i - j) / steps)
        for i in range(steps + 1)
        for j in range(steps + 1 - i)
    ]
    if min_fraction:
        rows = [r for r in rows if all(f == 0 or f >= min_fraction - 1e-9 for f in r)]
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

    def __init__(self, seed: int | None = 0, steps: int = GRID_STEPS, min_fraction: float = 0.0):
        if seed is not None:
            Settings(random_seed=seed).activate()
        grid = simplex_grid(steps, min_fraction)
        self._grid = grid.to_numpy()
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

    def nearest_candidate(self, mix: Sequence[float]) -> List[float]:
        """The search-space mix closest to ``mix`` (itself, if it is one)."""
        index = int(np.argmin(((self._grid - np.asarray(mix, dtype=float)) ** 2).sum(axis=1)))
        return [float(f) for f in self._grid[index]]

    def add(self, mixes: Sequence[Sequence[float]], distances: Sequence[float]) -> None:
        # A measured mix may be off the grid (an agent can choose any fractions); BayBE learns it as
        # the nearest mix in its search space.
        rows = [dict(zip(DYES, self.nearest_candidate(snap_to_grid(m, self.steps))), **{TARGET: d})
                for m, d in zip(mixes, distances)]
        self.campaign.add_measurements(pd.DataFrame(rows))
