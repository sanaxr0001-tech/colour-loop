"""The simulated plate reader: dye volumes in, a measured RGB colour out.

Everything physical about the demo lives here, in one small module. The model is Beer-Lambert:

    A_c = path_length * sum_d( epsilon[d, c] * conc_d )      absorbance in colour channel c
    T_c = 10 ** (-A_c)                                      fraction of that channel's light transmitted
    reading_c = 255 * T_c + noise                           what the reader reports

``conc_d`` is the concentration of dye ``d`` in the well, which is its stock concentration times its
volume fraction of the well. Because stock concentration and path length are fixed, they are folded
into one number per dye and channel: ``ABSORBANCE_AT_FULL_STRENGTH[d][c]`` is the absorbance a well
would have if it were 100 % dye ``d`` (an optical-density-per-channel figure, dimensionless).

The coefficients are invented but plausible: each dye is a subtractive primary that mostly absorbs
the colours it is not, and each also leaks a little into a second channel so mixes are not perfectly
separable (the optimiser has something to learn). They are NOT measurements of real dyes. They are
tuned so the default demo target (#7a4b9c, a purple) is reachable with a mix of all three dyes.
Very light or very dark targets are not reachable, because the dye fractions always sum to 1.

    red    absorbs green (strongly) and blue (partly)  -> looks red
    yellow absorbs blue (strongly) and a little green  -> looks yellow
    blue   absorbs red (strongly) and a little green   -> looks blue
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

DYES: tuple[str, ...] = ("red", "yellow", "blue")
CHANNELS: tuple[str, ...] = ("R", "G", "B")

# Absorbance of a well that is 100 % of one dye, per colour channel (rows: dyes, cols: R, G, B).
ABSORBANCE_AT_FULL_STRENGTH = np.array(
    [
        [0.10, 0.82, 0.26],  # red (a pinkish red)
        [0.02, 0.13, 0.77],  # yellow
        [0.67, 0.28, 0.03],  # blue
    ]
)

# Incident light is white: every channel starts at full scale.
FULL_SCALE = 255.0

DEFAULT_NOISE_SD = 1.5  # standard deviation of the reader noise, in 0-255 counts


@dataclass
class PlateReader:
    """Simulated absorbance plate reader with seeded Gaussian measurement noise."""

    noise_sd: float = DEFAULT_NOISE_SD
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def read(self, volumes_ul: Mapping[str, float]) -> tuple[int, int, int]:
        """Measure one well given the volume (in microlitres) of each dye in it."""
        clean = ideal_rgb(volumes_ul)
        noisy = clean + self._rng.normal(0.0, self.noise_sd, size=3)
        r, g, b = (int(v) for v in np.clip(np.rint(noisy), 0, 255))
        return r, g, b


def ideal_rgb(volumes_ul: Mapping[str, float]) -> np.ndarray:
    """Noise-free RGB (floats, 0-255) of a well, from Beer-Lambert.

    Only the *fractions* of the dyes matter (the well is always topped up to the same total
    volume), so a well holding no dye at all is simply white.
    """
    vols = np.array([float(volumes_ul.get(d, 0.0)) for d in DYES])
    if (vols < 0).any():
        raise ValueError("dye volumes must be non-negative")
    total = vols.sum()
    if total == 0:
        return np.full(3, FULL_SCALE)
    fractions = vols / total
    absorbance = fractions @ ABSORBANCE_AT_FULL_STRENGTH
    return FULL_SCALE * 10.0 ** (-absorbance)


def ideal_rgb_from_fractions(fractions: Sequence[float]) -> np.ndarray:
    """Noise-free RGB for dye fractions given in ``DYES`` order."""
    return ideal_rgb(dict(zip(DYES, fractions)))
