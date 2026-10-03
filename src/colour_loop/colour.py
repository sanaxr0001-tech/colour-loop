"""Colour helpers: hex <-> RGB, sRGB -> CIELAB, and the CIEDE2000 colour difference.

CIEDE2000 (Sharma, Wu & Dalal 2005) is the objective the loop minimises: a delta-E of about 1 is
the smallest difference a person can see, and below about 2-3 two swatches look identical.
"""

from __future__ import annotations

import math
from typing import Sequence

# D65 white point, 2 degree observer
_WHITE = (0.95047, 1.0, 1.08883)

_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` (or ``rrggbb``) into an (r, g, b) tuple of 0-255 ints."""
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a 6-digit hex colour like #7a4b9c, got {value!r}")
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"not a hex colour: {value!r}") from exc


def rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _linearise(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: Sequence[float]) -> tuple[float, float, float]:
    """Convert sRGB (0-255) to CIELAB under D65."""
    lin = [_linearise(c) for c in rgb]
    xyz = [sum(m * v for m, v in zip(row, lin)) for row in _RGB_TO_XYZ]

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = (f(v / w) for v, w in zip(xyz, _WHITE))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e_2000(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIEDE2000 colour difference between two CIELAB colours."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - math.sqrt(c_bar7 / (c_bar7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0

    dl = l2 - l1
    dc = c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
    dh_big = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh) / 2)

    l_bar = (l1 + l2) / 2
    c_bar = (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        h_bar = (h1p + h2p + 360) / 2
    else:
        h_bar = (h1p + h2p - 360) / 2

    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    d_theta = 30 * math.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * math.sqrt(c_bar**7 / (c_bar**7 + 25**7))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar
    s_h = 1 + 0.015 * c_bar * t
    r_t = -math.sin(math.radians(2 * d_theta)) * r_c

    return math.sqrt(
        (dl / s_l) ** 2
        + (dc / s_c) ** 2
        + (dh_big / s_h) ** 2
        + r_t * (dc / s_c) * (dh_big / s_h)
    )


def colour_distance(rgb1: Sequence[float], rgb2: Sequence[float]) -> float:
    """CIEDE2000 distance between two sRGB (0-255) colours."""
    return delta_e_2000(rgb_to_lab(rgb1), rgb_to_lab(rgb2))
