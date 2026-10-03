import numpy as np
import pytest

from colour_loop.colour import colour_distance, hex_to_rgb
from colour_loop.physics import ABSORBANCE_AT_FULL_STRENGTH, DYES, PlateReader, ideal_rgb, ideal_rgb_from_fractions


def test_no_dye_is_white():
    assert ideal_rgb({}) == pytest.approx([255, 255, 255])


def test_beer_lambert_absorbance_is_the_fraction_weighted_sum():
    fractions = np.array([0.5, 0.2, 0.3])
    rgb = ideal_rgb_from_fractions(fractions)
    absorbance = -np.log10(rgb / 255)
    assert absorbance == pytest.approx(fractions @ ABSORBANCE_AT_FULL_STRENGTH)


def test_only_fractions_matter_not_total_volume():
    assert ideal_rgb({"red": 90, "blue": 90}) == pytest.approx(ideal_rgb({"red": 10, "blue": 10}))


def test_each_pure_dye_looks_like_its_name():
    r, g, b = ideal_rgb({"red": 180})
    assert r > g and r > b
    r, g, b = ideal_rgb({"yellow": 180})
    assert r > b and g > b  # red + green light = yellow
    r, g, b = ideal_rgb({"blue": 180})
    assert b > r and b > g


def test_more_blue_dye_removes_more_red_light():
    reds = [ideal_rgb({"blue": f, "yellow": 1 - f})[0] for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a > b for a, b in zip(reds, reds[1:]))


def test_negative_volume_is_rejected():
    with pytest.raises(ValueError):
        ideal_rgb({"red": -1, "blue": 10})


def test_reader_noise_is_seeded_and_small():
    volumes = {"red": 90, "yellow": 18, "blue": 72}
    a = [PlateReader(seed=7).read(volumes) for _ in range(3)]
    b = [PlateReader(seed=7).read(volumes) for _ in range(3)]
    assert a == b
    reader = PlateReader(noise_sd=1.5, seed=1)
    clean = ideal_rgb(volumes)
    readings = np.array([reader.read(volumes) for _ in range(500)])
    assert np.abs(readings.mean(axis=0) - clean).max() < 0.5
    assert 1.0 < readings.std(axis=0).mean() < 2.0


def test_noise_free_reader_returns_rounded_ideal():
    volumes = {"red": 30, "yellow": 60, "blue": 90}
    assert PlateReader(noise_sd=0).read(volumes) == tuple(int(round(v)) for v in ideal_rgb(volumes))


def test_default_target_is_reachable():
    # the README's demo target must be inside the dyes' gamut, or the demo could never converge
    target = hex_to_rgb("#7a4b9c")
    steps = 50
    best = min(
        colour_distance(ideal_rgb_from_fractions((i / steps, j / steps, (steps - i - j) / steps)), target)
        for i in range(steps + 1)
        for j in range(steps + 1 - i)
    )
    assert best < 1.0
    assert len(DYES) == 3
