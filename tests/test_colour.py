import pytest

from colour_loop.colour import colour_distance, delta_e_2000, hex_to_rgb, rgb_to_hex, rgb_to_lab

# Reference pairs from Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
# Implementation Notes, Supplementary Test Data, and Mathematical Observations", Table 1.
SHARMA_PAIRS = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1, lab2, expected", SHARMA_PAIRS)
def test_ciede2000_matches_published_reference(lab1, lab2, expected):
    assert delta_e_2000(lab1, lab2) == pytest.approx(expected, abs=1e-4)
    assert delta_e_2000(lab2, lab1) == pytest.approx(expected, abs=1e-4)


def test_identical_colours_have_zero_distance():
    assert colour_distance((122, 75, 156), (122, 75, 156)) == 0


def test_white_and_black_in_lab():
    assert rgb_to_lab((255, 255, 255)) == pytest.approx((100, 0, 0), abs=1e-3)
    assert rgb_to_lab((0, 0, 0)) == pytest.approx((0, 0, 0), abs=1e-6)


def test_hex_round_trip():
    assert hex_to_rgb("#7a4b9c") == (122, 75, 156)
    assert hex_to_rgb("7A4B9C") == (122, 75, 156)
    assert rgb_to_hex((122, 75, 156)) == "#7a4b9c"
    assert rgb_to_hex((300, -4, 12.6)) == "#ff000d"


@pytest.mark.parametrize("bad", ["#7a4b9", "purple", "#gggggg", ""])
def test_bad_hex_is_rejected(bad):
    with pytest.raises(ValueError):
        hex_to_rgb(bad)
