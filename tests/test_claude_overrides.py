import math

import pytest

from colour_loop.claude_layer import format_prompt, parse_response, validate_override

PROPOSALS = [[0.5, 0.1, 0.4], [0.2, 0.2, 0.6], [1.0, 0.0, 0.0], [0.3, 0.3, 0.4]]


def test_valid_override_is_accepted():
    o = validate_override({"index": 2, "red": 0.5, "yellow": 0.1, "blue": 0.4, "why": "closer"}, 4)
    assert o.index == 2
    assert o.fractions == pytest.approx([0.5, 0.1, 0.4])
    assert o.why == "closer"


def test_near_unit_sum_is_renormalised():
    o = validate_override({"index": 0, "red": 0.334, "yellow": 0.333, "blue": 0.338}, 4)
    assert sum(o.fractions) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "replace, reason",
    [
        ({"index": 4, "red": 0.5, "yellow": 0.1, "blue": 0.4}, "not one of"),
        ({"index": -1, "red": 0.5, "yellow": 0.1, "blue": 0.4}, "not one of"),
        ({"index": True, "red": 0.5, "yellow": 0.1, "blue": 0.4}, "integer"),
        ({"index": "1", "red": 0.5, "yellow": 0.1, "blue": 0.4}, "integer"),
        ({"index": 0, "red": 1.2, "yellow": -0.1, "blue": -0.1}, "outside 0-1"),
        ({"index": 0, "red": -0.2, "yellow": 0.6, "blue": 0.6}, "outside 0-1"),
        ({"index": 0, "red": 0.5, "yellow": 0.5, "blue": 0.5}, "sum"),
        ({"index": 0, "red": 0.1, "yellow": 0.1, "blue": 0.1}, "sum"),
        ({"index": 0, "red": math.nan, "yellow": 0.5, "blue": 0.5}, "outside 0-1"),
        ({"index": 0, "red": "0.5", "yellow": 0.1, "blue": 0.4}, "number"),
        ({"index": 0, "red": 0.5, "blue": 0.5}, "number"),
        ("swap 0 for more blue", "object"),
    ],
)
def test_unsafe_override_is_rejected(replace, reason):
    with pytest.raises(ValueError, match=reason):
        validate_override(replace, len(PROPOSALS))


def test_parse_response_keeps_note_and_drops_bad_override():
    advice = parse_response(
        {"note": "Too red.", "replace": {"index": 9, "red": 1, "yellow": 0, "blue": 0}}, len(PROPOSALS)
    )
    assert advice.note == "Too red."
    assert advice.override is None
    assert "not one of" in advice.rejected


def test_parse_response_without_override():
    advice = parse_response({"note": "Fine.", "replace": None}, len(PROPOSALS))
    assert advice.override is None and advice.rejected is None and advice.error is None


@pytest.mark.parametrize("payload", [None, "text", ["note"], 3])
def test_parse_response_survives_garbage(payload):
    advice = parse_response(payload, len(PROPOSALS))
    assert advice.error and advice.override is None


def test_prompt_lists_target_history_and_proposals():
    history = [{"round": 1, "well": "A1", "fractions": [0.3, 0.3, 0.4], "hex": "#806070", "delta_e": 12.3}]
    text = format_prompt("#7a4b9c", history, PROPOSALS, history[0])
    assert "#7a4b9c" in text and "A1" in text and "dE=12.30" in text
    assert "3: 0.30/0.30/0.40" in text
