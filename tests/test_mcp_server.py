"""The MCP tools, called through the SDK's in-process client, and one test per guardrail."""

import asyncio
import json

import pytest
from mcp import Client

from colour_loop.mcp_server import MIN_FRACTION, LabConfig, LabSession, build_server

TOOLS = {"get_deck_state", "list_dyes", "dispense_mix", "read_plate", "suggest_next_mixes", "get_run_record"}
PURPLE = {"red": 0.48, "yellow": 0.12, "blue": 0.40}


def make_lab(tmp_path, **limits) -> LabSession:
    return LabSession(LabConfig(record_path=tmp_path / "run.jsonl", **limits))


def calls(lab: LabSession, *steps):
    """Run (tool, arguments) steps through an MCP client connected to the server in-process."""

    async def go():
        async with Client(build_server(lab)) as client:
            return [await client.call_tool(tool, args) for tool, args in steps]

    return asyncio.run(go())


def record(lab: LabSession, kind=None):
    events = [json.loads(line) for line in lab.config.record_path.read_text().splitlines()]
    return [e for e in events if kind is None or e["type"] == kind]


def assert_refused(lab, result, *words):
    """The call came back as a tool error saying why, and the refusal is in the run record."""
    assert result.is_error
    text = result.content[0].text
    assert "Refused" in text
    for word in words:
        assert word in text
    last = record(lab, "call")[-1]
    assert last["ok"] is False and last["refused"] is True and last["reason"]
    assert lab.refusals and lab.refusals[-1]["reason"] == last["reason"]


def test_lists_exactly_the_six_tools(tmp_path):
    async def go():
        async with Client(build_server(make_lab(tmp_path))) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert asyncio.run(go()) == TOOLS


def test_dispense_read_and_record(tmp_path):
    lab = make_lab(tmp_path)
    deck, dyes, dispensed, read, again, run = calls(
        lab,
        ("get_deck_state", {}),
        ("list_dyes", {}),
        ("dispense_mix", {"well": "a1", "fractions": PURPLE}),
        ("read_plate", {"wells": ["A1"]}),
        ("read_plate", {"wells": ["A1"]}),
        ("get_run_record", {}),
    )
    assert not any(r.is_error for r in (deck, dyes, dispensed, read, again, run))
    assert deck.structured_content["target"] == "#7a4b9c"
    assert deck.structured_content["plate"]["empty_wells"] == 96
    assert [d["name"] for d in dyes.structured_content["dyes"]] == ["red", "yellow", "blue"]

    d = dispensed.structured_content
    assert d["well"] == "A1" and d["dispensed_ul"] == {"red": 86.4, "yellow": 21.6, "blue": 72.0}
    assert lab.robot.well_volume_ul("A1") == pytest.approx(180.0)  # PyLabRobot's tracker agrees

    reading = read.structured_content["readings"][0]
    assert reading["new"] and reading["round"] == 1 and reading["hex"].startswith("#")
    assert 0 <= reading["delta_e"] < 10
    # a well is measured once; reading it again returns the same measurement and is not a round
    assert again.structured_content["readings"][0]["rgb"] == reading["rgb"]
    assert again.structured_content["round"] is None

    r = run.structured_content
    assert r["budget"]["wells_used"] == 1 and r["budget"]["rounds_done"] == 1 and r["refusals"] == []

    kinds = [e["type"] for e in record(lab)]
    assert kinds[0] == "session" and kinds.count("round") == 1 and kinds.count("call") == 6
    assert record(lab, "round")[0]["wells"][0]["well"] == "A1"


def test_suggestions_come_from_the_readings_and_are_pipettable(tmp_path):
    lab = make_lab(tmp_path)
    *_, suggest = calls(
        lab,
        ("dispense_mix", {"well": "A1", "fractions": PURPLE}),
        ("dispense_mix", {"well": "A2", "fractions": {"red": 0.5, "blue": 0.5}}),
        ("read_plate", {"wells": ["A1", "A2"]}),
        ("suggest_next_mixes", {"n": 3}),
    )
    s = suggest.structured_content
    assert s["based_on_readings"] == 2 and len(s["suggestions"]) == 3
    for mix in s["suggestions"]:
        assert sum(mix.values()) == pytest.approx(1.0)
        assert all(f == 0 or f >= MIN_FRACTION for f in mix.values())


def test_a_dispensed_suggestion_is_marked_as_one(tmp_path):
    lab = make_lab(tmp_path)
    (suggest,) = calls(lab, ("suggest_next_mixes", {"n": 1}))
    mix = suggest.structured_content["suggestions"][0]
    calls(lab, ("dispense_mix", {"well": "C3", "fractions": mix}))
    assert lab.wells["C3"]["decided_by"] == "suggestion"


# --- one test per guardrail ----------------------------------------------------------------------


def test_refuses_fractions_outside_0_to_1(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": {"red": 1.2, "yellow": -0.2, "blue": 0.0}}))
    assert_refused(lab, r, "outside 0-1")
    assert lab.wells == {}


def test_refuses_fractions_not_summing_to_1(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": {"red": 0.5, "yellow": 0.3, "blue": 0.1}}))
    assert_refused(lab, r, "sum to 0.900")
    # within the 0.01 tolerance is accepted (and normalised)
    (ok,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": {"red": 0.5, "yellow": 0.3, "blue": 0.205}}))
    assert not ok.is_error


def test_refuses_a_dye_volume_below_the_pipette_range(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": {"red": 0.9, "yellow": 0.1}}))
    assert_refused(lab, r, "yellow would be 18 uL", "20-300 uL")
    assert lab.wells == {}


def test_refuses_a_dye_volume_above_the_pipette_range(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": {"red": 1.0}, "total_volume_ul": 350}))
    assert_refused(lab, r, "red would be 350 uL")


def test_refuses_overfilling_a_well(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": {"red": 0.5, "yellow": 0.3, "blue": 0.2},
                                        "total_volume_ul": 250}))
    assert_refused(lab, r, "overfill well A1", "200 uL")
    assert lab.robot.well_volume_ul("A1") == 0


def test_refuses_writing_to_an_already_filled_well(tmp_path):
    lab = make_lab(tmp_path)
    first, second = calls(lab, ("dispense_mix", {"well": "B2", "fractions": PURPLE}),
                          ("dispense_mix", {"well": "B2", "fractions": {"red": 0.5, "blue": 0.5}}))
    assert not first.is_error
    assert_refused(lab, second, "already holds a mix")
    assert lab.robot.well_volume_ul("B2") == pytest.approx(180.0)  # not topped up


@pytest.mark.parametrize("well", ["Z9", "A13", "A0", "", "A1; B1"])
def test_refuses_unknown_wells(tmp_path, well):
    lab = make_lab(tmp_path)
    dispensed, read = calls(lab, ("dispense_mix", {"well": well, "fractions": PURPLE}),
                            ("read_plate", {"wells": [well]}))
    assert_refused(lab, dispensed, "unknown well")
    assert_refused(lab, read, "unknown well")


def test_refuses_more_wells_than_the_run_allows(tmp_path):
    lab = make_lab(tmp_path, max_wells=2)
    a, b, c = calls(lab, *[("dispense_mix", {"well": w, "fractions": PURPLE}) for w in ("A1", "A2", "A3")])
    assert not a.is_error and not b.is_error
    assert_refused(lab, c, "well limit", "2 wells")
    assert set(lab.wells) == {"A1", "A2"}


def test_refuses_more_rounds_than_the_run_allows(tmp_path):
    lab = make_lab(tmp_path, max_rounds=1)
    *_, read_b, dispense_c = calls(
        lab,
        ("dispense_mix", {"well": "A1", "fractions": PURPLE}),
        ("dispense_mix", {"well": "B1", "fractions": {"red": 0.5, "blue": 0.5}}),
        ("read_plate", {"wells": ["A1"]}),  # round 1 of 1
        ("read_plate", {"wells": ["B1"]}),  # would be round 2
        ("dispense_mix", {"well": "C1", "fractions": PURPLE}),
    )
    assert_refused(lab, dispense_c, "round limit")
    assert read_b.is_error and "round limit" in read_b.content[0].text
    assert len(lab.rounds) == 1 and "C1" not in lab.wells


def test_refuses_reading_an_empty_well(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("read_plate", {"wells": ["H12"]}))
    assert_refused(lab, r, "empty")


def test_arguments_of_the_wrong_type_are_refused_and_logged(tmp_path):
    lab = make_lab(tmp_path)
    (r,) = calls(lab, ("dispense_mix", {"well": "A1", "fractions": "lots of red"}))
    assert r.is_error
    last = record(lab, "call")[-1]
    assert last["ok"] is False and last["refused"] and last["reason"].startswith("invalid arguments")


def test_every_refusal_is_in_the_run_record(tmp_path):
    lab = make_lab(tmp_path)
    calls(lab,
          ("dispense_mix", {"well": "A1", "fractions": {"red": 2.0}}),
          ("dispense_mix", {"well": "Q1", "fractions": PURPLE}),
          ("dispense_mix", {"well": "A1", "fractions": PURPLE}),
          ("read_plate", {"wells": ["A2"]}))
    refused = [e for e in record(lab, "call") if not e["ok"]]
    assert len(refused) == 3 == len(lab.refusals)
    (run,) = calls(lab, ("get_run_record", {}))
    assert len(run.structured_content["refusals"]) == 3


def test_fractions_may_also_be_a_list_in_dye_order(tmp_path):
    lab = make_lab(tmp_path)
    out = asyncio.run(lab.dispense_mix("D4", [0.48, 0.12, 0.40]))
    assert out["dispensed_ul"] == {"red": 86.4, "yellow": 21.6, "blue": 72.0}
    with pytest.raises(Exception, match="3 fractions"):
        asyncio.run(lab.dispense_mix("D5", [0.5, 0.5]))
