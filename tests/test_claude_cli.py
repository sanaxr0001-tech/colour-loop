"""The headless call, against a stand-in ``claude`` executable (no real Claude is ever run here)."""

import json
import stat
import sys
import textwrap

import pytest

from colour_loop.claude_layer import ClaudeAdvisor

PROPOSALS = [[0.5, 0.1, 0.4], [0.2, 0.2, 0.6]]


def fake_claude(tmp_path, structured, logged_in=True):
    """Write a fake `claude` that answers `auth status` and `-p`, and logs its argv and env."""
    (tmp_path / "answer.json").write_text(json.dumps(structured))
    script = tmp_path / "claude"
    log = tmp_path / "calls.jsonl"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys
        with open({str(log)!r}, "a") as f:
            f.write(json.dumps({{"argv": sys.argv[1:], "has_api_key": "ANTHROPIC_API_KEY" in os.environ}}) + "\\n")
        if sys.argv[1:3] == ["auth", "status"]:
            print(json.dumps({{"loggedIn": {logged_in!r}}}))
        else:
            answer = json.load(open({str(tmp_path / "answer.json")!r}))
            print(json.dumps({{"is_error": False, "structured_output": answer}}))
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, log


def test_advice_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-forwarded")
    script, log = fake_claude(tmp_path, {"note": "Swap 1.", "replace": {"index": 1, "red": 0.4, "yellow": 0.0, "blue": 0.6}})
    advisor = ClaudeAdvisor(executable=str(script))
    assert advisor.check()
    advice = advisor.advise("#7a4b9c", [], PROPOSALS, None)
    assert advice.error is None
    assert advice.note == "Swap 1."
    assert advice.override.index == 1 and advice.override.fractions == pytest.approx([0.4, 0.0, 0.6])
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(c["has_api_key"] for c in calls)
    argv = calls[-1]["argv"]
    assert argv[0] == "-p" and "--json-schema" in argv and argv[argv.index("--model") + 1] == "sonnet"


def test_invalid_override_from_claude_is_ignored(tmp_path):
    script, _ = fake_claude(tmp_path, {"note": "Go big.", "replace": {"index": 0, "red": 0.9, "yellow": 0.9, "blue": 0.9}})
    advice = ClaudeAdvisor(executable=str(script)).advise("#7a4b9c", [], PROPOSALS, None)
    assert advice.note == "Go big."
    assert advice.override is None and "sum" in advice.rejected


def test_not_logged_in_means_unavailable(tmp_path):
    script, _ = fake_claude(tmp_path, {"note": "x"}, logged_in=False)
    advisor = ClaudeAdvisor(executable=str(script))
    assert not advisor.check()
    assert "not logged in" in advisor.unavailable_reason


def test_missing_cli_means_unavailable():
    advisor = ClaudeAdvisor(executable="definitely-not-a-real-claude-binary")
    assert not advisor.check()
    assert "not on PATH" in advisor.unavailable_reason


def test_timeout_is_reported_not_raised(tmp_path):
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\nimport time; time.sleep(5)\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    advice = ClaudeAdvisor(executable=str(script), timeout=0.5).advise("#7a4b9c", [], PROPOSALS, None)
    assert advice.error and "timed out" in advice.error
