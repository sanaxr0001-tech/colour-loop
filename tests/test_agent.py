"""Claude-driven mode, without Claude: a stand-in ``claude`` executable drives the real stdio server."""

import asyncio
import json
import stat
import sys
import textwrap

from colour_loop.agent import ALLOWED_TOOLS, AgentConfig, agent_live_state, child_env, run_agent

FAKE_CLAUDE = """\
#!{python}
# Stand-in for the Claude CLI: answers `auth status`, and for `-p` starts the MCP server named in
# --mcp-config over stdio, makes a few tool calls (one of them refused) and prints stream-json.
import asyncio, json, os, sys
from mcp import Client
from mcp.client.stdio import StdioServerParameters

argv = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(json.dumps({{"argv": argv, "has_api_key": "ANTHROPIC_API_KEY" in os.environ, "cwd": os.getcwd(),
                        "cwd_files": os.listdir(".")}}) + "\\n")
if argv[:2] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": True}}))
    sys.exit(0)

config = json.loads(argv[argv.index("--mcp-config") + 1])
(name, server), = config["mcpServers"].items()

def say(obj):
    print(json.dumps(obj), flush=True)

async def main():
    say({{"type": "system", "subtype": "init", "model": "fake", "apiKeySource": "none", "permissionMode": "dontAsk",
         "tools": ["mcp__" + name + "__dispense_mix"], "mcp_servers": [{{"name": name, "status": "connected"}}]}})
    params = StdioServerParameters(command=server["command"], args=server["args"])
    async with Client(params) as client:
        say({{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "Trying two purples."}}]}}}})
        await client.call_tool("dispense_mix", {{"well": "A1", "fractions": {{"red": 0.48, "yellow": 0.12, "blue": 0.40}}}})
        await client.call_tool("dispense_mix", {{"well": "A2", "fractions": {{"red": 0.5, "blue": 0.5}}}})
        await client.call_tool("dispense_mix", {{"well": "A2", "fractions": {{"red": 0.5, "blue": 0.5}}}})
        await client.call_tool("read_plate", {{"wells": ["A1", "A2"]}})
    say({{"type": "result", "subtype": "success", "is_error": False, "num_turns": 5, "duration_ms": 1234,
         "result": "Done."}})

asyncio.run(main())
"""


def fake_claude(tmp_path):
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "claude"
    script.write_text(FAKE_CLAUDE.format(python=sys.executable, log=str(log)))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, log


def test_the_headless_call_allows_only_the_lab_tools(tmp_path):
    config = AgentConfig(record_path=tmp_path / "run.jsonl", target="#a84378", max_wells=12, max_rounds=3)
    args = config.claude_args()
    value = lambda flag: args[args.index(flag) + 1]
    assert args[1] == "-p"
    assert value("--tools") == ""  # no built-in tools
    assert "--strict-mcp-config" in args  # no MCP servers from any other config
    assert value("--setting-sources") == ""  # no settings, so no hooks
    assert value("--permission-mode") == "dontAsk"
    assert value("--allowedTools").split(",") == ALLOWED_TOOLS and len(ALLOWED_TOOLS) == 6
    assert value("--output-format") == "stream-json"
    servers = json.loads(value("--mcp-config"))["mcpServers"]
    assert list(servers) == ["colour-lab"]
    server = servers["colour-lab"]
    assert server["type"] == "stdio"
    assert server["command"].endswith("colour-loop-mcp") or server["args"][:2] == ["-m", "colour_loop.mcp_server"]
    assert server["args"][server["args"].index("--target") + 1] == "#a84378"
    assert server["args"][server["args"].index("--max-wells") + 1] == "12"


def test_the_api_key_variable_is_dropped(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-forwarded")
    assert "ANTHROPIC_API_KEY" not in child_env()


def test_agent_run_end_to_end_with_a_stand_in_claude(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-forwarded")
    script, log = fake_claude(tmp_path)
    config = AgentConfig(record_path=tmp_path / "run.jsonl", executable=str(script), time_budget=120)
    live = agent_live_state(config)
    result = asyncio.run(run_agent(config, live=live))

    assert result.ok, result.error
    s = result.summary
    assert s["wells"] == 2 and s["rounds"] == 1 and s["refusals"] == 1 and s["tool_calls"] == 4
    assert "already holds a mix" in s["refusal_reasons"][0]
    assert s["claude"]["subtype"] == "success"

    events = [json.loads(line) for line in config.record_path.read_text().splitlines()]
    claude = [e for e in events if e["type"] == "claude"]
    assert [e["event"] for e in claude] == ["init", "text", "result"]
    assert claude[0]["api_key_source"] == "none"

    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(i["has_api_key"] for i in invocations)
    run_call = invocations[-1]
    assert run_call["argv"][0] == "-p" and run_call["cwd_files"] == []  # an empty working directory

    state = live.snapshot()
    assert [c["tool"] for c in state["feed"]] == ["dispense_mix"] * 3 + ["read_plate"]
    assert [c["ok"] for c in state["feed"]] == [True, True, False, True]
    assert state["feed"][2]["reason"].startswith("well A2 already holds a mix")
    assert {w["well"] for w in state["wells"]} == {"A1", "A2"} and all(w["hex"] for w in state["wells"])
    assert state["notes"][0]["note"] == "Trying two purples."
    assert state["finished"]


def test_time_budget_stops_claude(tmp_path):
    script = tmp_path / "claude"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, sys, time
        if sys.argv[1:3] == ["auth", "status"]:
            print(json.dumps({{"loggedIn": True}}))
        else:
            time.sleep(60)
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    config = AgentConfig(record_path=tmp_path / "run.jsonl", executable=str(script), time_budget=1)
    result = asyncio.run(run_agent(config))
    assert not result.ok and "time budget" in result.error
    assert result.seconds < 30
