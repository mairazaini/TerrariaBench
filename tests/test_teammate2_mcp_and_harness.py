"""Tests for Teammate 2 deliverables: 24-tool MCP server, harness runner,
65k-token context compaction, Gemini agent/Decisions API, and Nemotron agent."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from terraria_bench.grader import verify_seal
from terraria_bench.harness import (
    CompactHistory,
    GeminiAgent,
    GeminiDecisionsWrapper,
    NemotronAgent,
    build_multimodal_messages,
    call_with_backoff,
    run_agent_episode,
    run_eval_standard,
    synthesize_blueprint_matrix,
)
from terraria_bench.mcp_server import MCPToolServer
from terraria_bench.tasks import TASKS
from terraria_bench.tools import TOOL_NAMES, Episode


def test_mcp_server_jsonrpc_and_stdio():
    srv = MCPToolServer("first-night", 7, redact_seed=True)
    init_res = srv.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_res is not None
    assert "World: locked" in init_res["result"]["header"]

    tools_res = srv.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = tools_res["result"]["tools"]
    assert len(tools) == 24
    assert [t["name"] for t in tools] == list(TOOL_NAMES)

    ops_before = srv.ep.ops_left
    status_res = srv.handle_jsonrpc({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "terraria_status", "arguments": {}}
    })
    assert status_res["result"]["isError"] is False
    assert srv.ep.ops_left == ops_before  # free tool

    vp_res = srv.handle_jsonrpc({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "observe_viewport", "arguments": {"include_png": True}},
    })
    content = vp_res["result"]["content"]
    assert any(part["type"] == "image" and part["mimeType"] == "image/png" for part in content)
    assert srv.ep.ops_left == ops_before - 1

    # Verify stdio transport end-to-end
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 10, "method": "ping"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 11, "method": "episode/finalize", "params": {}}) + "\n"
    )
    stdout = io.StringIO()
    srv.serve_stdio(stdin, stdout)
    lines = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["result"] == {}
    assert verify_seal(lines[1]["result"])


def test_compact_history_and_early_quitter_runner(tmp_path: Path):
    ep = Episode("first-night", 7, redact_seed=True)
    hist = CompactHistory(threshold_tokens=80, keep_last=3)
    for i in range(6):
        hist.append({"turn": i + 1, "tool": "terraria_status", "payload": "x" * 200}, ep)
    ctx = hist.render_context()
    assert hist.compactions >= 1
    assert ctx[0]["type"] == "compacted_summary"
    assert [t["turn"] for t in ctx[1:]] == [4, 5, 6]

    class QuitterAgent:
        name = "quitter"

        def next_action(self, observation, context, episode):
            return None  # quit immediately

    out = run_agent_episode(QuitterAgent(), "first-night", 7, max_turns=150, save_dir=tmp_path)
    assert out["turns"] == 0
    assert out["score"]["windows"] == 0
    assert out["score"]["continuity"] == 0.0
    assert (tmp_path / "first-night_seed7.json").exists()

    class InfiniteProbeAgent:
        name = "probe"

        def next_action(self, observation, context, episode):
            return {"tool": "terraria_status", "args": {}}

    capped = run_agent_episode(InfiniteProbeAgent(), "first-night", 7, max_turns=5)
    assert capped["turns"] == 5


def test_gemini_blueprint_decisions_and_episode():
    ep = Episode("first-night", 7, redact_seed=True)
    ep.step("chop_trees", radius=40, max_trees=4)
    mat = synthesize_blueprint_matrix("shelter")
    res = ep.step("build_blueprint", matrix=mat, origin_dx=-4, origin_dy=-4)
    assert res["housing"]["valid"] is True

    decisions = GeminiDecisionsWrapper()
    dec = decisions.decide(
        {"next_step": {"type": "choice", "choices": ["build_shelter", "mine_ore"]}},
        "window=1 phase=Day unsheltered",
    )
    ans = dec["answers"]["next_step"]
    assert ans["value"] == "build_shelter"
    assert pytest.approx(sum(ans["probabilities"].values()), abs=1e-2) == 1.0
    assert 0.0 <= ans["confidence"] <= 1.0

    agent = GeminiAgent(live=False)
    out = run_agent_episode(agent, "first-night", 7, max_turns=150)
    assert out["ledger"]["windows"][-1]["w"] == TASKS["first-night"].windows
    assert out["score"]["sla"] > 0.5


def test_nemotron_backoff_multimodal_and_eval_standard():
    attempts = {"n": 0}
    delays: list[float] = []

    class RateLimitError(RuntimeError):
        status_code = 429

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429 Too Many Requests")
        return "ok"

    assert call_with_backoff(flaky, max_retries=4, base_delay=1.5, sleep_fn=delays.append) == "ok"
    assert delays == [1.5, 3.0]

    ep = Episode("first-night", 7, redact_seed=True)
    obs = ep.step("observe_viewport", include_png=True)
    msgs, tools = build_multimodal_messages(ep.task.prompt, obs, [])
    assert len(tools) == 24
    assert any(
        isinstance(m.get("content"), list)
        and any(part.get("type") == "image_url" for part in m["content"])
        for m in msgs
    )

    nem_out = run_agent_episode(NemotronAgent(live=False), "first-night", 7, max_turns=150)
    assert nem_out["score"]["sla"] > 0.5

    report = run_eval_standard(live=False)
    rows = {r["name"]: r for r in report["table1"]}
    assert rows["do-nothing"]["composite"] == 0.0
    assert rows["absentee"]["composite"] == 0.1
    assert rows["exploiter"]["composite"] < 0.05
    assert rows["smart-triage"]["composite"] > rows["rule-based"]["composite"] > rows["blast-list"]["composite"]
