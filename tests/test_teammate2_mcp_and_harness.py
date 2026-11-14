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
    GeminiLiveError,
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


def test_gemini_blueprint_decisions_and_episode(monkeypatch):
    ep = Episode("first-night", 7, redact_seed=True)
    mat = synthesize_blueprint_matrix("shelter")
    ep.step("chop_trees", radius=40, max_trees=4)
    ep.step("navigate_to", x=ep.world.home[0], y=ep.world.home[1], budget=300)
    # matrix tiles are paid from inventory: wood alone no longer makes a house
    probe = Episode("first-night", 7, redact_seed=True)
    probe.step("chop_trees", radius=40, max_trees=4)
    assert probe.step("build_blueprint", matrix=mat, origin_dx=-4, origin_dy=-4)["housing"]["valid"] is False
    ep.step("craft_item", item_id="workbench")
    ep.step("place_station_or_furniture", item_id="workbench", dx=-2, dy=0)
    ep.step("craft_item", item_id="door")
    res = ep.step("build_blueprint", matrix=mat, origin_dx=-4, origin_dy=-4)
    assert res["housing"]["valid"] is True and res["tiles_skipped"] == 0

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
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
    assert "SHELTER" in out["ledger"]["milestones"]


class _FakeDecisions:
    """Stands in for ``client.decisions`` and records what the wrapper sends."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, *, model, input, questions):
        self.requests.append({"model": model, "input": input, "questions": questions})
        pick = self.script.pop(0) if self.script else "end_window"
        opts = list(questions["next_action"]["criteria"])
        probs = {o: (0.9 if o == pick else 0.1 / max(1, len(opts) - 1)) for o in opts}

        class _Ans:
            choice = pick
            confidence = 0.9
            probabilities = probs

        class _Resp:
            answers = {"next_action": _Ans()}

        return _Resp()


class _FakeClient:
    def __init__(self, script):
        self.decisions = _FakeDecisions(script)


_PLAN = json.dumps({
    "strategy": "Chop wood, then rest.",
    "candidates": [
        {"id": "chop", "tool": "chop_trees", "args": {"radius": "40", "max_trees": 2, "bogus": 1},
         "when": "wood is low"},
        {"id": "status", "tool": "terraria_status", "args": {}, "when": "state unclear"},
        {"id": "nope", "tool": "not_a_tool", "args": {}, "when": "never"},
    ],
})


def test_gemini_two_tier_supervisor_plans_decision_model_acts():
    prompts = []

    def supervisor(prompt_text, png_b64):
        prompts.append(prompt_text)
        return "```json\n" + _PLAN + "\n```"

    client = _FakeClient(["replan", "chop", "not_on_menu", "chop"])
    agent = GeminiAgent(live=True, supervisor_fn=supervisor,
                        decisions=GeminiDecisionsWrapper(client=client, strict=True))
    seen = []
    out = run_agent_episode(agent, "first-night", 7, max_turns=60,
                            on_turn=lambda turn, tool, args, res: seen.append((tool, args)))

    windows = TASKS["first-night"].windows
    assert out["ledger"]["windows"][-1]["w"] == windows
    # window 1: plan, "replan" -> new plan, chop, off-menu answer -> new plan, chop, end_window
    assert seen[:3] == [("chop_trees", {"radius": 40, "max_trees": 2})] * 2 + [("end_window", {})]
    assert agent.stats["replans"] == 2
    assert agent.stats["supervisor_calls"] == windows + 2
    assert agent.stats["decision_calls"] == len(client.decisions.requests) == 5 + (windows - 1)
    assert "chop_trees(" in prompts[0] and "Window 1 is starting" in prompts[0]

    req = client.decisions.requests[0]
    assert req["model"] == "gemini-decision-1"
    assert req["input"][0]["type"] == "text" and "Chop wood" in req["input"][0]["text"]
    q = req["questions"]["next_action"]
    assert q["type"] == "choice" and q["instructions"]
    assert list(q["criteria"]) == ["chop", "status", "end_window", "replan"]
    assert [t["tier"] for t in agent.trace[:3]] == ["supervisor", "supervisor", "decision"]


def test_gemini_live_mode_never_falls_back_to_scripted_policy(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ep = Episode("first-night", 7, redact_seed=True)
    with pytest.raises(GeminiLiveError, match="GEMINI_API_KEY"):
        GeminiAgent(live=True).next_action({"ok": True}, [], ep)
    assert ep.world.window == 1 and not ep.world.milestones

    # a supervisor that never returns a usable plan is an error, not a silent end_window
    bad = GeminiAgent(live=True, supervisor_fn=lambda text, png: "no json here",
                      decisions=GeminiDecisionsWrapper(client=_FakeClient([]), strict=True))
    with pytest.raises(GeminiLiveError, match="no usable plan"):
        bad.next_action({"ok": True}, [], ep)

    class _Boom:
        class decisions:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("400 bad question")

    strict = GeminiDecisionsWrapper(client=_Boom(), strict=True)
    with pytest.raises(GeminiLiveError, match="400 bad question"):
        strict.decide({"q": {"type": "bool", "instructions": "Is it night?"}}, "state")
    lenient = GeminiDecisionsWrapper(client=_Boom())
    assert lenient.decide({"q": {"type": "bool", "instructions": "Is it night?"}}, "state")["source"] == "deterministic_fallback"
    assert "400 bad question" in lenient.last_error


def test_runner_reports_bad_tool_calls_instead_of_crashing():
    class Sloppy:
        name = "sloppy"
        calls = 0

        def next_action(self, observation, context, episode):
            self.calls += 1
            if self.calls == 1:
                return {"tool": "chop_trees", "args": {"no_such_arg": 1}}
            if self.calls == 2:
                self.saw = observation
            return {"tool": "end_window", "args": {}}

    agent = Sloppy()
    out = run_agent_episode(agent, "first-night", 7, max_turns=3)
    assert out["turns"] == 3
    assert agent.saw["ok"] is False and "invalid call chop_trees" in agent.saw["error"]


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
