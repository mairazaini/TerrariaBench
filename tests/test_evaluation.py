"""The evaluation path judges re-run: fair information, a model interface that can reach the reference
score, and baselines that behave (do nothing = 0, stopping early always costs)."""
from __future__ import annotations

import json

import pytest

from evaluation.judge import _prefix_score, diagnose
from evaluation.run_model import SYSTEM, _conversation, parse_reply, play
from terraria_bench.grader import grade
from terraria_bench.policies import run_policy_episode
from terraria_bench.provenance import load_calibration
from terraria_bench.tasks import EVAL_TASKS, REPORTED_SEEDS, SCORING, TASKS, describe_checkpoint
from terraria_bench.tools import TOOL_SCHEMAS


@pytest.fixture(scope="module")
def cal():
    return load_calibration()


def test_every_agent_is_told_the_scoring_rule_and_every_checkpoint():
    for task in TASKS.values():
        assert SCORING in task.prompt
        for window, kind in task.sla:
            assert f"- End of window {window}: {describe_checkpoint(kind)}" in task.prompt
    # the scoring text names the three weights the grader uses
    assert all(w in SCORING for w in ("70%", "20%", "10%"))


def test_model_prompt_lists_every_tool_with_its_parameters():
    for name, spec in TOOL_SCHEMAS.items():
        assert f"- {name}(" in SYSTEM
        for param in spec["params"]:
            assert param in SYSTEM


def test_parse_reply():
    assert parse_reply('{"why": "wood", "tool": "chop_trees", "args": {"radius": 40}}') == ("chop_trees", {"radius": 40}, "wood")
    assert parse_reply('thinking {"a": 1} then\n```json\n{"tool": "end_window"}\n```') == ("end_window", {}, "")
    assert parse_reply("no json here") is None
    assert parse_reply('{"tool": "x", "args": "oops"}') == ("x", {}, "")


@pytest.mark.parametrize("task_id", ["first-night", "eye-of-cthulhu"])
def test_the_model_interface_can_reach_the_reference_score(cal, task_id):
    """A 'model' that answers with the reference's calls, through the exact loop real models play in,
    gets the reference's score. So nothing in the interface (history compaction, result truncation,
    turn limits) stops an agent from playing a perfect game."""
    seed = 7
    ref = run_policy_episode("smart-triage", task_id, seed)
    actions = list(ref.actions_log)
    want = grade(ref.finalize(cal), cal["cells"][f"{task_id}:{seed}"])["composite"]
    it = iter(actions)

    def chat(messages):
        assert messages[0]["role"] == "system" and TASKS[task_id].prompt in messages[1]["content"]
        step = next(it)
        return json.dumps({"why": "reference", "tool": step["tool"], "args": step["args"]})

    out = play(chat, task_id, seed)
    assert out["stopped"] == "episode finished" and out["bad_replies"] == 0
    assert grade(out["ledger"], cal["cells"][f"{task_id}:{seed}"])["composite"] == want >= 0.9


def test_invalid_replies_cost_nothing_and_are_explained(cal):
    replies = iter(["garbage", '{"tool": "fly_to_moon"}', '{"tool": "chop_trees", "args": {"bogus": 1}}'] + ['{"tool": "end_window"}'] * 10)
    out = play(lambda m: next(replies), "first-night", 7)
    assert out["bad_replies"] == 3 and out["stopped"] == "episode finished"
    assert "not a valid call" in out["turns"][0]["result"]
    assert "rejected" in out["turns"][2]["result"]
    g = grade(out["ledger"], cal["cells"]["first-night:7"])
    assert g["composite"] == 0.0  # turning up and only ending windows is worth nothing


def test_conversation_keeps_recent_results_verbatim_and_summarises_old_ones():
    turns = [{"n": i, "window": 1, "tool": "chop_trees", "args": {}, "why": "w", "ok": True,
              "result": f"[World: locked | Win: 1/4] result number {i}"} for i in range(1, 31)]
    msgs = _conversation("first-night", turns, "[header]")
    text = "\n".join(m["content"] for m in msgs)
    assert "result number 30" in text and "result number 17" in text
    assert "1. w1 chop_trees({}) -> ok: result number 1" in text


@pytest.mark.parametrize("task_id", EVAL_TASKS)
def test_do_nothing_is_zero_and_stopping_early_always_costs(cal, task_id):
    for seed in REPORTED_SEEDS:
        cell = cal["cells"][f"{task_id}:{seed}"]
        ref = run_policy_episode("smart-triage", task_id, seed)
        actions = list(ref.actions_log)
        full = grade(ref.finalize(cal), cell)["composite"]
        assert _prefix_score(actions, task_id, seed, 0.0, cell) == 0.0
        for frac in (0.25, 0.5, 0.75):
            assert _prefix_score(actions, task_id, seed, frac, cell) < full, (task_id, seed, frac)


def test_diagnose_reads_unused_budget_and_missed_checkpoints(cal):
    out = play(lambda m: '{"tool": "end_window"}', "first-night", 7)
    ref = run_policy_episode("smart-triage", "first-night", 7).finalize(cal)
    d = diagnose({"summary": {"task": "first-night"}, "turns": out["turns"], "ledger": out["ledger"]}, ref)
    assert d["ops_unused_pct"] >= 90 and d["idle_windows"] == 4
    assert len(d["checkpoints_missed"]) == d["checkpoints_total"] == 3
    assert "SHELTER" in d["missing_milestones"]
