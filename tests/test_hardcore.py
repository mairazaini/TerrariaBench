"""Hardcore tasks: one life, stated up front, and the reference survives every seed."""
from __future__ import annotations

import pytest

from terraria_bench.grader import grade
from terraria_bench.policies import run_policy_episode
from terraria_bench.provenance import load_calibration
from terraria_bench.tasks import HARDCORE_RULE, HARDCORE_TASKS, HELD_OUT_SEEDS, REPORTED_SEEDS, TASKS
from terraria_bench.tools import Episode


def test_hardcore_tasks_mirror_their_base_task_and_state_the_rule():
    assert HARDCORE_TASKS == ("first-night-hardcore", "eye-of-cthulhu-hardcore")
    for task_id in HARDCORE_TASKS:
        task, base = TASKS[task_id], TASKS[task_id.removesuffix("-hardcore")]
        assert task.hardcore and not base.hardcore
        assert (task.windows, task.budget, task.sla, task.events, task.kit, task.brief) == \
            (base.windows, base.budget, base.sla, base.events, base.kit, base.brief)
        assert HARDCORE_RULE in task.prompt and HARDCORE_RULE not in base.prompt
        assert "inspect_housing_and_arena" in HARDCORE_RULE  # the tool that prevents the commonest death is named


@pytest.mark.parametrize("task_id", HARDCORE_TASKS)
def test_the_reference_never_dies_and_meets_every_checkpoint(task_id):
    cal = load_calibration()
    for seed in REPORTED_SEEDS + HELD_OUT_SEEDS:
        ep = run_policy_episode("smart-triage", task_id, seed)
        ledger = ep.finalize(cal)
        assert ledger["deaths"] == 0, (task_id, seed)
        assert all(ok for w in ledger["windows"] for _, ok in w["sla"]), (task_id, seed)
        assert grade(ledger, cal["cells"][f"{task_id}:{seed}"])["composite"] >= 0.95


def test_first_death_ends_the_run_and_later_checkpoints_are_missed():
    ep = Episode("first-night-hardcore", 7)
    last = {}
    while not ep.done:  # standing outside through the first night is fatal
        last = ep.step("end_window")
    assert ep.world.deaths == 1 and "Hardcore" in last["hardcore"] and last["done"] is True
    died_in = len(ep.windows_log)
    assert died_in < TASKS["first-night-hardcore"].windows
    refused = ep.step("chop_trees")
    assert refused["ok"] is False and "over" in refused["error"]
    assert len(ep.actions_log) == died_in  # the refused call was never recorded or metered
    ledger = ep.finalize()
    assert [w["attended"] for w in ledger["windows"]][died_in:] == [False] * (4 - died_in)
    assert not any(ok for w in ledger["windows"][died_in:] for _, ok in w["sla"])


def test_dying_late_scores_less_than_surviving_with_the_same_progress():
    """Same calls as the reference, except the very last one leaves the shelter: the run keeps its progress
    but loses the final window's survival and checkpoint."""
    cal = load_calibration()
    cell = cal["cells"]["first-night-hardcore:7"]
    ref = run_policy_episode("smart-triage", "first-night-hardcore", 7)
    actions = list(ref.actions_log)
    safe = grade(ref.finalize(cal), cell)
    ep = Episode("first-night-hardcore", 7)
    for a in actions[:-1]:
        ep.step(a["tool"], **a["args"])
    while not ep.done:  # walk away from home and let the window close outside
        ep.step("navigate_to", x=ep.world.home[0] + 30, y=ep.world.home[1])
    reckless = grade(ep.finalize(cal), cell)
    assert reckless["deaths"] == 1 and safe["deaths"] == 0
    assert reckless["composite"] < safe["composite"] - 0.05


def test_normal_tasks_are_unaffected():
    ep = Episode("first-night", 7)
    while not ep.done:
        ep.step("end_window")
    assert len(ep.windows_log) == 4 and ep.world.deaths >= 1  # died, respawned, kept playing
