"""Gates (i)-(iv) over every task x seed cell, plus the anchor pins.

The sealed artifact is loaded when it is fresh (source hash matches); if it
is missing or stale the calibration is rebuilt in memory so the gates are
always checked against the code as it stands. ``test_artifact_is_fresh``
separately fails when the committed artifact needs `python -m
terraria_bench.calibrate`.
"""
from __future__ import annotations

import pytest

from terraria_bench.calibrate import (
    ALL_SEEDS,
    FLAP_MARGIN,
    GATES,
    JUDGMENT_GAP,
    build_calibration,
    calibrate_cell,
    cell_key,
    check_gates,
    get_cell,
    threshold_for,
)
from terraria_bench.grader import grade
from terraria_bench.policies import ANCHORS, run_policy
from terraria_bench.provenance import CALIBRATION_PATH, ProvenanceError, load_calibration, source_hash
from terraria_bench.tasks import HELD_OUT_SEEDS, REPORTED_SEEDS, TASKS

CELLS = [(t, s) for t in TASKS for s in ALL_SEEDS]


@pytest.fixture(scope="session")
def calibration():
    try:
        return load_calibration()
    except ProvenanceError:
        return build_calibration()


def test_artifact_is_fresh():
    try:
        cal = load_calibration()
    except ProvenanceError as exc:
        pytest.fail(f"{exc} (artifact: {CALIBRATION_PATH})")
    assert cal["source_hash"] == source_hash()


def test_covers_every_cell(calibration):
    assert len(TASKS) == 12
    assert calibration["seeds"] == list(REPORTED_SEEDS) == [7, 11, 13]
    assert calibration["held_out_seeds"] == list(HELD_OUT_SEEDS) == [17, 23]
    assert set(calibration["cells"]) == {cell_key(t, s) for t, s in CELLS}
    assert len(calibration["cells"]) == 36 + 24


@pytest.mark.parametrize("task_id,seed", CELLS, ids=[cell_key(t, s) for t, s in CELLS])
def test_gates(calibration, task_id, seed):
    cell = calibration["cells"][cell_key(task_id, seed)]
    a = cell["anchors"]
    # Recompute the gates from the stored scores rather than trusting the flag.
    gates = check_gates(a, cell["threshold"])
    assert gates == cell["gates"]
    assert all(gates.values()), {g: gates[g] for g in GATES if not gates[g]}
    assert cell["gates_passed"] is True
    assert cell["threshold"] == threshold_for(a["rule-based"], a["smart-triage"])
    # anchor pins
    assert a["do-nothing"] == 0.0
    assert a["absentee"] == pytest.approx(0.1, abs=1e-4)
    assert a["exploiter"] < 0.05
    # gate arithmetic, spelled out
    assert min(a["rule-based"], a["smart-triage"]) > max(a["blast-list"], a["absentee"], a["do-nothing"])
    assert max(a["rule-based"], a["smart-triage"]) - a["blast-list"] >= JUDGMENT_GAP - 1e-9
    assert a["smart-triage"] - cell["threshold"] >= FLAP_MARGIN - 1e-9
    assert cell["s"] >= 1.0


def test_pass_rates(calibration):
    summary = calibration["summary"]
    assert set(summary) == set(ANCHORS)
    assert summary["smart-triage"]["pass_rate"] >= 0.95
    assert summary["blast-list"]["pass_rate"] == 0.0
    assert summary["do-nothing"]["pass_rate"] == summary["absentee"]["pass_rate"] == 0.0
    assert summary["exploiter"]["pass_rate"] == 0.0
    assert summary["do-nothing"]["mean_composite"] == 0.0
    assert summary["exploiter"]["mean_composite"] < 0.05
    order = [summary[p]["mean_composite"] for p in ("do-nothing", "absentee", "blast-list", "rule-based", "smart-triage")]
    assert order == sorted(order)


def test_get_cell_both_layouts(calibration):
    by_cells = get_cell(calibration, "first-night", 7)
    by_tasks = calibration["tasks"]["first-night"]["7"]
    for k in ("z", "s", "threshold"):
        assert by_cells[k] == by_tasks[k]
    assert get_cell({"tasks": calibration["tasks"]}, "first-night", 7)["s"] == by_cells["s"]
    with pytest.raises(KeyError):
        get_cell(calibration, "first-night", 99)


def test_cell_is_reproducible(calibration):
    """A live re-run of one cell reproduces the sealed numbers exactly."""
    live = calibrate_cell("shelter-crunch", 11)
    stored = calibration["cells"]["shelter-crunch:11"]
    for k in ("z", "s", "threshold", "anchors", "gates"):
        assert live[k] == stored[k], k


def test_grade_against_calibration(calibration):
    cell = get_cell(calibration, "first-night", 13)
    smart = grade(run_policy("smart-triage", "first-night", 13, calibration), cell)
    blast = grade(run_policy("blast-list", "first-night", 13, calibration), cell)
    assert smart["valid"] and smart["passed"] is True
    assert blast["valid"] and blast["passed"] is False
    assert smart["composite"] == calibration["cells"]["first-night:13"]["anchors"]["smart-triage"]
