"""Calibration runner and the four StoreBench calibration gates.

    python -m terraria_bench.calibrate [--serial] [--workers N] [--quiet]

For every (task, seed) cell over REPORTED_SEEDS + HELD_OUT_SEEDS it runs the
six anchors, sets

    z = delta_net(absentee)
    s = max(1, best honest delta_net - z)
    threshold = midpoint(rule-based, smart-triage)   (flap-margin rule on ties)

grades every anchor with (z, s), enforces Gates (i)-(iv), and writes the sealed
``artifacts/calibration.json`` plus the pinned world hashes in
``artifacts/seed_data.json``. Exit status is non-zero if any gate fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from .engine import world_hash
from .grader import components, grade
from .policies import ANCHORS, HONEST_ANCHORS, run_policy
from .provenance import CALIBRATION_PATH, SEED_DATA_PATH, source_hash, write_sealed
from .tasks import HELD_OUT_SEEDS, REPORTED_SEEDS, TASKS

ALL_SEEDS = tuple(REPORTED_SEEDS) + tuple(HELD_OUT_SEEDS)
JUDGMENT_GAP = 0.15
FLAP_MARGIN = 0.02
EXPLOIT_CEILING = 0.05
ABSENTEE_CEILING = 0.1001
_EPS = 1e-9

GATES = ("i_ordering", "ii_judgment_gap", "iii_flap_margin", "iv_exploit_suppression")


def cell_key(task_id: str, seed: int) -> str:
    return f"{task_id}:{seed}"


def threshold_for(c_rb: float, c_st: float) -> float:
    """Midpoint of the two judgment anchors; when they tie (gap < 2 * flap
    margin) the bar sits one flap margin under the better of the two."""
    if abs(c_st - c_rb) < 2 * FLAP_MARGIN:
        return round(max(c_rb, c_st) - FLAP_MARGIN, 4)
    return round(0.5 * (c_rb + c_st), 4)


def check_gates(scores: dict, threshold: float) -> dict:
    dn, ab, bl, rb, st, ex = (scores[p] for p in ANCHORS)
    return {
        "i_ordering": min(rb, st) > max(bl, ab, dn),
        "ii_judgment_gap": max(rb, st) - bl >= JUDGMENT_GAP - _EPS,
        "iii_flap_margin": st - threshold >= FLAP_MARGIN - _EPS,
        "iv_exploit_suppression": ex < min(bl, rb, st) and ex < EXPLOIT_CEILING and dn == 0.0 and ab <= ABSENTEE_CEILING,
    }


def score_cell(task_id: str, seed: int, ledgers: dict) -> dict:
    """Turn the six anchor ledgers of one cell into its calibration entry."""
    comp = {p: components(ledgers[p]) for p in ANCHORS}
    z = comp["absentee"]["delta_net"]
    best = max(comp[p]["delta_net"] for p in HONEST_ANCHORS)
    s = max(1.0, round(best - z, 6))
    graded = {p: grade(ledgers[p], {"z": z, "s": s}) for p in ANCHORS}
    scores = {p: graded[p]["composite"] for p in ANCHORS}
    threshold = threshold_for(scores["rule-based"], scores["smart-triage"])
    gates = check_gates(scores, threshold)
    return {
        "task": task_id,
        "seed": seed,
        "z": z,
        "s": s,
        "threshold": threshold,
        "anchors": scores,
        "delta_net": {p: comp[p]["delta_net"] for p in ANCHORS},
        "gates": gates,
        "gates_passed": all(gates.values()),
    }


def calibrate_cell(task_id: str, seed: int) -> dict:
    """Run the six anchors on one cell (in-process) and score it."""
    return score_cell(task_id, seed, {p: run_policy(p, task_id, seed) for p in ANCHORS})


def default_workers() -> int:
    """One worker per physical core: the engine is pure-Python CPU work, and
    hyperthread siblings slow the pool down (measured 9.9s at 4 workers vs
    13.0s at 8 on a 4-core/8-thread machine)."""
    return max(1, (os.cpu_count() or 2) // 2)


def _worker_init() -> None:
    # Each ledger's run manifest shells out to `git rev-parse HEAD` (~30 ms).
    # The commit cannot change during a calibration run, so memoize it per
    # worker process; ledger contents are unchanged.
    import functools

    from . import provenance
    provenance.git_commit = functools.lru_cache(maxsize=1)(provenance.git_commit)


def _anchor_job(args):
    policy, task_id, seed = args
    return run_policy(policy, task_id, seed)


def run_cells(cells, workers: int | None = None) -> list[dict]:
    cells = list(cells)
    workers = 1 if workers is not None and workers <= 1 else min(len(cells) * len(ANCHORS), workers or default_workers())
    if workers <= 1:
        return [calibrate_cell(t, s) for t, s in cells]
    # One job per (anchor, cell), heaviest first, so the pool drains evenly.
    weight = {"do-nothing": 0, "absentee": 0, "exploiter": 1, "blast-list": 2, "smart-triage": 3, "rule-based": 4}
    jobs = sorted(((p, t, s) for t, s in cells for p in ANCHORS),
                  key=lambda j: -(TASKS[j[1]].windows * TASKS[j[1]].budget * (1 + weight[j[0]])))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        ledgers = dict(zip(jobs, pool.map(_anchor_job, jobs, chunksize=1)))
    return [score_cell(t, s, {p: ledgers[(p, t, s)] for p in ANCHORS}) for t, s in cells]


def summarize(cells: dict) -> dict:
    summary = {}
    for p in ANCHORS:
        vals = [c["anchors"][p] for c in cells.values()]
        passed = [c["anchors"][p] >= c["threshold"] for c in cells.values()]
        summary[p] = {"mean_composite": round(sum(vals) / len(vals), 4),
                      "pass_rate": round(sum(passed) / len(passed), 4)}
    return summary


def build_calibration(workers: int | None = None, tasks=None, seeds=None) -> dict:
    tasks = tuple(tasks or TASKS)
    seeds = tuple(seeds or ALL_SEEDS)
    results = run_cells([(t, s) for t in tasks for s in seeds], workers=workers)
    cells = {cell_key(r["task"], r["seed"]): r for r in results}
    nested: dict = {}
    for r in results:
        nested.setdefault(r["task"], {})[str(r["seed"])] = {k: r[k] for k in ("z", "s", "threshold")}
    return {
        "source_hash": source_hash(),
        "seeds": [s for s in seeds if s in REPORTED_SEEDS],
        "held_out_seeds": [s for s in seeds if s in HELD_OUT_SEEDS],
        "cells": cells,
        "tasks": nested,
        "summary": summarize(cells),
        "all_gates_passed": all(r["gates_passed"] for r in results),
    }


def get_cell(calibration: dict, task_id: str, seed: int) -> dict:
    """The {z, s, threshold} entry for a cell, from either lookup layout."""
    cell = calibration.get("cells", {}).get(cell_key(task_id, seed))
    if cell is None:
        cell = calibration.get("tasks", {}).get(task_id, {}).get(str(seed))
    if cell is None:
        raise KeyError(f"no calibration for cell {cell_key(task_id, seed)}")
    return cell


def pin_seeds(seeds=ALL_SEEDS) -> dict:
    return write_sealed(SEED_DATA_PATH, {"source_hash": source_hash(),
                                         "seeds": {str(s): world_hash(s) for s in seeds}})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m terraria_bench.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: cpu_count // 2)")
    ap.add_argument("--serial", action="store_true", help="run in-process, no worker pool")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    pin_seeds()
    cal = build_calibration(workers=1 if args.serial else args.workers)
    sealed = write_sealed(CALIBRATION_PATH, cal)
    if not args.quiet:
        print(f"{'cell':26s} {'z':>7s} {'s':>8s} {'thr':>6s}  " + " ".join(f"{p[:6]:>6s}" for p in ANCHORS) + "  gates")
        for key, c in cal["cells"].items():
            marks = "".join("+" if c["gates"][g] else "-" for g in GATES)
            print(f"{key:26s} {c['z']:7.2f} {c['s']:8.2f} {c['threshold']:6.3f}  "
                  + " ".join(f"{c['anchors'][p]:6.3f}" for p in ANCHORS) + f"  {marks}")
    print("\nanchor          mean   pass_rate")
    for p, row in cal["summary"].items():
        print(f"{p:14s} {row['mean_composite']:6.3f}   {row['pass_rate']:6.2%}")
    failed = [k for k, c in cal["cells"].items() if not c["gates_passed"]]
    print(f"\n{len(cal['cells']) - len(failed)}/{len(cal['cells'])} cells pass gates (i)-(iv) "
          f"in {time.perf_counter() - t0:.1f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print(f"wrote {CALIBRATION_PATH} (sha256 {sealed['sha256'][:12]}) and {SEED_DATA_PATH}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
