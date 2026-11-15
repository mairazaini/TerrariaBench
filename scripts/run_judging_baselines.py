#!/usr/bin/env python3
"""Run the AfterQuery Break-the-Agent baseline suite on a task (default: eye-of-cthulhu).

    uv run python scripts/run_judging_baselines.py --task eye-of-cthulhu --seed 7

Runs:
  1. Reference policy (smart-triage)
  2. Do-nothing baseline (no actions)
  3. Stop-early baseline (status + end_window once, then quits)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from terraria_bench.grader import grade  # noqa: E402
from terraria_bench.policies import run_policy_episode  # noqa: E402
from terraria_bench.provenance import load_calibration  # noqa: E402
from terraria_bench.tasks import EVAL_TASKS, HARDCORE_TASKS  # noqa: E402
from terraria_bench.tools import Episode  # noqa: E402


def run_stop_early(task_id: str, seed: int, calibration: dict) -> dict:
    ep = Episode(task_id, seed, redact_seed=True)
    ep.step("terraria_status")
    ep.step("end_window")
    return ep.finalize(calibration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="eye-of-cthulhu", choices=list(HARDCORE_TASKS) + list(EVAL_TASKS))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cal = load_calibration()
    cell = cal["cells"][f"{args.task}:{args.seed}"]

    ref_ledger = run_policy_episode("smart-triage", args.task, args.seed).finalize(cal)
    nop_ledger = run_policy_episode("do-nothing", args.task, args.seed).finalize(cal)
    early_ledger = run_stop_early(args.task, args.seed, cal)

    rows = []
    for name, ledger in (("reference (smart-triage)", ref_ledger), ("do-nothing", nop_ledger), ("stop-early", early_ledger)):
        g = grade(ledger, cell)
        rows.append({"baseline": name, "composite": g["composite"], "passed": g.get("passed"), "eye_progress_pct": g.get("eye_progress_pct")})
    print(json.dumps({"task": args.task, "seed": args.seed, "threshold": cell["threshold"], "baselines": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
