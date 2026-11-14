"""Repeatable Eye of Cthulhu trials with varied starting states and action replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..provenance import ARTIFACTS_DIR
from ..tasks import Task
from ..tools import Episode
from .runner import MAX_TURNS, _build_agent, run_agent_episode


def eye_trial_tasks() -> tuple[Task, ...]:
    """Return a small gear/readiness matrix sharing the same boss timing and world seeds."""
    eye_spawn = ({"kind": "eye_spawn", "w": 2},)
    common = {
        "split": "eval",
        "windows": 4,
        "budget": 20,
        "objectives": ("shelter", "gear", "arena", "boss:eye"),
        "events": eye_spawn,
        "sla": ((2, "boss:eye:0.6"), (4, "kill:eye")),
    }
    return (
        Task(
            "eye_trial_starter", "Eye trial: starter gear",
            brief=("Prepare for the Eye of Cthulhu. It will arrive on the second night. "
                   "You have only your starter equipment and a Suspicious Looking Eye."),
            kit={"suspicious_looking_eye": 1},
            **common,
        ),
        Task(
            "eye_trial_ranger", "Eye trial: early ranger kit",
            brief=("Prepare for the Eye of Cthulhu. It will arrive on the second night. "
                   "You have a wooden bow, arrows, wood, platforms and a campfire; build a useful arena."),
            kit={"suspicious_looking_eye": 1, "wooden_bow": 1, "wooden_arrow": 300,
                 "wood": 80, "wood_platform": 120, "campfire": 1},
            equip=("wooden_bow",), start_max_hp=160, start_milestones=("T1",),
            **common,
        ),
        Task(
            "eye_trial_gold", "Eye trial: prepared gold loadout",
            brief=("Fight the Eye of Cthulhu. It will arrive on the second night. "
                   "You have gold armor, a gold bow, arrows, healing potions and arena materials; "
                   "use the first day to position and prepare."),
            kit={"suspicious_looking_eye": 1, "gold_bow": 1, "gold_armor": 1,
                 "wooden_arrow": 500, "lesser_healing_potion": 5, "wood": 100,
                 "wood_platform": 160, "campfire": 1},
            equip=("gold_bow", "gold_armor"), start_max_hp=300,
            start_milestones=("T1", "T2", "T3", "T4"),
            **common,
        ),
    )


def replay_trial_report(report_path: Path) -> dict[str, Any]:
    """Replay each saved action sequence and verify its sealed final state hash."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tasks = {task.id: task for task in eye_trial_tasks()}
    seed = int(report["seed"])
    replayed = []
    for scenario in report.get("scenarios") or []:
        task_id = scenario.get("task")
        if task_id not in tasks:
            raise ValueError(f"Unknown Eye trial task in report: {task_id!r}")
        episode = Episode(tasks[task_id], seed, redact_seed=True)
        for action in scenario.get("actions") or []:
            if episode.done:
                raise ValueError(f"Action log continues after episode ended: {task_id}")
            episode.step(str(action["tool"]), **dict(action.get("args") or {}))
        ledger = episode.finalize()
        expected = (scenario.get("ledger") or {}).get("sha256")
        replayed.append({
            "task": task_id,
            "verified": bool(expected) and ledger.get("sha256") == expected,
            "expected_sha256": expected,
            "actual_sha256": ledger.get("sha256"),
        })
    return {"report": str(report_path), "replayed": replayed,
            "verified": bool(replayed) and all(item["verified"] for item in replayed)}


def run_eye_trials(agent_name: str, seed: int = 7, live: bool = False,
                   max_turns: int = MAX_TURNS, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the three fixed loadout scenarios and persist their ledgers plus replayable actions."""
    scenarios = []
    model_name = ""
    for task in eye_trial_tasks():
        agent = _build_agent(agent_name, live=live)
        model_name = str(getattr(agent, "model", getattr(getattr(agent, "decisions", None), "model", "")))
        result = run_agent_episode(agent, task, seed, calibration=None, max_turns=max_turns)
        scenarios.append({
            "task": task.id,
            "title": task.title,
            "score": result["score"],
            "turns": result["turns"],
            "ledger": result["ledger"],
            "actions": result["actions"],
        })

    report = {"format": "terrariabench-eye-trials-v1", "agent": agent_name,
              "model": model_name, "live": live, "seed": int(seed), "scenarios": scenarios}
    if output_dir is None:
        output_dir = ARTIFACTS_DIR / "runs" / "eye_trials" / agent_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"matrix_seed{seed}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run/replay the Eye of Cthulhu loadout trial matrix")
    parser.add_argument("--agent", choices=["gemini", "nemotron", "verda_deepseek", "verda_kimi",
                                                "gemma_base", "gemma_tuned"], default="gemini")
    parser.add_argument("--live", action="store_true", help="Use the configured live model provider")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path, help="Replay and verify an existing matrix report")
    args = parser.parse_args(argv)

    if args.replay:
        result = replay_trial_report(args.replay)
        print(json.dumps(result, indent=2))
        return 0 if result["verified"] else 1
    report = run_eye_trials(args.agent, args.seed, args.live, args.max_turns, args.output_dir)
    for scenario in report["scenarios"]:
        score = scenario["score"]
        killed = "KILL:eye" in (scenario["ledger"].get("milestones") or [])
        print(f"{scenario['task']}: eye_progress={score.get('eye_progress_pct', 0):.2f}% "
              f"eye_kill={killed} "
              f"turns={scenario['turns']}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
