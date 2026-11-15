"""Score real tModLoader runs recorded by live_play.py on port 8765.

``python -m terraria_bench.real_eval --all --save`` rescores every committed real-game recording
from its frames: no game, no model and no network are needed, so anyone can check the live numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .provenance import REPO_DIR

DEFAULT_RECORDING = REPO_DIR / "artifacts" / "recordings" / "real_terraria_live.json"
REAL_RUNS_DIR = REPO_DIR / "artifacts" / "runs" / "real_terraria"
RECORDINGS_DIR = DEFAULT_RECORDING.parent

# The road to the Eye of Cthulhu in the order a player walks it. The first unmet stage is where a run stalled.
STAGES: tuple[str, ...] = (
    "wood_gathered", "workbench", "shelter", "weapons", "ammunition_ready", "stone_or_ore_mined",
    "furnace_smelting", "anvil_placed", "metal_gear", "lenses_collected", "altar_found", "arena_built",
    "summon_ready",
)
# Stages that make up the first-night house task (the simulator's ``first-night`` cell asks for the same).
HOUSE_STAGES: tuple[str, ...] = ("wood_gathered", "workbench", "shelter")


def _rel(path: Path) -> str:
    """Repo-relative POSIX path when possible, so saved scores carry no machine-specific prefix."""
    try:
        return path.resolve().relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _frame_state(frame: Any) -> dict[str, Any]:
    state = frame.get("state") if isinstance(frame, dict) else None
    return state if isinstance(state, dict) else {}


def _trajectory(frames: list[Any]) -> dict[str, Any]:
    """What changed across the recorded frames: new milestones, deaths, depth and model latency."""
    states = [_frame_state(f) for f in frames]
    first, last = (states[0], states[-1]) if states else ({}, {})
    start_ms = {str(m) for m in (first.get("milestones") or [])}
    gained = [str(m) for m in (last.get("milestones") or []) if str(m) not in start_ms]
    depths = [
        s["referee"]["depth_below_surface"] for s in states
        if isinstance(s.get("referee"), dict) and s["referee"].get("depth_below_surface") is not None
    ]
    latency: dict[str, float] = {}
    for tier in ("supervisor", "short_term", "vision", "decision"):
        seen = []
        for f in frames:
            t = (f.get("tiers") or {}).get(tier) if isinstance(f, dict) else None
            if isinstance(t, dict) and t.get("last_latency_s"):
                seen.append(float(t["last_latency_s"]))
        if seen:
            latency[tier] = round(statistics.median(seen), 2)
    turns = [f.get("turn") for f in frames if isinstance(f, dict) and f.get("turn") is not None]
    return {
        "first_turn": turns[0] if turns else None,
        "last_turn": turns[-1] if turns else None,
        "milestones_gained_in_recording": gained,
        "deaths_in_recording": int(last.get("deaths") or 0) - int(first.get("deaths") or 0),
        "max_depth_below_surface": max(depths) if depths else None,
        "median_latency_s": latency,
    }


def _referee_check(last_state: dict[str, Any], eye_killed: bool, deaths: int) -> dict[str, Any]:
    """Compare the score against what the game itself reported (the mod's referee block), not the model.

    A recording whose claimed kill or death count disagrees with the referee is marked inconsistent.
    """
    referee = last_state.get("referee") if isinstance(last_state.get("referee"), dict) else None
    if referee is None:
        return {"available": False, "consistent": None, "mismatches": []}
    mismatches = []
    if bool(referee.get("downed_eye")) != bool(eye_killed):
        mismatches.append("eye_killed")
    if referee.get("deaths") is not None and int(referee["deaths"]) != int(deaths):
        mismatches.append("deaths")
    return {"available": True, "consistent": not mismatches, "mismatches": mismatches}


def score_real_recording(recording_path: Path = DEFAULT_RECORDING) -> dict[str, Any]:
    """Compute summary metrics and EyeProgress score from a recorded tModLoader run."""
    data = json.loads(recording_path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    score_obj = data.get("score") if isinstance(data.get("score"), dict) else {}
    last_frame = frames[-1] if frames and isinstance(frames[-1], dict) else {}
    last_state = last_frame.get("state") if isinstance(last_frame.get("state"), dict) else {}

    eye_prog = (
        data.get("eye_progress")
        or score_obj.get("eye_breakdown")
        or last_frame.get("eye_progress")
        or last_state.get("eye_breakdown")
        or {}
    )
    milestones = data.get("milestones") or last_state.get("milestones") or []

    if data.get("final_deaths") is not None:
        deaths = int(data["final_deaths"])
    elif score_obj.get("deaths") is not None:
        deaths = int(score_obj["deaths"])
    else:
        deaths = int(last_state.get("deaths") or 0)

    eye_pct = float(
        eye_prog.get("eye_progress_pct")
        if eye_prog.get("eye_progress_pct") is not None
        else (score_obj.get("eye_progress_pct") or 0.0)
    )
    eye_killed = bool(eye_prog.get("eye_killed"))
    survival = max(0.0, round(1.0 / (1.0 + 0.5 * deaths), 4))
    checklist = eye_prog.get("task_checklist") or {}
    stages_done = [s for s in STAGES if checklist.get(s)]
    stalled_at = next((s for s in STAGES if not checklist.get(s)), None)
    referee = _referee_check(last_state, eye_killed, deaths)
    summary = {
        "source": "tmodloader_real_game",
        "run_id": data.get("run_id", recording_path.stem),
        "player": str(data.get("player") or "model"),
        "models": {
            name: tier["model"] for name, tier in (data.get("tiers") or {}).items()
            if isinstance(tier, dict) and tier.get("model")
        } or {k: data[k] for k in ("supervisor_model", "vision_model", "decision_model") if data.get(k)},
        "recording": _rel(recording_path),
        "frames": len(frames),
        "house_built": all(checklist.get(s) for s in HOUSE_STAGES),
        "stages_done": len(stages_done),
        "stages_total": len(STAGES),
        "stalled_at": None if eye_killed else stalled_at,
        "referee": referee,
        "trajectory": _trajectory(frames),
        "eye_progress_pct": eye_pct,
        "eye_killed": eye_killed,
        "passed": eye_killed and referee["consistent"] is not False,
        "max_hp": int(eye_prog.get("max_hp") or last_state.get("max_hp") or 100),
        "defense": int(eye_prog.get("defense") or last_state.get("defense") or 0),
        "deaths": deaths,
        "survival_score": survival,
        "milestones": milestones,
        "task_checklist": eye_prog.get("task_checklist") or {},
        "breakdown": {
            "health_score": float(eye_prog.get("health_score") or 0.0),
            "armor_score": float(eye_prog.get("armor_score") or 0.0),
            "tasks_score": float(eye_prog.get("tasks_score") or 0.0),
            "boss_fight_score": float(eye_prog.get("boss_fight_score") or 0.0),
            "eye_kill_score": float(eye_prog.get("eye_kill_score") or 0.0),
        },
    }
    return summary


def find_real_recordings(folder: Path = RECORDINGS_DIR) -> list[Path]:
    """Every real-game recording in ``folder`` (simulator replays share the folder and are skipped)."""
    found = []
    for path in sorted(folder.glob("*.json")):
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(head, dict) and str(head.get("task_id", "")).startswith("real-terraria"):
            found.append(path)
    # live_play rewrites DEFAULT_RECORDING during a session. Once sessions are archived under their own
    # names (real_terraria_run1.json, ...), the rolling file is a duplicate or a half-finished run: skip it.
    archived = [path for path in found if path.name != DEFAULT_RECORDING.name]
    return archived or found


def score_all(folder: Path = RECORDINGS_DIR, save: bool = False, out_dir: Path = REAL_RUNS_DIR) -> list[dict[str, Any]]:
    """Rescore every real-game recording; with ``save`` write one score file per recording plus summary.json."""
    rows = [score_real_recording(path) for path in find_real_recordings(folder)]
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            name = Path(row["recording"]).stem
            (out_dir / f"{name}_score.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Recording | Player | Frames | Eye progress | House built | Stages | Stalled at | Deaths | Passed |",
        "|---|---|---:|---:|---|---:|---|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{Path(r['recording']).name}` | {r['player']} | {r['frames']} | {r['eye_progress_pct']:.0f}% "
            f"| {'yes' if r['house_built'] else 'no'} | {r['stages_done']}/{r['stages_total']} "
            f"| {r['stalled_at'] or '-'} | {r['deaths']} | {'yes' if r['passed'] else 'no'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a real tModLoader TerrariaBench run")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--all", action="store_true", help="Rescore every real-game recording in artifacts/recordings/")
    parser.add_argument("--save", action="store_true", help="Write ledger to artifacts/runs/real_terraria/")
    args = parser.parse_args(argv)
    if args.all:
        rows = score_all(save=args.save)
        if not rows:
            print(json.dumps({"error": f"No real-game recordings in {_rel(RECORDINGS_DIR)}"}))
            return 1
        print(format_table(rows))
        return 0
    if not args.recording.exists():
        print(json.dumps({"error": f"Recording not found: {args.recording}"}))
        return 1
    summary = score_real_recording(args.recording)
    if args.save:
        REAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REAL_RUNS_DIR / "latest_score.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
