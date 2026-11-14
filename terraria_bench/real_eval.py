"""Score real tModLoader runs recorded by live_play.py on port 8765."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .provenance import REPO_DIR

DEFAULT_RECORDING = REPO_DIR / "artifacts" / "recordings" / "real_terraria_live.json"
REAL_RUNS_DIR = REPO_DIR / "artifacts" / "runs" / "real_terraria"


def score_real_recording(recording_path: Path = DEFAULT_RECORDING) -> dict[str, Any]:
    """Compute summary metrics and EyeProgress score from a recorded tModLoader run."""
    data = json.loads(recording_path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    eye_prog = data.get("eye_progress") or (frames[-1].get("eye_progress") if frames else {}) or {}
    last_state = (frames[-1].get("state") if frames else {}) or {}
    milestones = data.get("milestones") or []
    deaths = int(last_state.get("deaths") or 0)
    eye_pct = float(eye_prog.get("eye_progress_pct") or 0.0)
    eye_killed = bool(eye_prog.get("eye_killed"))
    survival = max(0.0, round(1.0 / (1.0 + 0.5 * deaths), 4))
    summary = {
        "source": "tmodloader_real_game",
        "recording": str(recording_path),
        "frames": len(frames),
        "eye_progress_pct": eye_pct,
        "eye_killed": eye_killed,
        "passed": eye_killed,
        "max_hp": int(eye_prog.get("max_hp") or last_state.get("max_hp") or 100),
        "defense": int(eye_prog.get("defense") or last_state.get("defense") or 0),
        "deaths": deaths,
        "survival_score": survival,
        "milestones": milestones,
        "task_checklist": eye_prog.get("task_checklist") or {},
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a real tModLoader TerrariaBench run")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--save", action="store_true", help="Write ledger to artifacts/runs/real_terraria/")
    args = parser.parse_args(argv)
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
