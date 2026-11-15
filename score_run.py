#!/usr/bin/env python3
"""
score_run.py
Objective post-run scoring script for TerrariaBench.
Evaluates runs strictly from tModLoader checkpoints.jsonl logs.
"""

import sys
import json
from pathlib import Path

def score_run(run_dir: str):
    checkpoints_path = Path(run_dir) / "checkpoints.jsonl"
    if not checkpoints_path.exists():
        return {"status": "error", "reason": "checkpoints.jsonl missing", "score": 0.0}

    total_score = 0.0
    direct_tile_control_used = False
    events = []

    with open(checkpoints_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            events.append(data)
            if data.get("direct_tile_control_used"):
                direct_tile_control_used = True
            if "score" in data:
                total_score = max(total_score, float(data["score"]))

    if direct_tile_control_used:
        return {"status": "disqualified", "reason": "direct_tile_control_used", "score": 0.0}

    return {"status": "scored", "score": total_score, "events_count": len(events)}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "runs/smoke_01"
    print(json.dumps(score_run(target), indent=2))
