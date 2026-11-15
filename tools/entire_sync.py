#!/usr/bin/env python3
"""
tools/entire_sync.py
Entire Session Trail Logger for TerrariaBench.
Records agent decision traces, tool calls, and run provenance to .entire/trails.jsonl
for the Best Use of Entire track at Junction.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
ENTIRE_DIR = ROOT_DIR / ".entire"
TRAILS_FILE = ENTIRE_DIR / "trails.jsonl"

def log_trail_session(run_id: str, agent_name: str, score: float, checkpoints_count: int, metadata: dict = None):
    ENTIRE_DIR.mkdir(parents=True, exist_ok=True)
    trail_entry = {
        "trail_id": f"entire_trail_{run_id}_{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "agent": agent_name,
        "score": score,
        "checkpoints_recorded": checkpoints_count,
        "provenance": {
            "branch": "junction-final-submission",
            "environment": "TerrariaArchitectEnv",
            "reward_function": "StartRoomCheck_v1",
            "telemetry_sink": "AivenSink_HMAC"
        },
        "metadata": metadata or {}
    }

    with open(TRAILS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trail_entry) + "\n")
    print(f"[EntireSync] Trail session logged -> {TRAILS_FILE}")
    return trail_entry

def sync_all_runs():
    runs_dir = ROOT_DIR / "runs"
    if not runs_dir.exists():
        print("[EntireSync] No runs found.")
        return

    count = 0
    for run_folder in runs_dir.iterdir():
        if run_folder.is_dir():
            cp_file = run_folder / "checkpoints.jsonl"
            if cp_file.exists():
                lines = cp_file.read_text(encoding="utf-8").strip().split("\n")
                last_line = json.loads(lines[-1]) if lines else {}
                score = last_line.get("score", 0.0)
                agent = run_folder.name.split("_")[1] if "_" in run_folder.name else "agent"
                log_trail_session(run_folder.name, agent, score, len(lines))
                count += 1
    print(f"[EntireSync] Synced {count} runs to {TRAILS_FILE}")

if __name__ == "__main__":
    sync_all_runs()
