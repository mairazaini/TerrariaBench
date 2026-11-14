#!/usr/bin/env python3
"""
tools/entire_sync.py - MLH Entire Provenance & Session Metadata Logger

Logs agent trajectory sessions, OpenCode / Codex execution artifacts, and Git provenance
into `.entire/` metadata ledger for the 'Best Use of Entire' track.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

ENTIRE_DIR = Path(".entire")
SESSIONS_FILE = ENTIRE_DIR / "sessions.jsonl"
PROVENANCE_FILE = ENTIRE_DIR / "provenance.json"


def ensure_entire_dir():
    ENTIRE_DIR.mkdir(parents=True, exist_ok=True)


def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "untracked"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        branch = "unknown"
    return commit, branch


def log_session(agent_name: str, task: str, actions_count: int, reward: float, metadata: dict = None):
    ensure_entire_dir()
    commit, branch = get_git_info()
    record = {
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": agent_name,
        "task": task,
        "git_commit": commit,
        "git_branch": branch,
        "actions_count": actions_count,
        "final_reward": reward,
        "environment": "TerrariaBench-04-physics-and-shelter",
        "metadata": metadata or {},
    }
    with open(SESSIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def record_provenance(event: str, commit_sha: str = None):
    ensure_entire_dir()
    commit, branch = get_git_info()
    sha = commit_sha or commit
    provenance = {
        "event": event,
        "commit": sha,
        "branch": branch,
        "timestamp": time.time(),
        "runner": "craci" if os.environ.get("GITHUB_ACTIONS") else "local",
        "framework": "Entire-Agent-Provenance-v1",
        "tracks": ["AfterQuery-RL", "MLH-Entire", "CRACI"],
    }
    with open(PROVENANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"[Entire] Provenance recorded: {provenance['commit'][:8]} ({event})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entire Session & Provenance Logger")
    parser.add_argument("--event", type=str, default="manual_sync", help="Event name")
    parser.add_argument("--commit", type=str, default=None, help="Commit SHA")
    parser.add_argument("--agent", type=str, default=None, help="Log agent session")
    parser.add_argument("--task", type=str, default="04_physics_and_shelter", help="Task name")
    parser.add_argument("--actions", type=int, default=0, help="Number of actions executed")
    parser.add_argument("--reward", type=float, default=0.0, help="Reward achieved")

    args = parser.parse_args()
    if args.agent:
        rec = log_session(args.agent, args.task, args.actions, args.reward)
        print(f"[Entire] Logged session for {args.agent}: reward={args.reward}")
    else:
        record_provenance(args.event, args.commit)
