"""Generate the Harbor tasks under harbor/tasks/ from harbor/template/.

    python harbor/build_tasks.py            # all eval tasks, seed 7
    python harbor/build_tasks.py --seed 11 --tasks first-night goblin-siege

Each task directory is a self-contained Harbor task: its `environment/game/`
build context carries a copy of the simulator core and the sealed calibration,
and `solution/actions.jsonl` is the recorded run of the scripted `smart-triage`
anchor policy. Re-run after any change to `terraria_bench/` or `artifacts/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from terraria_bench.grader import grade  # noqa: E402
from terraria_bench.policies import run_policy_episode  # noqa: E402
from terraria_bench.provenance import load_calibration  # noqa: E402
from terraria_bench.tasks import EVAL_TASKS, TASKS  # noqa: E402

TEMPLATE = HERE / "template"
OUT = HERE / "tasks"
# Simulator modules the game server needs (stdlib only). Nothing else is shipped.
CORE = ("__init__.py", "engine.py", "tools.py", "grader.py", "policies.py", "provenance.py",
        "tasks.py", "mcp_server.py")
TEXT_SUFFIXES = {".md", ".toml", ".yaml", ".sh", ".py", ""}


def _fill(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(f"__{key}__", value)
    return text


def build(task_id: str, seed: int, calibration: dict) -> Path:
    task = TASKS[task_id]
    name = f"terraria-{task_id}" if seed == 7 else f"terraria-{task_id}-seed{seed}"
    dest = OUT / name
    if dest.exists():
        shutil.rmtree(dest)

    total_ops = task.windows * task.budget
    values = {
        "TASK_ID": task_id,
        "SEED": str(seed),
        # Not a secret from the repo's readers; it only stops the agent's container
        # (which never sees tests/ or the sidecar's environment) from ending its own run.
        "TOKEN": hashlib.sha256(f"terrariabench-verifier:{task_id}:{seed}".encode()).hexdigest()[:32],
        "TITLE": task.title,
        "PROMPT": task.prompt,
        "WINDOWS": str(task.windows),
        "BUDGET": str(task.budget),
        "DIFFICULTY": "medium" if total_ops <= 120 else "hard",
        "AGENT_TIMEOUT": str(max(1800, total_ops * 30)),
    }
    for src in TEMPLATE.rglob("*"):
        if src.is_dir() or "__pycache__" in src.parts:
            continue
        dst = dest / src.relative_to(TEMPLATE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_SUFFIXES:
            # LF always: these files run inside Linux containers.
            dst.write_text(_fill(src.read_text(encoding="utf-8"), values), encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(src, dst)

    game = dest / "environment" / "game"
    (game / "terraria_bench").mkdir()
    for module in CORE:  # bytes, so source_hash() matches the sealed calibration
        shutil.copyfile(REPO / "terraria_bench" / module, game / "terraria_bench" / module)
    (game / "artifacts").mkdir()
    for artifact in ("calibration.json", "seed_data.json"):
        shutil.copyfile(REPO / "artifacts" / artifact, game / "artifacts" / artifact)

    episode = run_policy_episode("smart-triage", task_id, seed)
    with open(dest / "solution" / "actions.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for step in episode.actions_log:
            fh.write(json.dumps({"tool": step["tool"], "args": step["args"]}) + "\n")
    oracle = grade(episode.finalize(calibration), calibration["cells"][f"{task_id}:{seed}"])["composite"]
    print(f"{name}: {len(episode.actions_log)} oracle calls, oracle score {oracle:.4f}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks", nargs="*", default=list(EVAL_TASKS), choices=list(EVAL_TASKS))
    args = parser.parse_args()
    calibration = load_calibration()
    for task_id in args.tasks:
        build(task_id, args.seed, calibration)


if __name__ == "__main__":
    main()
