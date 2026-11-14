"""The generated Harbor tasks must match the template and the live simulator source.

`harbor/tasks/` holds copies (each Harbor environment is a self-contained build
context). A stale copy would silently grade against an old world, so drift fails here.
Fix with: python harbor/build_tasks.py
"""
from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

from terraria_bench.tasks import EVAL_TASKS, TASKS

REPO = Path(__file__).resolve().parents[1]
HARBOR = REPO / "harbor"

_spec = importlib.util.spec_from_file_location("build_tasks", HARBOR / "build_tasks.py")
build_tasks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_tasks)


@pytest.mark.parametrize("task_id", EVAL_TASKS)
def test_task_has_the_harbor_layout(task_id):
    root = HARBOR / "tasks" / f"terraria-{task_id}"
    for rel in ("instruction.md", "task.toml", "environment/Dockerfile", "environment/docker-compose.yaml",
                "environment/game/Dockerfile", "environment/game/server.py", "tests/test.sh",
                "solution/solve.sh", "solution/actions.jsonl"):
        assert (root / rel).is_file(), rel
    config = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
    assert config["environment"]["mcp_servers"][0]["url"] == "http://game:8000/mcp"
    assert config["agent"]["timeout_sec"] >= 1800
    text = "".join((root / rel).read_text(encoding="utf-8")
                   for rel in ("instruction.md", "task.toml", "environment/docker-compose.yaml", "tests/test.sh"))
    assert "__" not in text.replace("__main__", ""), "unfilled template placeholder"
    assert TASKS[task_id].prompt in (root / "instruction.md").read_text(encoding="utf-8")
    for rel in ("tests/test.sh", "solution/solve.sh", "environment/terraria"):
        assert b"\r\n" not in (root / rel).read_bytes(), f"{rel} must use LF line endings"


@pytest.mark.parametrize("task_id", EVAL_TASKS)
def test_task_copies_are_not_stale(task_id):
    game = HARBOR / "tasks" / f"terraria-{task_id}" / "environment" / "game"
    for module in build_tasks.CORE:
        assert (game / "terraria_bench" / module).read_bytes() == (REPO / "terraria_bench" / module).read_bytes(), module
    for artifact in ("calibration.json", "seed_data.json"):
        assert (game / "artifacts" / artifact).read_bytes() == (REPO / "artifacts" / artifact).read_bytes(), artifact
    assert (game / "server.py").read_bytes() == (HARBOR / "template" / "environment" / "game" / "server.py").read_bytes()


def test_instruction_does_not_leak_the_hidden_deadlines_or_seed():
    for task_id in EVAL_TASKS:
        text = (HARBOR / "tasks" / f"terraria-{task_id}" / "instruction.md").read_text(encoding="utf-8")
        assert "seed" not in text.lower()
        assert "sla" not in text.lower()


def test_oracle_actions_are_well_formed():
    for task_id in EVAL_TASKS:
        path = HARBOR / "tasks" / f"terraria-{task_id}" / "solution" / "actions.jsonl"
        steps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert steps and all(set(s) == {"tool", "args"} for s in steps)
