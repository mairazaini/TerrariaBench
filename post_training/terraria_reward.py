"""Prompts and reward for GRPO post-training, grounded in the real benchmark.

Pure Python (no torch / modal), so it runs and is tested on a laptop.

Every training prompt is one real training cell (TRAIN_TASKS x REPORTED_SEEDS).
The model answers with an opening plan: one JSON tool call per line. The reward
replays that plan through the real ``Episode`` tool surface, lets the remaining
windows settle unattended, and scores the sealed ledger with
``grader.training_reward`` (the [-2, 1] clipped channel) against the cell's
calibrated (z, s). Eval tasks and held-out seeds are never touched.
"""
from __future__ import annotations

import json

from terraria_bench.grader import DENSE_CLIP, training_reward
from terraria_bench.provenance import load_calibration
from terraria_bench.tasks import EVAL_TASKS, REPORTED_SEEDS, TASKS, TRAIN_TASKS
from terraria_bench.tools import TOOL_SCHEMAS, Episode

TRAIN_CELLS = tuple((t, s) for t in TRAIN_TASKS for s in REPORTED_SEEDS)
assert not set(TRAIN_TASKS) & set(EVAL_TASKS), "train/eval task leakage"

MAX_CALLS = 24            # plan calls replayed per rollout (bounded by completion length anyway)
MALFORMED_REWARD = DENSE_CLIP[0]   # no parseable tool call at all
BAD_LINE_PENALTY = 0.02   # per unparseable / unknown-tool / bad-argument line

_TOOL_LIST = "\n".join(
    f"- {name}({', '.join(f'{k}: {v}' for k, v in spec['params'].items())}): {spec['description']}"
    for name, spec in TOOL_SCHEMAS.items()
)

SYSTEM_PROMPT = (
    "You control a Terraria player through tools. Tools:\n" + _TOOL_LIST + "\n\n"
    "Reply with your opening plan as one JSON object per line and nothing else, e.g.\n"
    '{"tool": "chop_trees", "args": {}}\n'
    '{"tool": "build_blueprint", "args": {"preset": "shelter"}}\n'
    "Calls run in order. Whatever you leave unplanned plays out with you standing where you are."
)


def build_prompt(task_id: str, seed: int) -> str:
    ep = Episode(task_id, seed, redact_seed=True)
    return f"{SYSTEM_PROMPT}\n\n{TASKS[task_id].prompt}\n\nCurrent state: {ep.header()}\n\nPlan:\n"


def training_rows(n: int) -> list[dict]:
    """n prompt rows cycling over the 12 training cells."""
    rows = []
    for i in range(n):
        task_id, seed = TRAIN_CELLS[i % len(TRAIN_CELLS)]
        rows.append({"prompt": build_prompt(task_id, seed), "task_id": task_id, "seed": seed})
    return rows


def parse_plan(text: str) -> tuple[list[tuple[str, dict]], int]:
    """(valid calls, number of bad lines). Code fences and blank lines are ignored."""
    calls, bad = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            name, args = obj["tool"], obj.get("args") or {}
            if name not in TOOL_SCHEMAS or not isinstance(args, dict):
                raise ValueError(name)
        except (ValueError, KeyError, TypeError):
            bad += 1
            continue
        calls.append((name, args))
    return calls, bad


def rollout_reward(text: str, task_id: str, seed: int, calibration: dict) -> float:
    if task_id not in TRAIN_TASKS or seed not in REPORTED_SEEDS:
        raise ValueError(f"{task_id}:{seed} is not a training cell")
    calls, bad = parse_plan(text)
    if not calls:
        return MALFORMED_REWARD
    ep = Episode(task_id, seed)
    for name, args in calls[:MAX_CALLS]:
        if ep.done:
            break
        try:
            ep.step(name, **args)
        except Exception:  # bad argument names/types from the model
            bad += 1
    reward = training_reward(ep.finalize(), calibration["cells"][f"{task_id}:{seed}"])
    return max(DENSE_CLIP[0], min(DENSE_CLIP[1], reward - BAD_LINE_PENALTY * bad))


_CAL = None


def grpo_reward(completions, task_id, seed, **kwargs) -> list[float]:
    """TRL GRPOTrainer reward function (dataset columns arrive as keyword lists)."""
    global _CAL
    if _CAL is None:
        _CAL = load_calibration()
    out = []
    for c, t, s in zip(completions, task_id, seed):
        text = c if isinstance(c, str) else c[-1]["content"]
        out.append(rollout_reward(text, t, int(s), _CAL))
    return out
