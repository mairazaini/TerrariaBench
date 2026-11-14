"""Closed-loop play and behaviour-cloning data: one tool call per turn, seeing each result.

Open-loop plans do not transfer between world seeds (an expert plan that scores
1.0 on its own seed scores 0.2-0.5 with deaths on another), so training data and
evaluation both use this turn-by-turn format instead.

    python -m post_training.closed_loop            # build + verify the BC dataset, print stats
"""
from __future__ import annotations

import json

from terraria_bench.grader import grade
from terraria_bench.policies import run_policy_episode
from terraria_bench.provenance import load_calibration
from terraria_bench.tasks import EVAL_TASKS, REPORTED_SEEDS, TASKS, TRAIN_TASKS
from terraria_bench.tools import TOOL_SCHEMAS, Episode

TRAIN_CELLS = tuple((t, s) for t in TRAIN_TASKS for s in REPORTED_SEEDS)
assert not set(TRAIN_TASKS) & set(EVAL_TASKS), "train/eval task leakage"

RECENT = 8                 # steps shown with their full result
MIN_DEMO_COMPOSITE = 0.95  # a demonstration episode must grade at least this, with zero deaths

_TOOL_LIST = "\n".join(
    f"- {name}({', '.join(f'{k}: {v}' for k, v in spec['params'].items())}): {spec['description']}"
    for name, spec in TOOL_SCHEMAS.items()
)

# Early-game order from the Terraria wiki (Guide:Getting started, Guide:Eye of Cthulhu
# strategies), restated in this environment's tools. Every line is something the
# verified demonstrations actually do.
STRATEGY = (
    "Strategy:\n"
    "1. Day 1: chop_trees, then build_blueprint preset=shelter before the first night.\n"
    "2. Craft and equip a wooden weapon and wood_armor (a wooden_bow plus wooden_arrow when a fight is announced).\n"
    "3. mine_vein stone, craft a furnace and place it at home.\n"
    "4. explore_cavern, mine iron_ore, smelt iron_bar, craft an anvil, then iron_bow, iron_armor, iron_pickaxe.\n"
    "5. Then silver, then gold. Loot life crystals you find.\n"
    "6. Be inside your shelter before every night ends: navigate_to home with operations to spare.\n"
    "7. Head home when HP is low. Dying loses half your raw materials and next window's operations."
)

SYSTEM_PROMPT = (
    "You control a Terraria player through tools, one call per turn. Tools:\n" + _TOOL_LIST + "\n\n"
    + STRATEGY + "\n\n"
    'Reply with exactly one JSON object and nothing else, e.g. {"tool": "chop_trees", "args": {}}'
)
SYSTEM_PROMPT_NO_STRATEGY = SYSTEM_PROMPT.replace(STRATEGY + "\n\n", "")


def _call(tool: str, args: dict) -> str:
    return json.dumps({"tool": tool, "args": args}, separators=(", ", ": "))


def _body(result: dict) -> str:
    return json.dumps({k: v for k, v in result.items() if k not in ("header", "text")}, sort_keys=True)[:400]


def render(task_id: str, history: list, header: str, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Chat messages for the next decision. history = [(tool, args, result_body_str)]."""
    home = next((json.loads(r).get("position") for t, a, r in history
                 if t == "build_blueprint" and a.get("preset") == "shelter" and '"valid": true' in r), None)
    parts = [system, "", TASKS[task_id].prompt, ""]
    if home:
        parts.append(f"Home (shelter): {home}")
    old, recent = history[:-RECENT], history[-RECENT:]
    if old:
        parts.append("Earlier calls: " + "; ".join(f"{t}({json.dumps(a, separators=(',', ':'))})" for t, a, _ in old))
    for t, a, r in recent:
        parts.append(f"> {_call(t, a)}\n{r}")
    parts.append(f"\nNow: {header}\nNext call:")
    # A single user turn: Gemma chat templates have no system role.
    return [{"role": "user", "content": "\n".join(parts)}]


def parse_action(text: str):
    """First JSON object in the reply -> (tool, args), or None."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        tool, args = obj["tool"], obj.get("args") or {}
    except (ValueError, KeyError, TypeError):
        return None
    return (tool, args) if tool in TOOL_SCHEMAS and isinstance(args, dict) else None


def play(generate, task_id: str, seed: int, system: str = SYSTEM_PROMPT, max_bad: int = 8) -> dict:
    """Play one episode with generate(messages) -> str. Returns the sealed ledger
    plus counts. Unparseable or crashing replies cost nothing in-world but the
    episode stops (remaining windows unattended) after max_bad of them."""
    ep = Episode(task_id, seed, redact_seed=True)
    history, bad, turns = [], 0, 0
    max_turns = TASKS[task_id].windows * TASKS[task_id].budget * 2
    while not ep.done and turns < max_turns and bad < max_bad:
        turns += 1
        action = parse_action(generate(render(task_id, history, ep.header(), system)))
        if action is None:
            bad += 1
            continue
        try:
            result = ep.step(*action[:1], **action[1])
        except Exception as exc:  # wrong argument names / types
            bad += 1
            history.append((action[0], action[1], json.dumps({"ok": False, "error": str(exc)[:200]})))
            continue
        history.append((action[0], action[1], _body(result)))
    return {"ledger": ep.finalize(), "turns": turns, "bad": bad}


def expert_examples(policy: str = "smart-triage", cells=TRAIN_CELLS, calibration=None) -> tuple[list[dict], list[dict]]:
    """(examples, per-episode report). Each example is {"messages": [...user..., assistant]}
    taken from an expert episode that was re-played through the tool surface and
    graded; episodes below MIN_DEMO_COMPOSITE or with any death are dropped."""
    cal = calibration or load_calibration()
    examples, report = [], []
    for task_id, seed in cells:
        if task_id not in TRAIN_TASKS or seed not in REPORTED_SEEDS:
            raise ValueError(f"{task_id}:{seed} is not a training cell")
        actions = run_policy_episode(policy, task_id, seed).actions_log
        ep = Episode(task_id, seed, redact_seed=True)
        history, rows = [], []
        for a in actions:
            rows.append({"messages": render(task_id, history, ep.header())
                         + [{"role": "assistant", "content": _call(a["tool"], a["args"])}],
                         "task_id": task_id, "seed": seed})
            history.append((a["tool"], a["args"], _body(ep.step(a["tool"], **a["args"]))))
        ledger = ep.finalize()
        g = grade(ledger, cal["cells"][f"{task_id}:{seed}"])
        keep = g["valid"] and g["composite"] >= MIN_DEMO_COMPOSITE and ledger["deaths"] == 0
        report.append({"cell": f"{task_id}:{seed}", "steps": len(rows), "composite": g["composite"],
                       "deaths": ledger["deaths"], "kept": keep,
                       "max_prompt_chars": max(len(r["messages"][0]["content"]) for r in rows)})
        if keep:
            examples.extend(rows)
    return examples, report


if __name__ == "__main__":
    ex, rep = expert_examples()
    for r in rep:
        print(r)
    print(f"{len(ex)} examples from {sum(r['kept'] for r in rep)}/{len(rep)} episodes")
    # every target must parse back to the exact call, and replaying targets must reproduce the grade
    assert all(parse_action(e["messages"][-1]["content"]) for e in ex)
