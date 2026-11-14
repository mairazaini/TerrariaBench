"""Main TerrariaBench evaluation runner, 65k-token context compactor, and leaderboard printer."""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .. import provenance
from ..engine import AIR, CAMPFIRE, DOOR, WORKBENCH
from ..grader import components, grade
from ..provenance import ARTIFACTS_DIR, CALIBRATION_PATH, ProvenanceError, load_sealed, verify_seal, write_sealed
from ..tasks import EVAL_TASKS, REPORTED_SEEDS, TASKS
from ..tools import Episode
from .gemini_agent import GeminiAgent, GeminiLiveError
from .nemotron_agent import (
    DEFAULT_VERDA_DEEPSEEK_MODEL,
    DEFAULT_VERDA_KIMI_MODEL,
    VERDA_INFERENCE_BASE_URL,
    NemotronAgent,
    load_dotenv_if_present,
)

load_dotenv_if_present()

MAX_TURNS = 150
COMPACT_TOKEN_THRESHOLD = 65_000
KEEP_LAST_TURNS = 3
ANCHORS = ("smart-triage", "rule-based", "blast-list", "absentee", "do-nothing", "exploiter")


@runtime_checkable
class AgentProtocol(Protocol):
    name: str

    def next_action(self, observation: dict[str, Any], context: list[dict[str, Any]], ep: Episode) -> Any: ...


class CompactHistory:
    """Stores turn history and compacts older turns above ``threshold_tokens`` keeping ``keep_last`` verbatim."""

    def __init__(self, threshold_tokens: int = COMPACT_TOKEN_THRESHOLD, keep_last: int = KEEP_LAST_TURNS):
        self.threshold_tokens = int(threshold_tokens)
        self.keep_last = int(keep_last)
        self.turns: list[dict[str, Any]] = []
        self.summary: dict[str, Any] | None = None
        self.compactions: int = 0
        self.compacted_turns: int = 0

    def estimated_tokens(self) -> int:
        payload = ([self.summary] + self.turns) if self.summary is not None else self.turns
        return len(json.dumps(payload, separators=(",", ":"), sort_keys=True)) // 4

    def append(self, turn_record: dict[str, Any], ep: Episode) -> None:
        self.turns.append(turn_record)
        if self.estimated_tokens() > self.threshold_tokens and len(self.turns) > self.keep_last:
            dropped = self.turns[:-self.keep_last]
            self.turns = self.turns[-self.keep_last:]
            self.compacted_turns += len(dropped)
            self.compactions += 1
            w = ep.world
            self.summary = {
                "type": "compacted_summary",
                "compacted_turns": self.compacted_turns,
                "window": w.window,
                "phase": w.phase,
                "hp": w.hp,
                "max_hp": w.max_hp,
                "tech_tier": w.tech_tier,
                "equipped": dict(w.eq),
                "milestones": sorted(w.milestones),
                "home": list(w.home),
                "known_ores_or_chests": sorted([list(p) for p in list(w.known)[:12]]),
            }

    def render_context(self) -> list[dict[str, Any]]:
        return ([self.summary] + list(self.turns)) if self.summary is not None else list(self.turns)


def _normalize_action(raw: Any) -> tuple[str | None, dict[str, Any]]:
    if raw is None:
        return None, {}
    if isinstance(raw, dict):
        return raw.get("tool"), dict(raw.get("args") or {})
    if isinstance(raw, tuple) and len(raw) >= 2:
        return raw[0], dict(raw[1] or {})
    raise TypeError(f"Unsupported agent action format: {type(raw)!r}")


def run_agent_episode(
    agent: Any,
    task_id: str,
    seed: int,
    calibration: dict[str, Any] | None = None,
    max_turns: int = MAX_TURNS,
    save_dir: Path | None = None,
    on_turn: Any = None,
) -> dict[str, Any]:
    """Run ``agent`` over ``Episode(task_id, seed, redact_seed=True)`` up to ``max_turns``.

    A tool call the episode cannot execute (unknown tool, wrong argument names
    or types) is returned to the agent as an ``ok: False`` observation instead
    of aborting the run. ``on_turn(turn, tool, args, result)`` is called after
    every turn when given.
    """
    ep = Episode(task_id, seed, redact_seed=True)
    history = CompactHistory()
    obs: dict[str, Any] = {
        "ok": True,
        "header": ep.header(),
        "prompt": ep.task.prompt,
        "window": ep.world.window,
        "total_windows": ep.task.windows,
        "ops_left": ep.ops_left,
        "ops_budget": ep.window_budget,
    }
    turns_played = 0
    for turn in range(1, max_turns + 1):
        if ep.done:
            break
        tool_name, tool_args = _normalize_action(agent.next_action(obs, history.render_context(), ep))
        if not tool_name:
            break
        try:
            res = ep.step(tool_name, **tool_args)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
            res = {
                "ok": False,
                "error": f"invalid call {tool_name}({tool_args}): {type(exc).__name__}: {exc}",
                "header": ep.header(),
                "window": ep.world.window,
                "ops_left": ep.ops_left,
            }
        turns_played = turn
        if on_turn is not None:
            on_turn(turn, tool_name, tool_args, res)
        compact_res = {k: v for k, v in res.items() if k != "viewport_png_b64"}
        history.append({"turn": turn, "tool": tool_name, "args": tool_args, "result": compact_res}, ep)
        obs = res

    ledger = ep.finalize(calibration)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        write_sealed(save_dir / f"{task_id}_seed{seed}.json", ledger)

    cell = (calibration or {}).get("cells", {}).get(f"{task_id}:{seed}")
    score = grade(ledger, cell) if (cell and cell.get("s", 0) > 0) else components(ledger)
    return {
        "task": task_id,
        "seed": seed,
        "turns": turns_played,
        "compactions": history.compactions,
        "ledger": ledger,
        "score": score,
    }


class EpisodeRunner:
    """Convenience runner class wrapping ``run_agent_episode`` with shared calibration."""

    def __init__(self, calibration: dict[str, Any] | None = None, max_turns: int = MAX_TURNS):
        self.calibration = calibration
        self.max_turns = max_turns

    def run(self, agent: Any, task_id: str, seed: int, save_dir: Path | None = None) -> dict[str, Any]:
        return run_agent_episode(
            agent,
            task_id,
            seed,
            calibration=self.calibration,
            max_turns=self.max_turns,
            save_dir=save_dir,
        )


def _run_builtin_anchor(policy: str, task_id: str, seed: int, calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Track 1 ``policies.run_policy`` if installed, or deterministic built-in anchors."""
    try:
        policies_mod = importlib.import_module("terraria_bench.policies")
        if hasattr(policies_mod, "run_policy"):
            return policies_mod.run_policy(policy, task_id, seed, calibration=calibration)
    except ImportError:
        pass

    ep = Episode(task_id, seed)
    w = ep.world
    if policy == "do-nothing":
        return ep.finalize(calibration)

    if policy == "absentee":
        while not ep.done:
            ep.step("terraria_status")
            ep.step("end_window")
        return ep.finalize(calibration)

    if policy == "exploiter":
        # Replays closed exploits in window 1 (leaky shelter matrix, place/break placed block, quit early)
        ep.step("build_blueprint", matrix=[[3, 3, 3], [3, 0, 0], [3, 3, 3]], origin_dx=-1, origin_dy=-1)
        ep.step("mine_vein", x=w.pos[0] - 1, y=w.pos[1] - 1, tile_type="any", max_tiles=2)
        ep.step("end_window")
        return ep.finalize(calibration)

    if policy == "blast-list":
        while not ep.done:
            if w.inv.get("wood", 0) < 12 and any(tx not in w.withered for tx in w.trees):
                ep.step("chop_trees", radius=30, max_trees=2)
            elif w.targets():
                ep.step("engage_combat_burst")
            else:
                ep.step("mine_vein", tile_type="stone", max_tiles=4)
            if not ep.done:
                ep.step("end_window")
        return ep.finalize(calibration)

    if policy == "rule-based":
        while not ep.done:
            if "SHELTER" not in w.milestones:
                if w.inv.get("wood", 0) < 40 and any(tx not in w.withered for tx in w.trees):
                    ep.step("chop_trees", radius=40, max_trees=4)
                if not ep.done and w.inv.get("wood", 0) >= 20:
                    ep.step("build_blueprint", preset="shelter")
            if not ep.done and not w.has("wooden_sword") and w.has("wood", 7) and w.near_tile(WORKBENCH, 4):
                ep.step("craft_item", item_id="wooden_sword")
            if not ep.done and w.has("wooden_sword") and w.eq["weapon"] != "wooden_sword":
                ep.step("equip_item", item_id="wooden_sword")
            if not ep.done and not w.has("wood_armor") and w.has("wood", 30) and w.near_tile(WORKBENCH, 4):
                ep.step("craft_item", item_id="wood_armor")
            if not ep.done and w.has("wood_armor") and w.eq["armor"] != "wood_armor":
                ep.step("equip_item", item_id="wood_armor")
            if not ep.done and not w.housing()["valid"] and "SHELTER" in w.milestones:
                ep.step("navigate_to", x=w.home[0], y=w.home[1], budget=300)
            if not ep.done:
                ep.step("end_window")
        return ep.finalize(calibration)

    # smart-triage
    while not ep.done:
        if "SHELTER" not in w.milestones:
            if w.inv.get("wood", 0) < 48 and any(tx not in w.withered for tx in w.trees):
                ep.step("chop_trees", radius=45, max_trees=4)
            if not ep.done and w.inv.get("wood", 0) >= 20:
                ep.step("build_blueprint", preset="shelter")
        if not ep.done and w.tech_tier < 1:
            if w.inv.get("wood", 0) < 40 and any(tx not in w.withered for tx in w.trees):
                ep.step("chop_trees", radius=45, max_trees=3)
            if not ep.done and not w.has("wooden_sword") and w.has("wood", 7) and w.near_tile(WORKBENCH, 4):
                ep.step("craft_item", item_id="wooden_sword")
            if not ep.done and w.has("wooden_sword") and w.eq["weapon"] != "wooden_sword":
                ep.step("equip_item", item_id="wooden_sword")
            if not ep.done and not w.has("wood_armor") and w.has("wood", 30) and w.near_tile(WORKBENCH, 4):
                ep.step("craft_item", item_id="wood_armor")
            if not ep.done and w.has("wood_armor") and w.eq["armor"] != "wood_armor":
                ep.step("equip_item", item_id="wood_armor")

        if not ep.done and "arena" in ep.task.objectives and "ARENA" not in w.milestones and ep.ops_left > 3:
            if w.inv.get("wood", 0) < 35 and any(tx not in w.withered for tx in w.trees):
                ep.step("chop_trees", radius=45, max_trees=3)
            if not ep.done and w.inv.get("wood", 0) >= 20:
                ep.step("build_blueprint", preset="arena")

        if not ep.done and "boss:wall_of_flesh" in ep.task.objectives:
            while not ep.done and ep.ops_left > 2 and w.pos[1] < 88:
                ep.step("explore_cavern", target_depth=90, auto_torch=True)
            if not ep.done and w.pos[1] >= 88 and "HELLBRIDGE" not in w.milestones:
                ep.step("build_blueprint", preset="hellbridge")
            if not ep.done and w.pos[1] >= 88 and w.boss is None and w.has("guide_voodoo_doll") and ep.ops_left >= 6:
                ep.step("summon_or_trigger_boss", boss="wall_of_flesh")

        if not ep.done and "depth" in ep.task.objectives and w.max_depth < 55 and ep.ops_left > 5:
            ep.step("explore_cavern", target_depth=58, auto_torch=True)
        if not ep.done and "crystal" in ep.task.objectives and w.crystals < 1 and ep.ops_left > 4:
            ep.step("loot_container_or_crystal", kind="life_crystal")

        # Tier 2 iron progression when gear/ore is valuable and ops allow returning home
        if not ep.done and w.tech_tier < 2 and "boss:wall_of_flesh" not in ep.task.objectives and ep.ops_left > 5:
            if "FURNACE" not in w.milestones and not w.has("furnace"):
                if w.inv.get("stone", 0) < 20:
                    ep.step("mine_vein", tile_type="stone", max_tiles=12)
                if not ep.done and w.inv.get("torch", 0) < 3 and w.has("wood", 1) and w.has("gel", 1):
                    ep.step("craft_item", item_id="torch", quantity=1)
                if not ep.done and w.has("stone", 20) and w.has("wood", 4) and w.has("torch", 3):
                    if not w.near_tile(WORKBENCH, 4):
                        ep.step("navigate_to", x=w.home[0], y=w.home[1], budget=300)
                    if not ep.done and w.near_tile(WORKBENCH, 4):
                        ep.step("craft_item", item_id="furnace")
            if not ep.done and "FURNACE" not in w.milestones and w.has("furnace"):
                if not w.housing()["valid"]:
                    ep.step("navigate_to", x=w.home[0], y=w.home[1], budget=300)
                if not ep.done:
                    ep.step("place_station_or_furniture", item_id="furnace", dx=1, dy=0)
            if not ep.done and w.inv.get("iron_bar", 0) + w.inv.get("iron_ore", 0) // 2 < 10 and ep.ops_left > 5:
                ep.step("mine_vein", tile_type="iron_ore", max_tiles=10)
            if not ep.done and w.inv.get("iron_ore", 0) >= 2 and "FURNACE" in w.milestones and ep.ops_left > 4:
                if not w.near_tile(WORKBENCH, 4):
                    ep.step("navigate_to", x=w.home[0], y=w.home[1], budget=300)
                if not ep.done:
                    ep.step("craft_item", item_id="iron_bar", quantity=w.inv["iron_ore"] // 2)
            if not ep.done and "ANVIL" not in w.milestones and w.has("iron_bar", 3) and w.near_tile(WORKBENCH, 4):
                if not w.has("anvil"):
                    ep.step("craft_item", item_id="anvil")
                if not ep.done and w.has("anvil"):
                    ep.step("place_station_or_furniture", item_id="anvil", dx=-1, dy=0)
            if not ep.done and "ANVIL" in w.milestones:
                if not w.has("iron_broadsword") and w.has("iron_bar", 4):
                    ep.step("craft_item", item_id="iron_broadsword")
                if not ep.done and w.has("iron_broadsword") and w.eq["weapon"] not in ("iron_broadsword", "gold_bow"):
                    ep.step("equip_item", item_id="iron_broadsword")
                if not ep.done and not w.has("iron_armor") and w.has("iron_bar", 6):
                    ep.step("craft_item", item_id="iron_armor")
                if not ep.done and w.has("iron_armor") and w.eq["armor"] != "iron_armor":
                    ep.step("equip_item", item_id="iron_armor")

        # Boss combat: use ranged bow + arena tactics if available
        if not ep.done and w.boss is not None:
            if w.arena()["valid"] and not w.on_arena() and w.boss["kind"] == "eye" and ep.ops_left > 2:
                ep.step("deploy_combat_arena_tactics", tactic="grapple_tier")
            ranged_weapon = next((k for k in ("gold_bow", "iron_bow", "wooden_bow") if w.has(k)), None)
            if ranged_weapon and w.has("wooden_arrow", 10) and w.eq["weapon"] != ranged_weapon and ep.ops_left > 2:
                ep.step("equip_item", item_id=ranged_weapon)
            while not ep.done and ep.ops_left > 2 and w.boss is not None:
                ep.step("engage_combat_burst", kite_distance=16.0, dodge_priority=0.6, vertical_platform_use=True)
                if not ep.done and w.hp < w.max_hp * 0.5 and w.has("lesser_healing_potion") and w.abs_tick >= w.sick_until:
                    ep.step("consume_buff_or_heal", item_id="lesser_healing_potion")

        while not ep.done and ep.ops_left > 1 and not w.housing()["valid"] and "SHELTER" in w.milestones and w.boss is None and w.pos != w.home:
            ep.step("navigate_to", x=w.home[0], y=w.home[1], budget=300)
        if not ep.done:
            ep.step("end_window")
    return ep.finalize(calibration)


def _reference_calibration() -> dict[str, Any]:
    if CALIBRATION_PATH.exists():
        try:
            cal = load_sealed(CALIBRATION_PATH)
            if cal.get("cells"):
                return cal
        except ProvenanceError:
            pass

    cells: dict[str, Any] = {}
    for task_id in EVAL_TASKS:
        for seed in REPORTED_SEEDS:
            ledgers = {p: _run_builtin_anchor(p, task_id, seed) for p in ANCHORS}
            comps = {p: components(ledgers[p]) for p in ANCHORS}
            z = comps["absentee"]["delta_net"]
            best = max(comps[p]["delta_net"] for p in ("blast-list", "rule-based", "smart-triage"))
            s = max(1.0, round(best - z, 6))
            raw_cell = {"z": z, "s": s}
            scores = {p: grade(ledgers[p], raw_cell)["composite"] for p in ANCHORS}
            thresh = round(0.5 * (scores["rule-based"] + scores["smart-triage"]), 4)
            if abs(scores["smart-triage"] - scores["rule-based"]) < 0.04:
                thresh = round(max(scores["rule-based"], scores["smart-triage"]) - 0.02, 4)
            cells[f"{task_id}:{seed}"] = {
                "task": task_id,
                "seed": seed,
                "z": z,
                "s": s,
                "threshold": thresh,
            }
    return {"cells": cells, "source": "harness_reference"}


def _aggregate_graded(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "episodes": 0, "composite": None, "pass_rate": None,
            "business": None, "reliability": None, "continuity": None, "eye_progress": None,
        }
    passed = [r["passed"] for r in rows if "passed" in r]
    eye_vals = [r.get("eye_progress") for r in rows if r.get("eye_progress") is not None]
    return {
        "episodes": n,
        "composite": round(sum(r["composite"] for r in rows) / n, 4),
        "pass_rate": round(sum(1 for p in passed if p) / len(passed), 4) if passed else None,
        "business": round(sum(r["business"] for r in rows) / n, 4),
        "reliability": round(sum(r["reliability"] for r in rows) / n, 4),
        "continuity": round(sum(r["continuity"] for r in rows) / n, 4),
        "survival": round(sum(r["survival"] for r in rows) / n, 4),
        "sla": round(sum(r["sla"] for r in rows) / n, 4),
        "windows": round(sum(r["windows"] for r in rows) / n, 2),
        "eye_progress": round(sum(eye_vals) / len(eye_vals), 4) if eye_vals else None,
    }


def _load_saved_runs(agent_name: str, cal: dict[str, Any]) -> list[dict[str, Any]]:
    folder = ARTIFACTS_DIR / "runs" / agent_name
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ledger = obj.get("ledger", obj) if isinstance(obj, dict) else None
        if not isinstance(ledger, dict) or not verify_seal(ledger):
            continue
        cell = cal.get("cells", {}).get(f"{ledger.get('task')}:{ledger.get('seed')}")
        if cell and cell.get("s", 0) > 0:
            out.append(grade(ledger, cell))
    return out


def _print_leaderboards(table1: list[dict[str, Any]], table3: list[dict[str, Any]]) -> None:
    def _fmt(v: Any, pct: bool = False) -> str:
        if v is None:
            return "-"
        return f"{v * 100:.0f}%" if pct else f"{v:.3f}"

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        t1 = Table(title="TerrariaBench StoreBench Table 1 — Main Evaluation Leaderboard")
        for col in ("Policy / Model", "Status", "Composite", "Pass", "Business", "Reliab.", "Contin."):
            t1.add_column(col)
        for row in table1:
            t1.add_row(
                row["name"],
                row["status"],
                _fmt(row["composite"]),
                _fmt(row["pass_rate"], pct=True),
                _fmt(row["business"]),
                _fmt(row["reliability"]),
                _fmt(row["continuity"]),
            )
        console.print(t1)

        t3 = Table(title="TerrariaBench StoreBench Table 3 — Post-Training Transfer (Gemma)")
        for col in ("Policy", "Status", "Composite", "Pass", "Business", "Reliab.", "Contin.", "Survival", "SLA", "Windows"):
            t3.add_column(col)
        for row in table3:
            t3.add_row(
                row["name"],
                row["status"],
                _fmt(row["composite"]),
                _fmt(row["pass_rate"], pct=True),
                _fmt(row["business"]),
                _fmt(row["reliability"]),
                _fmt(row["continuity"]),
                _fmt(row.get("survival")),
                _fmt(row.get("sla")),
                str(row.get("windows") if row.get("windows") is not None else "-"),
            )
        console.print(t3)
        return
    except ImportError:
        pass

    print("\n=== StoreBench Table 1 — Main Evaluation Leaderboard ===")
    print(f"{'Policy / Model':20s} {'Status':14s} {'Composite':>9s} {'Pass':>7s} {'Business':>9s} {'Reliab.':>8s} {'Contin.':>8s}")
    print("-" * 80)
    for row in table1:
        print(
            f"{row['name']:20s} {row['status']:14s} {_fmt(row['composite']):>9s} "
            f"{_fmt(row['pass_rate'], pct=True):>7s} {_fmt(row['business']):>9s} "
            f"{_fmt(row['reliability']):>8s} {_fmt(row['continuity']):>8s}"
        )
    print("\n=== StoreBench Table 3 — Post-Training Transfer (Gemma) ===")
    print(f"{'Policy':20s} {'Status':14s} {'Composite':>9s} {'Pass':>7s} {'Business':>9s} {'Reliab.':>8s} {'Contin.':>8s}")
    print("-" * 80)
    for row in table3:
        print(
            f"{row['name']:20s} {row['status']:14s} {_fmt(row['composite']):>9s} "
            f"{_fmt(row['pass_rate'], pct=True):>7s} {_fmt(row['business']):>9s} "
            f"{_fmt(row['reliability']):>8s} {_fmt(row['continuity']):>8s}"
        )


def _build_agent(agent_name: str, live: bool) -> Any:
    load_dotenv_if_present()
    if agent_name == "gemini":
        return GeminiAgent(live=live)
    if agent_name == "nemotron":
        return NemotronAgent(live=live)
    verda_models = {
        "verda_deepseek": ("TERRARIA_VERDA_DEEPSEEK_MODEL", DEFAULT_VERDA_DEEPSEEK_MODEL),
        "verda_kimi": ("TERRARIA_VERDA_KIMI_MODEL", DEFAULT_VERDA_KIMI_MODEL),
    }
    if agent_name in verda_models:
        if live and not os.environ.get("VERDA_INFERENCE_API_KEY"):
            raise ValueError("VERDA_INFERENCE_API_KEY is required for live Verda model runs")
        model_env, default_model = verda_models[agent_name]
        agent = NemotronAgent(
            model=os.environ.get(model_env, default_model),
            base_url=os.environ.get("VERDA_INFERENCE_BASE_URL", VERDA_INFERENCE_BASE_URL),
            live=live,
        )
        agent.name = agent_name
        return agent
    if agent_name in ("gemma_base", "gemma_tuned"):
        # Imported here: the Gemma agent needs torch, which the scripted evaluation must not require.
        from .gemma_agent import AgentConfig as GemmaConfig, GemmaAgent
    if agent_name == "gemma_base":
        return GemmaAgent(config=GemmaConfig(), live=live)
    if agent_name == "gemma_tuned":
        # A run may only be recorded as "tuned" when tuned weights are really in use: a local adapter,
        # or an endpoint that serves one (see post_training/modal_grpo_gemma.py::serve_vllm).
        ckpt = ARTIFACTS_DIR / "checkpoints" / "gemma_grpo_final"
        served = os.environ.get("GEMMA_TUNED_BASE_URL", "")
        if ckpt.exists():
            agent = GemmaAgent(config=GemmaConfig(adapter_path=str(ckpt)), live=live)
        elif served:
            cfg = GemmaConfig(model_name_or_path=os.environ.get("GEMMA_TUNED_MODEL", "gemma-terraria-bc"))
            agent = GemmaAgent(config=cfg, live=live, base_url=served)
        elif live:
            raise ValueError(
                "no tuned Gemma to run: set GEMMA_TUNED_BASE_URL to an endpoint serving the adapter, "
                f"or place a LoRA checkpoint at {ckpt}")
        else:
            agent = GemmaAgent(config=GemmaConfig(), live=False)
        agent.name = "gemma_tuned"
        return agent
    raise ValueError(f"unknown agent '{agent_name}'")


def run_eval_standard(live: bool = False) -> dict[str, Any]:
    """Execute StoreBench judging protocol over EVAL_TASKS x REPORTED_SEEDS."""
    load_dotenv_if_present()
    orig_git_commit = provenance.git_commit
    provenance.git_commit = functools.lru_cache(maxsize=1)(orig_git_commit)
    try:
        cal = _reference_calibration()
        table1: list[dict[str, Any]] = []

        for policy in ANCHORS:
            rows = []
            for task_id in EVAL_TASKS:
                for seed in REPORTED_SEEDS:
                    ledger = _run_builtin_anchor(policy, task_id, seed, calibration=cal)
                    rows.append(grade(ledger, cal["cells"][f"{task_id}:{seed}"]))
            agg = _aggregate_graded(rows)
            table1.append({"name": policy, "status": "anchor", **agg})

        if live:
            if os.environ.get("GEMINI_API_KEY"):
                gem = _build_agent("gemini", True)
                for task_id in EVAL_TASKS:
                    for seed in REPORTED_SEEDS:
                        run_agent_episode(gem, task_id, seed, calibration=cal, save_dir=ARTIFACTS_DIR / "runs" / "gemini")
            if any(os.environ.get(k) for k in ("VERDA_INFERENCE_KEY", "VERDA_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY")):
                nem = _build_agent("nemotron", True)
                for task_id in EVAL_TASKS:
                    for seed in REPORTED_SEEDS:
                        run_agent_episode(nem, task_id, seed, calibration=cal, save_dir=ARTIFACTS_DIR / "runs" / "nemotron")
            if os.environ.get("VERDA_INFERENCE_API_KEY"):
                for model_key in ("verda_deepseek", "verda_kimi"):
                    verda = _build_agent(model_key, True)
                    for task_id in EVAL_TASKS:
                        for seed in REPORTED_SEEDS:
                            run_agent_episode(
                                verda, task_id, seed, calibration=cal,
                                save_dir=ARTIFACTS_DIR / "runs" / model_key,
                            )
            if os.environ.get("FEATHERLESS_API_KEY") or os.environ.get("MODAL_BASE_URL"):
                gb = _build_agent("gemma_base", True)
                try:
                    gt = _build_agent("gemma_tuned", True)
                except ValueError as exc:
                    gt = None
                    print(f"Gemma Tuned skipped: {exc}", file=sys.stderr)
                for task_id in EVAL_TASKS:
                    for seed in REPORTED_SEEDS:
                        run_agent_episode(gb, task_id, seed, calibration=cal, save_dir=ARTIFACTS_DIR / "runs" / "gemma_base")
                        if gt is not None:
                            run_agent_episode(gt, task_id, seed, calibration=cal, save_dir=ARTIFACTS_DIR / "runs" / "gemma_tuned")
    finally:
        provenance.git_commit = orig_git_commit

    for model_key, label in (
        ("human", "Human experts"),
        ("gemini", "Gemini 4 / 2.5"),
        ("nemotron", "Nemotron 3.5 VL"),
        ("verda_deepseek", "Verda DeepSeek V4.1 Flash"),
        ("verda_kimi", "Verda Kimi K3.1 Pro"),
        ("gemma_base", "Gemma Base"),
        ("gemma_tuned", "Gemma Tuned"),
    ):
        saved = _load_saved_runs(model_key, cal)
        agg = _aggregate_graded(saved)
        table1.append({"name": label, "status": "evaluated" if saved else "not run", **agg})

    table3 = [
        {"name": "Gemma Base", "status": "evaluated" if _load_saved_runs("gemma_base", cal) else "not run", **_aggregate_graded(_load_saved_runs("gemma_base", cal))},
        {"name": "Gemma Tuned", "status": "evaluated" if _load_saved_runs("gemma_tuned", cal) else "not run", **_aggregate_graded(_load_saved_runs("gemma_tuned", cal))},
    ]
    _print_leaderboards(table1, table3)
    report = {"table1": table1, "table3": table3, "calibration_source": cal.get("source", "sealed")}
    if live:
        out_path = ARTIFACTS_DIR / "runs" / "eval_standard.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def play_episode(
    agent_name: str = "gemini",
    task_id: str = "first-night",
    seed: int = 7,
    live: bool = False,
    max_turns: int = MAX_TURNS,
    save: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Play one episode with a named agent, printing each turn. Returns the run dict."""
    agent = _build_agent(agent_name, live=live)
    try:
        cal: dict[str, Any] | None = _reference_calibration()
    except Exception:
        cal = None  # stale or missing calibration: report raw components instead of a graded score

    def show(turn: int, tool: str, args: dict[str, Any], res: dict[str, Any]) -> None:
        shown = {k: ("<matrix>" if k == "matrix" else v) for k, v in args.items()}
        status = "ok" if res.get("ok", True) else f"FAIL {res.get('error', '')}"
        print(f"[{turn:3d}] {tool} {json.dumps(shown, default=str)} -> {status}\n      {res.get('header', '')}")

    mode = "live" if live else "offline scripted stand-in (no model calls)"
    print(f"== {agent_name} | {task_id} seed {seed} | {mode} ==")
    if live and agent_name == "gemini":
        print(f"supervisor={agent.model} decision={agent.decisions.model}")
    elif live and (agent_name == "nemotron" or agent_name.startswith("verda_")):
        print(f"provider={agent.provider} base_url={agent.base_url} model={agent.model}")
    elif live and agent_name.startswith("gemma"):
        print(f"provider={agent.provider} base_url={agent.base_url} model={agent.config.model_name_or_path}")
    save_dir = (ARTIFACTS_DIR / "runs" / agent_name) if (save and live) else None
    out = run_agent_episode(agent, task_id, seed, calibration=cal, max_turns=max_turns,
                            save_dir=save_dir, on_turn=show if verbose else None)
    score = out["score"]
    print(f"turns={out['turns']} windows={len(out['ledger']['windows'])}/{TASKS[task_id].windows} "
          f"milestones={sorted(out['ledger'].get('milestones', []))}")
    print("score: " + json.dumps({k: v for k, v in score.items() if isinstance(v, (int, float, bool))}, sort_keys=True))
    if getattr(agent, "stats", None):
        print("agent stats: " + json.dumps(agent.stats, sort_keys=True))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerrariaBench Multi-Model Evaluation Harness")
    parser.add_argument("--eval-standard", action="store_true", help="Run StoreBench standard evaluation protocol")
    parser.add_argument("--live", action="store_true", help="Invoke live API models when API keys are set")
    parser.add_argument("--play", choices=["gemini", "nemotron", "verda_deepseek", "verda_kimi", "gemma_base", "gemma_tuned"], help="Play one episode with this agent and print each turn")
    parser.add_argument("--task", default="first-night", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--save", action="store_true", help="With --play --live: write the sealed ledger to artifacts/runs/<agent>")
    parser.add_argument("--quiet", action="store_true", help="With --play: print only the summary")
    args = parser.parse_args(argv)

    if args.play:
        try:
            play_episode(args.play, args.task, args.seed, live=args.live, max_turns=args.max_turns,
                         save=args.save, verbose=not args.quiet)
        except GeminiLiveError as exc:
            print(f"live Gemini run aborted: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.eval_standard:
        run_eval_standard(live=args.live)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
