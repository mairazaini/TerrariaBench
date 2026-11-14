"""Scoring (StoreBench Eq. 1, adapted).

    R = 0.7 * clip[0,1]((dNet - z) / s)
      + 0.2 * (0.5 * survival_rate + 0.5 * objective_sla_rate)
      + 0.1 * (windows_attended / windows_in_task)

dNet is the growth of Net Progression Value over the episode. ``z`` is the
absentee anchor's growth and ``s`` the gap from z to the best honest anchor in
the same task-seed cell; both come from the sealed calibration artifact. The
grader reads only the sealed episode ledger and fails closed: a missing,
corrupt or mismatched ledger grades to zero.
"""
from __future__ import annotations

import re
from typing import Any

from .provenance import verify_seal
from .tasks import TASKS

W_BUSINESS, W_RELIABILITY, W_CONTINUITY = 0.7, 0.2, 0.1
PROGRESS_STEP = 2.0   # a window "made progress" if the Phi high-water mark rose by this much
IDLE_GRACE = 2        # idle windows tolerated before survival credit decays
DENSE_CLIP = (-2.0, 1.0)

# The exploit catalogue. Each entry has exactly one pinning regression test in
# tests/test_reward_exploits.py, and the exploiter anchor replays the playable
# ones at every calibration.
EXPLOITS = (
    ("continuity_for_quitters", "Attend a few windows, then quit and keep the continuity ratio.",
     "Continuity's denominator is the task's full window count, never the windows played."),
    ("dirt_box_camping", "Seal yourself in and idle to farm survival.",
     "Survival credit needs recent progress and decays with idleness; half of reliability is SLA checkpoints."),
    ("place_break_farming", "Place blocks and mine them back for mining value.",
     "Tiles carry placed-by-player provenance and drop nothing; milestones are unique unlocks."),
    ("slime_grinding", "Farm trivial hostiles for combat progression.",
     "Kill value is capped per family; boss credit is locked to the HP high-water mark."),
    ("boss_resummon_scumming", "Despawn and re-summon a boss to re-earn the opening damage.",
     "Boss credit uses the episode-wide minimum HP ratio per boss type."),
    ("craft_trash_cycling", "Craft, trash and re-craft the same item.",
     "Crafting credit is a set indicator over unique items; trashing only loses salvage value."),
    ("suicide_respawn_teleport", "Die on purpose to teleport home for free.",
     "A death forfeits progression and costs more next-window operations than walking home."),
    ("last_window_dumping", "Start a boss fight at the end and let the episode expire.",
     "Settlement resolves any live boss fight as a loss, including in the final window."),
    ("partial_enclosure_spoofing", "Claim housing with a leaky or doorway-only structure.",
     "Housing is verified by flood fill; any gap, platform hole or open side fails."),
    ("chest_duplicate_looting", "Loot the same container or crystal twice.",
     "Looting consumes the tile; a second attempt finds nothing."),
    ("healing_potion_stalling", "Chain potions to out-heal a fight indefinitely.",
     "Potion Sickness blocks a second potion for 60 seconds of simulated time."),
)


def sla_met(world, kind: str, sheltered: bool) -> bool:
    """Evaluate one hidden checkpoint against the world right after settlement."""
    head, _, rest = kind.partition(":")
    if head == "shelter":
        return bool(sheltered)
    if head == "arena":
        return "ARENA" in world.milestones
    if head == "tier":
        return world.tech_tier >= int(rest)
    if head == "depth":
        return world.max_depth >= int(rest)
    if head == "crystal":
        return world.crystals >= int(rest)
    if head == "def":
        from .engine import ITEMS
        return bool(world.eq["armor"]) and ITEMS[world.eq["armor"]]["defense"] >= int(rest)
    if head == "milestone":
        return rest in world.milestones
    if head == "boss":
        boss, _, thr = rest.rpartition(":")
        return world.boss_min.get(boss, 1.0) <= float(thr)
    if head == "kill":
        return f"KILL:{rest}" in world.milestones
    raise ValueError(f"unknown SLA kind: {kind}")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def zero(reason: str) -> dict:
    return {"composite": 0.0, "business": 0.0, "reliability": 0.0, "continuity": 0.0, "survival": 0.0,
            "sla": 0.0, "windows": 0, "delta_net": 0.0, "deaths": 0,
            "eye_progress": 0.0, "eye_progress_pct": 0.0, "valid": False, "reason": reason}


def compute_eye_progress(state: dict[str, Any]) -> dict[str, Any]:
    """Compute a normalized [0.0, 1.0] (0%–100%) Eye of Cthulhu progression score.

    Weights (monotonic high-water marks prevent resummon or drop/re-craft exploits):
    - health_score (10%): Max HP progression via Life Crystals (100 -> 180+ HP).
    - armor_score (10%): Equipped armor and total defense rating (0 -> 8+ defense).
    - tasks_score (35%): 9 tech-tree milestones toward Eye of Cthulhu
      (wood, workbench, shelter, weapons, arrows, stone/furnace, anvil/metal, lenses/altar, arena/summon).
    - boss_fight_score (25%): damaged the Eye of Cthulhu (5%) + lowest Eye HP ratio achieved (up to 20%).
      The Eye appearing without taking damage scores nothing.
    - eye_kill_score (20%, and locks total score to 100% / 1.0): Eye of Cthulhu defeated.
    """
    ms = {str(m) for m in (state.get("milestones") or [])}
    # A sealed simulator ledger carries exact inputs in ``eye``. They are measured as gains over the
    # task's starting kit, so a scenario that starts the player with 300 HP or gold armor credits only
    # what the agent added. Live sessions pass the same fields flat (EyeProgressTracker.as_dict).
    sealed = state.get("eye") if isinstance(state.get("eye"), dict) else None
    if sealed is not None:
        state = {
            **state,
            "max_hp": 100.0 + max(0.0, float(sealed.get("max_hp", 100)) - float(sealed.get("start_max_hp", 100))),
            "defense": max(0.0, float(sealed.get("armor_peak", 0)) - float(sealed.get("armor_start", 0))),
            "eye_fought": bool(sealed.get("fought")),
            "eye_min_hp_ratio": sealed.get("min_hp_ratio", 1.0),
        }

    # 1. Health / Life Crystals (up to 0.10)
    max_hp = float(state.get("max_hp") or 100.0)
    crystals = int(state.get("life_crystals") or state.get("crystals") or 0)
    if crystals == 0 and any(m in ms for m in ("MINE:crystal", "USE:crystal")):
        crystals = sum(1 for m in ms if m in ("MINE:crystal", "USE:crystal"))
    hp_frac = _clip((max_hp - 100.0) / 80.0)
    if hp_frac > 0.0:
        health_score = 0.10 * hp_frac
    elif crystals > 0:
        health_score = min(0.09, 0.03 * crystals)
    else:
        health_score = 0.0

    # 2. Armor & Defense (up to 0.10)
    defense = float(state.get("defense") or 0.0)
    armor_pieces = int(state.get("armor_pieces") or 0)
    if defense == 0.0 and sealed is None:  # legacy ledgers only: infer from what was crafted
        if any(m.startswith("CRAFT:") and any(k in m for k in ("gold_armor", "platinum_armor", "silver_armor", "tungsten_armor", "iron_armor", "lead_armor")) for m in ms):
            defense = 8.0
        elif "CRAFT:wood_armor" in ms:
            defense = 3.0
    armor_score = 0.10 * _clip(max(defense / 8.0, min(3, armor_pieces) / 3.0 * 0.5))

    # 3. Tasks Completed Along the Way (up to 0.35)
    wood = float(state.get("wood") or 0.0)
    arrows = float(state.get("arrows") or (150.0 if ("CRAFT:wooden_arrow" in ms or "CRAFT:flaming_arrow" in ms) else 0.0))
    lenses = float(state.get("lenses") or (6.0 if ("CRAFT:suspicious_looking_eye" in ms or state.get("has_summon_item")) else 0.0))

    has_workbench = bool(state.get("workbench") or "WORKBENCH" in ms or "CRAFT:workbench" in ms)
    has_shelter = bool(state.get("shelter") or "SHELTER" in ms)
    has_wood_gathered = bool(state.get("wood_gathered") or wood >= 40.0 or has_workbench or has_shelter)
    has_weapons = bool(
        state.get("weapons")
        or any(m.startswith("CRAFT:") and ("bow" in m or "sword" in m) for m in ms)
    )
    has_stone_or_ore = bool(
        state.get("stone_or_ore") or any(m.startswith("MINE:") for m in ms)
        or "FURNACE" in ms or "CAVERN" in ms or any(m.startswith("CRAFT:") and m.endswith("_bar") for m in ms)
    )
    has_furnace = bool(
        state.get("furnace") or "FURNACE" in ms or "CRAFT:furnace" in ms
        or any("bar" in m.lower() for m in ms)
    )
    has_anvil = bool(state.get("anvil") or "ANVIL" in ms or "CRAFT:anvil" in ms)
    has_metal_tier = bool(
        state.get("metal_bow")
        or any(
            m.startswith("CRAFT:") and any(k in m for k in ("copper_", "tin_", "iron_", "lead_", "silver_", "tungsten_", "gold_", "platinum_"))
            and any(g in m for g in ("bow", "sword", "armor"))
            for m in ms
        )
    )
    eye_killed = bool(state.get("eye_killed") or "KILL:eye" in ms)
    raw_min_ratio = state.get("eye_min_hp_ratio", 1.0)
    eye_min_hp_ratio = 0.0 if eye_killed else _clip(float(1.0 if raw_min_ratio is None else raw_min_ratio))
    # "Fought" means the Eye took damage. The Eye merely appearing (it can spawn on its own) is not
    # progress, so an idle player standing under it earns nothing here.
    eye_fought = bool(eye_killed or eye_min_hp_ratio < 1.0)

    has_altar = bool(state.get("altar_found") or state.get("has_summon_item") or "CRAFT:suspicious_looking_eye" in ms or eye_killed)
    has_arena = bool(state.get("arena_built") or "ARENA" in ms)
    has_summon = bool(state.get("has_summon_item") or "CRAFT:suspicious_looking_eye" in ms or eye_killed)

    checklist = {
        "wood_gathered": has_wood_gathered,
        "workbench": has_workbench,
        "shelter": has_shelter,
        "weapons": has_weapons,
        "ammunition_ready": arrows >= 150.0,
        "stone_or_ore_mined": has_stone_or_ore,
        "furnace_smelting": has_furnace,
        "anvil_placed": has_anvil,
        "metal_gear": has_metal_tier,
        "lenses_collected": lenses >= 6.0,
        "altar_found": has_altar,
        "arena_built": has_arena,
        "summon_ready": has_summon,
    }
    tasks_score = (
        (0.03 if has_wood_gathered else 0.0)
        + (0.04 if has_workbench else 0.0)
        + (0.04 if has_shelter else 0.0)
        + (0.04 if has_weapons else 0.0)
        + 0.04 * _clip(arrows / 150.0)
        + (0.02 if has_stone_or_ore else 0.0)
        + (0.02 if has_furnace else 0.0)
        + (0.02 if has_anvil else 0.0)
        + (0.02 if has_metal_tier else 0.0)
        + 0.02 * _clip(lenses / 6.0)
        + (0.02 if has_altar else 0.0)
        + (0.02 if has_arena else 0.0)
        + (0.02 if has_summon else 0.0)
    )

    # 4. Boss Fight & Damage Dealt to Eye of Cthulhu (up to 0.25)
    boss_fight_score = (0.05 + 0.20 * (1.0 - eye_min_hp_ratio)) if eye_fought else 0.0

    # 5. Eye of Cthulhu Defeated -> 100% (1.0)
    if eye_killed:
        eye_kill_score = 0.20
        total_score = 1.0
    else:
        eye_kill_score = 0.0
        total_score = round(min(0.80, health_score + armor_score + tasks_score + boss_fight_score), 4)

    return {
        "eye_progress_score": total_score,
        "eye_progress_pct": round(total_score * 100.0, 2),
        "health_score": round(health_score, 4),
        "armor_score": round(armor_score, 4),
        "tasks_score": round(tasks_score, 4),
        "boss_fight_score": round(boss_fight_score, 4),
        "eye_kill_score": round(eye_kill_score, 4),
        "eye_fought": eye_fought,
        "eye_min_hp_ratio": round(eye_min_hp_ratio, 4),
        "eye_killed": eye_killed,
        "max_hp": int(round(max_hp)),
        "defense": int(round(defense)),
        "task_checklist": checklist,
    }


class EyeProgressTracker:
    """Monotonic high-water progress tracker for live Terraria sessions."""

    def __init__(self) -> None:
        self.max_hp: float = 100.0
        self.defense: float = 0.0
        self.armor_pieces: int = 0
        self.life_crystals: int = 0
        self.wood: float = 0.0
        self.arrows: float = 0.0
        self.lenses: float = 0.0
        self.wood_gathered: bool = False
        self.workbench: bool = False
        self.shelter: bool = False
        self.weapons: bool = False
        self.stone_or_ore: bool = False
        self.furnace: bool = False
        self.anvil: bool = False
        self.metal_bow: bool = False
        self.altar_found: bool = False
        self.arena_built: bool = False
        self.has_summon_item: bool = False
        self.eye_fought: bool = False
        self.eye_min_hp_ratio: float = 1.0
        self.eye_killed: bool = False

    def update(
        self,
        hud: dict[str, Any],
        referee: dict[str, Any] | None = None,
        planner_mem: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ref = referee if isinstance(referee, dict) else {}
        mem = planner_mem if isinstance(planner_mem, dict) else {}
        inv: dict[str, int] = {}
        for source in (hud.get("backpack"), ref.get("inventory")):
            if isinstance(source, dict):
                for k, v in source.items():
                    if isinstance(v, (int, float)):
                        inv[str(k)] = max(inv.get(str(k), 0), int(v))
        for entry in hud.get("hotbar") or []:
            if isinstance(entry, dict) and entry.get("name"):
                name = str(entry["name"])
                inv[name] = max(inv.get(name, 0), int(entry.get("stack") or 0))

        self.max_hp = max(self.max_hp, float(hud.get("max_hp") or 100.0))
        raw_def = ref.get("defense") if ref.get("defense") is not None else hud.get("defense")
        if isinstance(raw_def, dict):
            raw_def = raw_def.get("flat") or raw_def.get("total") or 0
        self.defense = max(self.defense, float(raw_def or 0.0))

        worn = [str(x) for x in (hud.get("equipment_now") or []) if x]
        self.armor_pieces = max(self.armor_pieces, min(3, sum(1 for w in worn if re.search(r"(Helmet|Hat|Mask|Hood|Breastplate|Chainmail|Mail|Shirt|Greaves|Leggings)", w))))
        self.life_crystals = max(self.life_crystals, int(inv.get("Life Crystal", 0)) + max(0, int((self.max_hp - 100.0) // 20)))

        wood_now = float(inv.get("Wood", 0))
        arrows_now = float(sum(v for k, v in inv.items() if "Arrow" in k))
        lenses_now = float(inv.get("Lens", 0))
        self.wood = max(self.wood, wood_now)
        self.arrows = max(self.arrows, arrows_now)
        self.lenses = max(self.lenses, lenses_now)
        if wood_now >= 40:
            self.wood_gathered = True

        stations_mem = mem.get("stations") if isinstance(mem.get("stations"), dict) else {}
        wv = hud.get("world_view") if isinstance(hud.get("world_view"), dict) else {}
        wv_stations = {str((s or {}).get("kind") or "") for s in (wv.get("stations") or []) if isinstance(s, dict)}
        if "Work Bench" in inv or "Work Bench" in stations_mem or "Work Bench" in wv_stations:
            self.workbench = True
        if (wv.get("house_status") or {}).get("valid_room"):
            self.shelter = True
        if any(
            ("Bow" in k or "Broad" in k or "Sword" in k) and k != "Copper Shortsword"
            for k in inv
        ):
            self.weapons = True
        if inv.get("Stone Block", 0) >= 10 or any(k.endswith(" Ore") for k in inv):
            self.stone_or_ore = True
        if "Furnace" in inv or "Furnace" in stations_mem or "Furnace" in wv_stations or any(k.endswith(" Bar") for k in inv):
            self.furnace = True
        if any(k in inv for k in ("Iron Anvil", "Lead Anvil")) or "Anvil" in stations_mem or "Anvil" in wv_stations:
            self.anvil = True
        if any(k.endswith(" Bow") and k != "Wooden Bow" for k in inv):
            self.metal_bow = True
        if mem.get("altars") or wv.get("altars"):
            self.altar_found = True
        if mem.get("arena"):
            self.arena_built = True
        if "Suspicious Looking Eye" in inv or mem.get("summoned"):
            self.has_summon_item = True

        boss = hud.get("boss") if isinstance(hud.get("boss"), dict) else {}
        boss_name = str(boss.get("name") or ref.get("boss_name") or "").lower()
        if "eye" in boss_name or "cthulhu" in boss_name:
            self.eye_fought = True
            b_max = float(boss.get("max_hp") or 2800.0)
            b_hp = float(boss.get("hp") if boss.get("hp") is not None else ref.get("boss_hp", b_max))
            if b_max > 0:
                self.eye_min_hp_ratio = min(self.eye_min_hp_ratio, _clip(b_hp / b_max))

        if ref.get("downed_eye") or mem.get("eye_defeated"):
            self.eye_killed = True
            self.eye_fought = True
            self.eye_min_hp_ratio = 0.0

        return compute_eye_progress(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_hp": self.max_hp,
            "defense": self.defense,
            "armor_pieces": self.armor_pieces,
            "life_crystals": self.life_crystals,
            "wood": self.wood,
            "arrows": self.arrows,
            "lenses": self.lenses,
            "wood_gathered": self.wood_gathered,
            "workbench": self.workbench,
            "shelter": self.shelter,
            "weapons": self.weapons,
            "stone_or_ore": self.stone_or_ore,
            "furnace": self.furnace,
            "anvil": self.anvil,
            "metal_bow": self.metal_bow,
            "altar_found": self.altar_found,
            "arena_built": self.arena_built,
            "has_summon_item": self.has_summon_item,
            "eye_fought": self.eye_fought,
            "eye_min_hp_ratio": self.eye_min_hp_ratio,
            "eye_killed": self.eye_killed,
        }

    def unlocked_milestones(self) -> list[str]:
        res = compute_eye_progress(self.as_dict())
        out = [k.upper() for k, ok in res["task_checklist"].items() if ok]
        if self.life_crystals > 0:
            out.append(f"LIFE_CRYSTALS:{self.life_crystals}")
        if self.defense >= 8:
            out.append("METAL_ARMOR_DEF")
        elif self.armor_pieces > 0:
            out.append("ARMOR_EQUIPPED")
        if res["eye_fought"]:
            out.append("EYE_FOUGHT")
        if self.eye_killed:
            out.append("KILL:eye")
        return out


def components(ledger: dict) -> dict:
    """Everything that does not depend on calibration."""
    task = TASKS[ledger["task"]]
    total = task.windows  # exploit 1: fixed denominator
    recs = ledger["windows"]
    attended = sum(1 for r in recs if r["attended"])
    high, last, credit = ledger["phi0"], None, 0.0
    for r in recs:
        if r["attended"] and r["phi"] >= high + PROGRESS_STEP:
            last = r["w"]
        high = max(high, r["phi"])
        if r["attended"] and r["deaths"] == 0 and last is not None:
            idle = r["w"] - last
            credit += 1.0 if idle <= IDLE_GRACE else 0.5 ** (idle - IDLE_GRACE)
    checks = [met for r in recs for _, met in r["sla"]]
    eye_bp = compute_eye_progress(ledger)
    return {
        "delta_net": round(ledger["phi_final"] - ledger["phi0"], 6),
        "survival": credit / total,
        "sla": (sum(1 for m in checks if m) / len(task.sla)) if task.sla else 0.0,
        "continuity": attended / total,
        "windows": attended,
        "deaths": sum(r["deaths"] for r in recs),
        "eye_progress": eye_bp["eye_progress_score"],
        "eye_progress_pct": eye_bp["eye_progress_pct"],
        "eye_breakdown": eye_bp,
    }


def grade(ledger, cell) -> dict:
    """Composite score for one episode. ``cell`` is the calibration entry for the
    ledger's (task, seed): {"z": ..., "s": ...}."""
    if not isinstance(ledger, dict) or not verify_seal(ledger):
        return zero("missing or corrupt ledger")
    if ledger.get("task") not in TASKS or len(ledger.get("windows", ())) != TASKS[ledger["task"]].windows:
        return zero("ledger is not a finalized episode")
    if not isinstance(cell, dict) or not cell.get("s", 0) > 0:
        return zero("no calibration for this task-seed cell")
    c = components(ledger)
    business = _clip((c["delta_net"] - cell["z"]) / cell["s"])
    reliability = 0.5 * c["survival"] + 0.5 * c["sla"]
    composite = W_BUSINESS * business + W_RELIABILITY * reliability + W_CONTINUITY * c["continuity"]
    out = {"composite": round(composite, 4), "business": round(business, 4), "reliability": round(reliability, 4),
           "continuity": round(c["continuity"], 4), "survival": round(c["survival"], 4), "sla": round(c["sla"], 4),
           "windows": c["windows"], "delta_net": c["delta_net"], "deaths": c["deaths"],
           "eye_progress": c["eye_progress"], "eye_progress_pct": c["eye_progress_pct"],
           "eye_breakdown": c["eye_breakdown"],
           "valid": True, "reason": ""}
    if "threshold" in cell:
        out["passed"] = composite >= cell["threshold"]
    return out


def dense_rewards(ledger: dict, cell: dict) -> list:
    """Per-window training signal: Phi(s_t) - Phi(s_{t-1}) normalized by the
    cell scale. Unclipped, the terms telescope exactly to (dNet / s); the
    training channel clips each term to [-2, 1]."""
    out, prev = [], ledger["phi0"]
    for r in ledger["windows"]:
        out.append(_clip((r["phi"] - prev) / cell["s"], *DENSE_CLIP))
        prev = r["phi"]
    return out


def training_reward(ledger: dict, cell: dict) -> float:
    """Scalar episode reward for GRPO: normalized business clipped to [-2, 1]."""
    return _clip((ledger["phi_final"] - ledger["phi0"] - cell["z"]) / cell["s"], *DENSE_CLIP)
