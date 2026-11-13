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
            "sla": 0.0, "windows": 0, "delta_net": 0.0, "deaths": 0, "valid": False, "reason": reason}


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
    return {
        "delta_net": round(ledger["phi_final"] - ledger["phi0"], 6),
        "survival": credit / total,
        "sla": (sum(1 for m in checks if m) / len(task.sla)) if task.sla else 0.0,
        "continuity": attended / total,
        "windows": attended,
        "deaths": sum(r["deaths"] for r in recs),
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
           "windows": c["windows"], "delta_net": c["delta_net"], "deaths": c["deaths"], "valid": True, "reason": ""}
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
