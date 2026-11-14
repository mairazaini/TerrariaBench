"""The six scripted anchor policies (StoreBench §4 calibration anchors).

Every anchor acts *only* through ``Episode.step(tool, **kwargs)`` and returns
the sealed ledger from ``Episode.finalize``. Anchors read the world state the
way an operator reads the status header and inventory; they never mutate the
world directly. All anchors are deterministic: the same (task, seed) always
produces a byte-identical ledger.

    do-nothing    calls no tools                                 -> composite 0
    absentee      status + end_window every window               -> defines z
    blast-list    high-volume, mindless chopping/mining/swinging -> low
    rule-based    rigid, task-agnostic checklist clerk           -> judgment anchor
    smart-triage  task-, alert- and state-aware operator         -> judgment anchor
    exploiter     replays the closed exploit catalogue, quits    -> < 0.05
"""
from __future__ import annotations

from .engine import ANVIL, CAVERN_Y, CRYSTAL, FURNACE, ITEMS, PLACEABLE, UNDERWORLD_Y
from .tools import Episode

ANCHORS = ("do-nothing", "absentee", "blast-list", "rule-based", "smart-triage", "exploiter")
HONEST_ANCHORS = ("blast-list", "rule-based", "smart-triage")

# Bars needed for a full metal kit: weapon 4 + armor 6 + pickaxe 5.
_KIT_BARS = {"broadsword": 4, "bow": 4, "armor": 6, "pickaxe": 5}


# --------------------------------------------------------------------- pilot
class Pilot:
    """Thin read-only view over an Episode plus the operator's bookkeeping.

    ``act`` is the only path to the world. Reads (``w``) mirror what the status
    header, inventory and housing tools expose to any agent."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.w = ep.world

    # ---- state reads
    @property
    def done(self) -> bool:
        return self.ep.done

    @property
    def ops(self) -> int:
        return self.ep.ops_left

    @property
    def window(self) -> int:
        return self.w.window

    def inv(self, item: str) -> int:
        return self.w.inv.get(item, 0)

    def at_home(self) -> bool:
        return self.w.pos == self.w.home

    def home_cost(self) -> int:
        """Conservative operations needed to walk home and close the window:
        climbing costs ~10 ticks a tile, walking ~5, and a navigate call buys
        300 ticks; tunnels wander, so pad the estimate by half."""
        if self.at_home():
            return 1
        (x, y), (hx, hy) = self.w.pos, self.w.home
        ticks = 5 * abs(x - hx) + 10 * max(0, y - hy) + 5 * max(0, hy - y)
        return 1 + -(-int(ticks * 1.5) // 300)

    @property
    def hurt(self) -> bool:
        return self.w.hp < 0.45 * self.w.max_hp

    def room(self, need: int = 1, reserve: int | None = None) -> bool:
        """True when ``need`` more operations still leave enough to get home."""
        if self.done:
            return False
        reserve = self.home_cost() if reserve is None else reserve
        return self.ops - need >= reserve

    # ---- actions
    def act(self, tool: str, **kw) -> dict:
        if self.done:
            return {"ok": False, "error": "done"}
        return self.ep.step(tool, **kw)

    def go_home(self, max_calls: int = 6) -> None:
        wi = self.window
        for _ in range(max_calls):
            if self.done or self.window != wi or self.at_home() or self.ops <= 1:
                return
            self.act("navigate_to", x=self.w.home[0], y=self.w.home[1])

    def close_window(self, wi: int, go_home: bool = True) -> None:
        if self.done or self.window != wi:
            return
        if go_home:
            self.go_home()
        if not self.done and self.window == wi:
            self.act("end_window")

    def equip_if_better(self, item: str) -> bool:
        it = ITEMS.get(item)
        if not it or not self.inv(item):
            return False
        cur = self.w.eq.get(it["kind"])
        if cur and ITEMS[cur]["tier"] >= it["tier"]:
            return False
        return bool(self.act("equip_item", item_id=item).get("ok"))

    def craft(self, item: str, quantity: int = 1) -> bool:
        return bool(self.act("craft_item", item_id=item, quantity=quantity).get("ok"))


def _run(task_id: str, seed: int, calibration, play) -> dict:
    ep = Episode(task_id, seed)
    if play is not None:
        play(Pilot(ep))
    return ep.finalize(calibration)


# ================================================================ do-nothing
def _do_nothing(p: Pilot) -> None:
    return None


# ================================================================== absentee
def _absentee(p: Pilot) -> None:
    while not p.done:
        p.act("terraria_status")
        p.act("end_window")


# ================================================================ blast-list
def _blast_list(p: Pilot) -> None:
    """Spends every operation on a fixed chop / mine / swing rotation, wherever
    it happens to be standing. Never builds, never climbs the tech tree."""
    rotation = (
        ("chop_trees", {"radius": 40, "max_trees": 1}),
        ("mine_vein", {"tile_type": "stone", "max_tiles": 6}),
        ("engage_combat_burst", {}),
    )
    i = 0
    while not p.done:
        tool, kw = rotation[i % len(rotation)]
        p.act(tool, **kw)
        i += 1


# ============================================================ shared routines
def _gather_wood(p: Pilot, target: int, max_calls: int = 3, reserve: int | None = None) -> None:
    for _ in range(max_calls):
        if p.inv("wood") >= target or not p.room(reserve=reserve):
            return
        if not p.act("chop_trees", radius=60, max_trees=4).get("ok"):
            return


def _shelter(p: Pilot) -> bool:
    if "SHELTER" in p.w.milestones and p.w.housing()["valid"]:
        return True
    if p.inv("wood") < 26 and not p.inv("workbench"):
        _gather_wood(p, 26, reserve=1)
    res = p.act("build_blueprint", preset="shelter")
    return bool(res.get("ok") and res.get("housing", {}).get("valid"))


def _wood_gear(p: Pilot, bow: bool = False) -> None:
    """Tier 1: a wooden weapon and wood armor, equipped only if an upgrade."""
    weapon = "wooden_bow" if bow else "wooden_sword"
    need = (10 if bow else 7) + 30
    if p.inv("wood") < need:
        _gather_wood(p, need + 4, max_calls=2)
    for item in (weapon, "wood_armor"):
        if not p.inv(item) and ITEMS[item]["tier"] > ITEMS[p.w.eq.get(ITEMS[item]["kind"]) or "copper_shortsword"]["tier"]:
            p.craft(item)
        p.equip_if_better(item)


def _descend(p: Pilot, depth: int, max_calls: int = 6, careful: bool = True) -> None:
    for _ in range(max_calls):
        if p.w.pos[1] >= depth or not p.room(2) or (careful and p.hurt):
            return
        before = p.w.pos
        p.act("explore_cavern", target_depth=depth, auto_torch=False)
        if p.w.pos == before:
            return


def _mine(p: Pilot, ore: str, target: int, max_calls: int = 6, depth: int | None = None,
          careful: bool = False) -> None:
    """Mine ``ore`` until the inventory holds ``target`` of it. A careful
    miner stops when hurt and never pushes far past the ore's home depth."""
    for _ in range(max_calls):
        if p.inv(ore) >= target or not p.room(2) or (careful and p.hurt):
            return
        if depth is not None and p.w.pos[1] < depth - 6:
            _descend(p, depth, max_calls=2, careful=careful)
            continue
        res = p.act("mine_vein", tile_type=ore, max_tiles=12)
        if not res.get("ok") or not res.get("gained", {}).get(ore):
            if depth is None:
                return
            floor = depth + 12 if careful else 86
            if p.w.pos[1] >= floor:
                return
            _descend(p, min(p.w.pos[1] + (6 if careful else 10), floor), max_calls=1, careful=careful)


def _place_station(p: Pilot, item: str, dx: int) -> bool:
    """Place a crafting station inside the shelter (dx from the home tile)."""
    if p.w.near_tile(PLACEABLE[item], radius=4):
        return True
    if not p.inv(item):
        return False
    return bool(p.act("place_station_or_furniture", item_id=item, dx=dx, dy=0).get("ok"))


def _forge(p: Pilot, metal: str, pieces: tuple[str, ...]) -> None:
    """At home: furnace -> bars -> anvil -> metal kit, equipping upgrades."""
    if not p.at_home():
        return
    if not p.w.near_tile(FURNACE, 4):
        if not p.inv("furnace"):
            if p.inv("torch") < 3 and p.inv("gel") and p.inv("wood"):
                p.craft("torch", quantity=min(p.inv("gel"), p.inv("wood"), 2))
            p.craft("furnace")
        _place_station(p, "furnace", 2)
    ore = p.inv(f"{metal}_ore")
    if ore >= 2 and p.w.near_tile(FURNACE, 4):
        p.craft(f"{metal}_bar", quantity=ore // 2)
    if not p.w.near_tile(ANVIL, 4):
        if not p.inv("anvil") and p.inv("iron_bar") >= 3:
            p.craft("anvil")
        _place_station(p, "anvil", -1)
    if not p.w.near_tile(ANVIL, 4):
        return
    for piece in pieces:
        item = f"{metal}_{piece}"
        cur = p.w.eq.get(ITEMS[item]["kind"])
        if p.inv(item) or (cur and ITEMS[cur]["tier"] >= ITEMS[item]["tier"]):
            p.equip_if_better(item)
            continue
        if p.inv(f"{metal}_bar") >= _KIT_BARS[piece] and (piece != "pickaxe" or p.inv("wood") >= 2):
            if p.craft(item):
                p.equip_if_better(item)


def _bars_needed(p: Pilot, metal: str, pieces: tuple[str, ...]) -> int:
    need = 0
    for piece in pieces:
        item = f"{metal}_{piece}"
        cur = p.w.eq.get(ITEMS[item]["kind"])
        if not p.inv(item) and not (cur and ITEMS[cur]["tier"] >= ITEMS[item]["tier"]):
            need += _KIT_BARS[piece]
    if metal == "iron" and not p.w.near_tile(ANVIL, 4) and not p.inv("anvil"):
        need += 3
    return max(0, need - p.inv(f"{metal}_bar"))


# ================================================================ rule-based
_RB_PIECES = ("broadsword", "armor", "pickaxe")


def _rule_based(p: Pilot) -> None:
    """A rigid, task-agnostic checklist. It never reads the brief, alerts or
    objectives: shelter + wood gear on day one, then stone, iron and the iron
    kit, then it keeps mining iron forever. Every window ends at home."""
    while not p.done:
        wi = p.window
        if "SHELTER" not in p.w.milestones:
            p.act("chop_trees", radius=35, max_trees=4)
            p.act("build_blueprint", preset="shelter")
            for item in ("wooden_sword", "wood_armor"):
                if not p.inv(item):
                    p.craft(item)
            for item in ("wooden_sword", "wood_armor"):
                p.equip_if_better(item)
            p.close_window(wi)
            continue
        iron_done = _bars_needed(p, "iron", _RB_PIECES) == 0
        if not iron_done:
            if p.inv("stone") < 20 and not p.w.near_tile(FURNACE, 4) and not p.inv("furnace"):
                _mine(p, "stone", 22, max_calls=3)
            if p.inv("iron_ore") < 2 * _bars_needed(p, "iron", _RB_PIECES):
                _mine(p, "iron_ore", 2 * _bars_needed(p, "iron", _RB_PIECES), max_calls=8, depth=44)
            p.go_home()
            if p.window == wi and p.room(reserve=1):
                _forge(p, "iron", _RB_PIECES)
        else:  # checklist exhausted: keep the shallow iron seam busy
            _mine(p, "iron_ore", p.inv("iron_ore") + 16, max_calls=3, depth=44, careful=True)
        p.close_window(wi)


# ============================================================== smart-triage
_ST_PIECES = ("bow", "armor", "pickaxe")


def _alert_kinds(p: Pilot) -> set:
    _, alerts = p.w.alerts()
    out = set()
    for a in alerts:
        t = a["text"]
        if "Chilling winds" in t:
            out.add("chill")
        elif "evil presence" in t:
            out.add("eye_rumor")
        elif "Blood Moon is rising" in t:
            out.add("blood_moon")
        elif "goblin army is attacking" in t:
            out.add("siege")
        elif "Slime is falling" in t:
            out.add("slime_rain")
        elif "has awoken" in t:
            out.add("boss")
    return out


def _arrows(p: Pilot, want: int = 150) -> None:
    """Keep a quiver: 25 arrows per (wood + stone) at the workbench."""
    if p.inv("wooden_arrow") >= want or not p.at_home():
        return
    batches = min((want - p.inv("wooden_arrow")) // 25 + 1, p.inv("wood") - 4, p.inv("stone"))
    if batches > 0 and p.room():
        p.craft("wooden_arrow", quantity=batches)


def _fight_mobs(p: Pilot, family: str, kills_needed: int, max_bursts: int = 8) -> None:
    weapon = p.w.eq["weapon"]
    for _ in range(max_bursts):
        if p.w.kills.get(family, 0) >= kills_needed or not p.room():
            return
        targets = [m for m in p.w.roster() if m["family"] == family]
        if not targets:
            return
        if p.w.hp < 0.45 * p.w.max_hp:
            p.act("retreat_and_stabilize", use_potion=True)
            if p.w.hp < 0.3 * p.w.max_hp:
                return
            continue
        ranged = ITEMS[weapon]["ranged"] if weapon else False
        res = p.act("engage_combat_burst", target_id=targets[0]["id"], kite_distance=16 if ranged else 0,
                    dodge_priority=0.5, vertical_platform_use=True)
        if not res.get("ok") or res.get("died"):
            return


def _siege(p: Pilot, max_bursts: int = 16) -> None:
    for _ in range(max_bursts):
        if not p.w.army or not p.room():
            return
        if p.w.hp < 0.4 * p.w.max_hp:
            p.act("retreat_and_stabilize", use_potion=True)
            if p.w.hp < 0.3 * p.w.max_hp:
                return
            continue
        res = p.act("engage_combat_burst", target_id=p.w.army[0]["id"], kite_distance=16, dodge_priority=0.5)
        if not res.get("ok") or res.get("died"):
            return


def _boss_fight(p: Pilot, kind: str, reserve: int = 1, kite: float = 16.0) -> None:
    heal_at = 0.45 * p.w.max_hp
    while p.w.boss and p.room(reserve=reserve):
        if p.w.hp < heal_at and p.inv("lesser_healing_potion") and p.w.abs_tick >= p.w.sick_until:
            p.act("consume_buff_or_heal", item_id="lesser_healing_potion")
            continue
        res = p.act("engage_combat_burst", target_id=f"boss_{kind}", kite_distance=kite, dodge_priority=0.5,
                    vertical_platform_use=True)
        if not res.get("ok") or res.get("died"):
            return


def _arena(p: Pilot) -> bool:
    """Build the 3-tier arena well clear of the shelter so it cannot breach it."""
    if "ARENA" in p.w.milestones:
        return True
    need = 18 + 11 + 2
    if p.inv("wood") + p.inv("wood_platform") // 2 < need:
        _gather_wood(p, need + 4, max_calls=2)
    if not p.room(2):
        return False
    hx = p.w.home[0]
    ax = hx + 32 if hx + 32 < 178 else hx - 32
    p.act("navigate_to", x=ax, y=p.w.surf[ax] - 1)
    if abs(p.w.pos[0] - ax) > 3:
        return False
    return bool(p.act("build_blueprint", preset="arena").get("ok")) and "ARENA" in p.w.milestones


def _progress_metal(p: Pilot, metal: str, depth: int, bow: bool = True) -> None:
    """One window's worth of work toward ``metal`` gear: gather, return, forge."""
    wi = p.window
    if metal == "iron" and not p.w.near_tile(FURNACE, 4) and not p.inv("furnace") and p.inv("stone") < 20:
        _mine(p, "stone", 24, max_calls=3)
    need = _bars_needed(p, metal, _ST_PIECES)
    if need and p.inv(f"{metal}_ore") < 2 * need:
        _mine(p, f"{metal}_ore", 2 * need, max_calls=8, depth=depth, careful=True)
        if p.room(2) and p.inv("stone") < 10:
            _mine(p, "stone", p.inv("stone") + 6, max_calls=1, careful=True)
    if p.room(2) and p.w.crystals < 3:
        _crystal(p, max_calls=1)
    p.go_home()
    if p.window == wi and p.at_home():
        _forge(p, metal, _ST_PIECES)
        if bow:
            _arrows(p)


def _crystal(p: Pilot, max_calls: int = 3) -> None:
    for _ in range(max_calls):
        if not p.room(2):
            return
        found = p.w.nearest_known({CRYSTAL}, maxd=60)  # crystals revealed by earlier scans
        if found is None:
            return
        res = p.act("loot_container_or_crystal", kind="life_crystal")
        if res.get("ok") or "already" in str(res.get("error", "")):
            continue
        if "could not reach" not in str(res.get("error", "")):
            return


def _metal_target(p: Pilot) -> tuple[str, int] | None:
    tier = p.w.tech_tier
    pick = ITEMS[p.w.eq["pickaxe"]]["tier"]
    if tier < 2 or pick < 2:  # the iron pickaxe gates silver and gold
        return "iron", 46
    if tier < 3:
        return "silver", 64
    if tier < 4:
        return "gold", 76
    return None


def _smart_wall_of_flesh(p: Pilot) -> None:
    """Gold kit, Voodoo doll: shelter, descend, bridge, fight, come home."""
    phase = "shelter"
    while not p.done:
        wi = p.window
        chill = "chill" in _alert_kinds(p) or p.w.event("cave_chill") is not None
        if phase == "shelter":
            _shelter(p)
            _gather_wood(p, 400, max_calls=2)
            phase = "descend"
            p.close_window(wi)
            continue
        if phase == "descend" and wi >= 2 and not chill:
            if wi == 2:  # the day-2 shelter check comes first; prepare and stay home
                _gather_wood(p, 600, max_calls=2)
                p.close_window(wi)
                continue
            _descend(p, UNDERWORLD_Y + 2, max_calls=10)
            if p.w.pos[1] >= UNDERWORLD_Y and p.room(1, reserve=0):
                p.act("build_blueprint", preset="hellbridge")
                phase = "fight" if "HELLBRIDGE" in p.w.milestones else phase
            p.close_window(wi, go_home=False)
            continue
        if phase == "fight" and p.w.pos[1] >= UNDERWORLD_Y:
            for m in [m for m in p.w.roster() if m["family"] == "demon"]:
                if p.ops > 16:
                    p.act("engage_combat_burst", target_id=m["id"], kite_distance=18, dodge_priority=0.5)
            if p.inv("ironskin_potion"):
                p.act("consume_buff_or_heal", item_id="ironskin_potion")
            if p.act("summon_or_trigger_boss", boss="wall_of_flesh").get("ok"):
                _boss_fight(p, "wall_of_flesh", reserve=2, kite=18)
                if p.w.hp < 0.6 * p.w.max_hp and p.w.abs_tick >= p.w.sick_until:
                    p.act("consume_buff_or_heal", item_id="lesser_healing_potion")
            phase = "return"
            p.close_window(wi, go_home=False)
            continue
        if phase == "return":
            p.go_home(max_calls=12)
            if p.at_home():
                phase = "mine"
            p.close_window(wi)
            continue
        # after the fight: keep progressing value from home if the depths are calm
        if not chill and p.room(4):
            _mine(p, "gold_ore", p.inv("gold_ore") + 16, max_calls=4, depth=72)
            _mine(p, "silver_ore", p.inv("silver_ore") + 12, max_calls=2, depth=60)
        p.close_window(wi)


def _smart_triage(p: Pilot) -> None:
    task = p.ep.task
    if task.id == "wall-of-flesh" or "boss:wall_of_flesh" in task.objectives:
        return _smart_wall_of_flesh(p)
    obj = set(task.objectives)
    wants_arena = "arena" in obj or "boss:eye" in obj
    wants_bow = wants_arena or bool(obj & {"siege", "blood_moon", "slime_rain"})
    summoned = False
    while not p.done:
        wi = p.window
        alerts = _alert_kinds(p)
        night = p.w.phase == "Night"
        # ---- day one: housing + tier 1, always
        if "SHELTER" not in p.w.milestones or not p.w.housing()["valid"] and p.at_home():
            _shelter(p)
        if p.w.tech_tier < 1:
            _wood_gear(p, bow=wants_bow and not p.inv("wooden_bow"))
            if wants_bow and not p.inv("wooden_bow") and p.w.eq["weapon"] != "wooden_bow":
                if p.inv("wood") >= 10:
                    p.craft("wooden_bow")
        if wants_bow and not any(ITEMS[k]["kind"] == "weapon" and ITEMS[k]["ranged"] for k in p.w.inv):
            if p.inv("wood") >= 10 and p.at_home():
                p.craft("wooden_bow")
        if wants_bow and p.w.eq["weapon"] and not ITEMS[p.w.eq["weapon"]]["ranged"]:
            bows = sorted((ITEMS[k]["tier"], k) for k in p.w.inv if ITEMS[k]["kind"] == "weapon" and ITEMS[k]["ranged"])
            if bows and bows[-1][0] >= ITEMS[p.w.eq["weapon"]]["tier"]:
                p.act("equip_item", item_id=bows[-1][1])
        if wants_bow and p.inv("wooden_arrow") < 100 and p.at_home():
            if p.inv("stone") < 4 and p.room(3):
                _mine(p, "stone", 8, max_calls=1)
                p.go_home()
            _arrows(p, 200)
        # ---- live threats first
        if p.w.boss and p.w.boss["kind"] == "eye":
            if p.room(2):
                p.act("deploy_combat_arena_tactics", tactic="grapple_tier")
            _boss_fight(p, "eye", reserve=1)
            if p.w.boss is None or p.done or p.window != wi:
                p.close_window(wi)
                continue
            p.close_window(wi, go_home=False)
            continue
        if "siege" in alerts and p.w.army:
            _siege(p)
            p.close_window(wi)
            continue
        if "blood_moon" in alerts and "BLOODMOON" not in p.w.milestones:
            _fight_mobs(p, "blood_zombie", 8 + 0, max_bursts=10)
            p.close_window(wi)
            continue
        if "slime_rain" in alerts and "SLIME_RAIN" not in p.w.milestones and p.w.phase == "Day":
            _fight_mobs(p, "rain_slime", 10, max_bursts=8)
            p.close_window(wi)
            continue
        # ---- planned objectives
        if wants_arena and "ARENA" not in p.w.milestones and (p.w.tech_tier >= 1) and wi >= 2:
            ready_for_arena = "arena-architect" == task.id or p.w.tech_tier >= 2 or wi >= task.windows - 8
            if ready_for_arena and p.room(3):
                _arena(p)
                p.go_home()
        if ("boss:eye" in obj and not summoned and p.inv("suspicious_looking_eye") and night
                and "ARENA" in p.w.milestones and p.w.tech_tier >= 2 and p.ops >= task.budget - 2
                and p.inv("wooden_arrow") >= 150):
            if p.act("summon_or_trigger_boss", boss="eye").get("ok"):
                summoned = True
                p.act("deploy_combat_arena_tactics", tactic="grapple_tier")
                _boss_fight(p, "eye", reserve=1)
                p.close_window(wi)
                continue
        chill = "chill" in alerts or p.w.event("cave_chill") is not None
        if "depth" in obj and p.w.max_depth < CAVERN_Y + 4 and not chill:
            _descend(p, CAVERN_Y + 6, max_calls=4)
        if "crystal" in obj and p.w.crystals < 1 and not chill and p.w.max_depth >= CAVERN_Y:
            _descend(p, 58, max_calls=2)
            _crystal(p)
        target = _metal_target(p)
        if target and not chill and p.room(4):
            _progress_metal(p, *target, bow=wants_bow)
        elif not chill and p.room(4):
            _mine(p, "gold_ore", p.inv("gold_ore") + 16, max_calls=3, depth=74, careful=True)
        elif chill and p.room(3) and p.at_home():
            _gather_wood(p, p.inv("wood") + 40, max_calls=2)
        if p.at_home() and p.w.tech_tier >= 2:
            _arrows(p, 200 if wants_bow else 0)
        p.close_window(wi)


# ================================================================= exploiter
def _exploiter(p: Pilot) -> None:
    from .exploits import play_exploiter_window
    wi = p.window
    play_exploiter_window(p.ep)
    if not p.done and p.window == wi:
        p.act("end_window")
    # ... and quits: finalize settles the rest of the episode unattended.


_POLICIES = {
    "do-nothing": _do_nothing,
    "absentee": _absentee,
    "blast-list": _blast_list,
    "rule-based": _rule_based,
    "smart-triage": _smart_triage,
    "exploiter": _exploiter,
}
assert tuple(_POLICIES) == ANCHORS


def run_policy(policy: str, task_id: str, seed: int, calibration: dict | None = None) -> dict:
    """Run a scripted anchor policy through Episode(task_id, seed) and return its sealed ledger."""
    if policy not in _POLICIES:
        raise KeyError(f"unknown anchor policy '{policy}'; expected one of {ANCHORS}")
    return _run(task_id, seed, calibration, _POLICIES[policy])


def run_policy_episode(policy: str, task_id: str, seed: int) -> Episode:
    """Like run_policy, but return the (unfinalized) Episode for replay/diagnostics."""
    ep = Episode(task_id, seed)
    _POLICIES[policy](Pilot(ep))
    return ep
