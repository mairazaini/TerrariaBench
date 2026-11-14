"""Deterministic tech-graph planner for one goal: defeat the Eye of Cthulhu in real Terraria (normal mode).

The tModLoader bridge executes *skills* (closed-loop programs that play with real swing timing and tool power; see
``mod/TerrariaRLBridge/Skills.cs``). This module decides which skill to run next. It reads only the sanitised,
human-visible HUD, keeps a durable on-disk memory of what has been found (altars, chests, stations, the arena) in
spawn-relative tile coordinates, and returns an ordered list of candidate skills with reasons. The caller either
runs the top candidate (autopilot) or lets a decision model pick among them.

Route: wood -> Work Bench -> bow + sword + wood armor -> stone -> arrows -> (ore -> furnace -> anvil -> metal bow)
-> 6 Lens at night + find a Demon/Crimson Altar -> Suspicious Looking Eye -> runway arena -> night -> summon -> fight.

Two layers, on purpose:

* ``Strategy`` is the LONG-TERM plan for one run: a handful of choices that trade preparation time against the odds
  of winning the fight (which bow, which armor, how much health, how many arrows, what to do on unarmed nights).
  A model writes it at the start of the run and may revise it after a death; it is stored in the run memory and
  rendered as a numbered ``roadmap`` with the current step marked. ``REFERENCE_STRATEGY`` is what the scripted
  reference run uses.
* ``candidates`` is the SHORT-TERM layer: the legal next skills given the strategy, the HUD and what memory holds.
  The planner only ever proposes steps that are possible right now, so a model cannot waste a turn on a skill that
  must fail, but which strategy it runs under, and which candidate it picks, is the model's decision.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "runs"

SUMMON = "Suspicious Looking Eye"
WOOD_ARMOR = ("Wood Helmet", "Wood Breastplate", "Wood Greaves")
# Best first. Each needs 7 bars at an anvil.
METAL_BOWS = ("Platinum Bow", "Gold Bow", "Tungsten Bow", "Silver Bow", "Lead Bow", "Iron Bow", "Tin Bow", "Copper Bow")
ANVILS = ("Iron Anvil", "Lead Anvil")
ARROW_NAMES = ("Wooden Arrow", "Flaming Arrow", "Frostburn Arrow", "Jester's Arrow", "Unholy Arrow", "Bone Arrow")

ARROWS_WOODEN_BOW = 900   # 3 damage a hit through 12 defense in phase one: the fight is long
ARROWS_METAL_BOW = 500
ORE_TARGET = 60           # 5 bars for an anvil + 7 for a bow, with slack for mixed ore types
ORE_ATTEMPT_LIMIT = 10    # failed underground prospecting runs before settling for the wooden bow
ARENA_WIDTH = 90
BLOCK_COOLDOWN = 5        # planner decisions a twice-failed skill stays off the list
METAL_ARMOR_ORE = 180     # a full iron-tier set is 60 bars
CRYSTAL_SEARCH_LIMIT = 6  # underground prospecting runs spent looking for Life Crystals before moving on
BUNKER_BLOCKS = 11        # two 3-high side walls and a 5-wide roof
HEAL_ITEMS = ("Healing Potion", "Lesser Healing Potion", "Mushroom")
SUMMON_HOURS = (19.5, 1.5)  # summon between 7:30 PM and 1:30 AM so the Eye cannot escape at dawn


@dataclass(frozen=True)
class Strategy:
    """Long-term plan for one run. Every field is a real trade-off, not a free upgrade:

    * ``bow``: "metal" needs ore, a furnace and an anvil (slow) but roughly doubles damage per arrow; "wooden" is
      ready on day one and makes the fight long enough to run into dawn.
    * ``armor``: "metal" costs about 180 more ore; "wood" is 3 defense; "none" saves the wood.
    * ``max_hp``: Life Crystals to look for underground before summoning (100 = do not look).
    * ``arrows``: ammunition to carry into the fight (0 = the planner default for the chosen bow).
    * ``night``: "bunker" walls the player in on a night spent unarmed or badly hurt; "work" keeps going.
    * ``pickaxe``: spend bars on a better pickaxe first, which makes all later mining faster.
    """

    bow: str = "metal"
    armor: str = "wood"
    max_hp: int = 140
    arrows: int = 0
    night: str = "bunker"
    pickaxe: bool = False
    arena_width: int = ARENA_WIDTH
    rationale: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> "Strategy":
        """Build from untrusted model output: unknown keys are dropped and every value is clamped."""
        d = raw if isinstance(raw, dict) else {}
        base = cls()

        def pick(key: str, allowed: tuple[str, ...], default: str) -> str:
            v = str(d.get(key, default)).strip().lower()
            return v if v in allowed else default

        def num(key: str, lo: int, hi: int, default: int) -> int:
            try:
                return max(lo, min(hi, int(float(d.get(key, default)))))
            except (TypeError, ValueError):
                return default

        raw_pick = d.get("pickaxe", base.pickaxe)
        return cls(
            bow=pick("bow", ("wooden", "metal"), base.bow),
            armor=pick("armor", ("none", "wood", "metal"), base.armor),
            max_hp=num("max_hp", 100, 400, base.max_hp) // 20 * 20,
            arrows=num("arrows", 0, 2000, base.arrows),
            night=pick("night", ("bunker", "work"), base.night),
            pickaxe=bool(raw_pick) and str(raw_pick).strip().lower() not in ("false", "0", "no"),
            arena_width=num("arena_width", 40, 200, base.arena_width),
            rationale=str(d.get("rationale") or "")[:400],
        )

    def as_dict(self) -> dict[str, Any]:
        return {"bow": self.bow, "armor": self.armor, "max_hp": self.max_hp, "arrows": self.arrows,
                "night": self.night, "pickaxe": self.pickaxe, "arena_width": self.arena_width,
                "rationale": self.rationale}


REFERENCE_STRATEGY = Strategy(rationale="Reference run: metal bow, wood armor, two Life Crystals, bunker on unarmed nights.")

STRATEGY_BRIEF = (
    "Write the long-term plan for this run as JSON with exactly these keys:\n"
    '{"bow": "wooden|metal", "armor": "none|wood|metal", "max_hp": 100-400 (multiple of 20), '
    '"arrows": 0-2000 (0 = default for the bow), "night": "bunker|work", "pickaxe": true|false, '
    '"arena_width": 40-200, "rationale": "<2-3 sentences: why these choices for THIS world and state>"}\n'
    "Facts: the Eye of Cthulhu has 2800 HP and 12 defense until half health, then 0 defense and faster charges. "
    "A Wooden Bow with Wooden Arrows does about 3 damage a hit in phase one; an iron-tier bow about 7. It must be "
    "summoned at night and leaves at dawn, so a slow fight is a lost fight. Metal gear needs ore from underground, "
    "a Furnace (20 Stone, 4 Wood, 3 Torch) and an Anvil (5 bars); a bow is 7 bars and a full armor set about 60. "
    "Life Crystals (+20 max HP each) are underground. The summoning item needs 6 Lens from Demon Eyes (night, "
    "surface) crafted at a Demon/Crimson Altar. Every extra preparation step costs in-game days and risks a death."
)


@dataclass
class Candidate:
    name: str
    args: dict[str, Any]
    reason: str

    @property
    def key(self) -> str:
        return self.name + json.dumps(self.args, sort_keys=True)

    def label(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({args}) - {self.reason}"


def _blank_memory() -> dict[str, Any]:
    return {
        "world": "",
        "altars": [],            # [[x, y]]
        "chests": {},            # "x,y" -> looted bool
        "life_crystals": [],     # [[x, y]]
        "stations": {},          # kind -> [x, y]
        "arena": None,           # {"x": int, "y": int, "width": int}
        "explored": {"left": 0, "right": 0},
        "ore_attempts": 0,
        "ore_given_up": False,
        "altar_digs": 0,
        "deaths": [],            # [[x, y]]
        "eye_defeated": False,
        "summoned": False,
        "strategy": None,        # Strategy.as_dict() once a plan has been written for this run
        "strategy_author": "",   # model id that wrote it, or "reference"
        "strategy_revisions": 0,
        "bunker": None,          # [x, y] while walled in for the night
        "crystal_searches": 0,
        "pickaxe_upgraded": False,
        "log": [],               # last skill outcomes, newest last
    }


@dataclass
class EyePlanner:
    """Tech-graph planner with durable memory. All coordinates are tiles relative to the world spawn point."""

    memory_path: Path | None = None
    mem: dict[str, Any] = field(default_factory=_blank_memory)
    _fails: dict[str, int] = field(default_factory=dict)
    _blocked: dict[str, int] = field(default_factory=dict)
    _decisions: int = 0
    _was_dead: bool = False

    def __post_init__(self) -> None:
        if self.memory_path is None:
            env = os.environ.get("TERRARIA_EYE_MEMORY")
            self.memory_path = Path(env) if env else DEFAULT_MEMORY_DIR / "eye_memory.json"
        self.memory_path = Path(self.memory_path)
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self.mem = {**_blank_memory(), **data}

    def save(self) -> None:
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.memory_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.mem, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.memory_path)
        except OSError:
            pass  # memory is an optimisation; a read-only disk must not stop the run

    # ------------------------------------------------------------------ long-term plan
    @property
    def strategy(self) -> Strategy:
        raw = self.mem.get("strategy")
        return Strategy.from_dict(raw) if isinstance(raw, dict) else REFERENCE_STRATEGY

    @property
    def has_authored_strategy(self) -> bool:
        return isinstance(self.mem.get("strategy"), dict)

    def set_strategy(self, raw: Any, author: str) -> Strategy:
        """Adopt a plan (validated and clamped) and remember who wrote it."""
        strategy = raw if isinstance(raw, Strategy) else Strategy.from_dict(raw)
        if self.has_authored_strategy:
            self.mem["strategy_revisions"] = int(self.mem.get("strategy_revisions") or 0) + 1
        self.mem["strategy"] = strategy.as_dict()
        self.mem["strategy_author"] = str(author)[:80]
        self.save()
        return strategy

    def roadmap(self, hud: dict[str, Any]) -> list[dict[str, Any]]:
        """The run's step-by-step plan under the current strategy: [{n, step, status}] where status is
        done | current | todo | skipped. Exactly one step is ``current`` until the Eye is dead."""
        st, mem = self.strategy, self.mem
        bag = hud.get("backpack") if isinstance(hud.get("backpack"), dict) else {}
        worn = [str(n) for n in (hud.get("equipment_now") or [])]

        def have(name: str) -> int:
            return int(bag.get(name) or 0)

        arrows = sum(have(n) for n in ARROW_NAMES)
        metal_bow = any(have(b) for b in METAL_BOWS)
        any_bow = metal_bow or have("Wooden Bow") > 0
        slots = {_slot_of(w): w for w in worn if _slot_of(w)}
        full_armor = len(slots) == 3
        metal_armor = full_armor and not any(w.startswith("Wood ") for w in slots.values())
        ore_abandoned = bool(mem.get("ore_given_up"))
        crystals_abandoned = int(mem.get("crystal_searches") or 0) >= CRYSTAL_SEARCH_LIMIT
        arrow_goal = st.arrows or (ARROWS_METAL_BOW if (metal_bow or st.bow == "metal") else ARROWS_WOODEN_BOW)
        max_hp = _num(hud.get("max_hp"), 100)
        # (text, done) where done is True/False, or None for "not part of this plan / abandoned"
        steps: list[tuple[str, bool | None]] = [
            ("Chop wood and place a Work Bench", bool(mem["stations"].get("Work Bench"))),
            ("Craft a Wooden Bow and Wooden Sword", any_bow),
            ("Craft and wear Wood armor", None if st.armor == "none" else full_armor),
            (f"Mine stone and craft {arrow_goal} arrows", arrows >= arrow_goal),
            ("Mine ore, build a Furnace and an Anvil, craft a metal bow",
             None if st.bow != "metal" or (ore_abandoned and not metal_bow) else metal_bow),
            ("Craft and wear metal armor",
             None if st.armor != "metal" or (ore_abandoned and not metal_armor) else metal_armor),
            (f"Find Life Crystals until max HP is {st.max_hp}",
             None if st.max_hp <= 100 or (crystals_abandoned and max_hp < st.max_hp) else max_hp >= st.max_hp),
            ("Find a Demon/Crimson Altar", bool(mem["altars"]) or have(SUMMON) > 0),
            ("Hunt Demon Eyes at night for 6 Lens and craft the Suspicious Looking Eye",
             have(SUMMON) > 0 or bool(mem.get("summoned"))),
            (f"Level a {st.arena_width}-tile arena near spawn", bool(mem.get("arena"))),
            ("Summon the Eye early in the night and win the fight", bool(mem.get("eye_defeated"))),
        ]
        won = bool(mem.get("eye_defeated"))
        out, current_set = [], won
        for n, (text, done) in enumerate(steps, 1):
            if done is not None and won:
                done = True  # arrows were shot and the summoning item was used: the step still happened
            if done is None:
                status = "skipped"
            elif done:
                status = "done"
            elif not current_set:
                status, current_set = "current", True
            else:
                status = "todo"
            out.append({"n": n, "step": text, "status": status})
        return out

    # ------------------------------------------------------------------ observation
    @staticmethod
    def _pos(hud: dict[str, Any]) -> tuple[int, int] | None:
        pos = hud.get("pos")
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            try:
                return int(pos["x"]), int(pos["y"])
            except (TypeError, ValueError):
                return None
        return None

    def observe(self, hud: dict[str, Any]) -> None:
        """Fold what is on screen into memory. Called every tick; cheap."""
        world = str(hud.get("world") or "")
        if world and world != self.mem.get("world"):
            # Positions are spawn-relative, so memory from another world would be wrong everywhere.
            if self.mem.get("world"):
                self.mem = _blank_memory()
                self._fails.clear()
                self._blocked.clear()
            self.mem["world"] = world
            self.save()
        pos = self._pos(hud)
        dead = bool(hud.get("dead"))
        if self.mem.get("bunker") and (dead or str(hud.get("time_phase") or "Day") != "Night"):
            self.mem["bunker"] = None
            self.save()
        if dead and not self._was_dead and pos:
            self.mem["deaths"] = (self.mem["deaths"] + [list(pos)])[-20:]
            self.save()
        self._was_dead = dead
        wv = hud.get("world_view")
        if not pos or not isinstance(wv, dict):
            return
        changed = False

        def absolute(entry: Any) -> list[int] | None:
            try:
                return [pos[0] + int(entry["dx"]), pos[1] + int(entry["dy"])]
            except (KeyError, TypeError, ValueError):
                return None

        for entry in wv.get("altars") or []:
            at = absolute(entry)
            if at and not any(abs(at[0] - a[0]) <= 3 and abs(at[1] - a[1]) <= 3 for a in self.mem["altars"]):
                self.mem["altars"].append(at)
                changed = True
        for entry in wv.get("chests") or []:
            at = absolute(entry)
            if at and f"{at[0]},{at[1]}" not in self.mem["chests"]:
                self.mem["chests"][f"{at[0]},{at[1]}"] = False
                changed = True
        for entry in wv.get("life_crystals") or []:
            at = absolute(entry)
            if at and at not in self.mem["life_crystals"]:
                self.mem["life_crystals"].append(at)
                changed = True
        seen_kinds = set()
        for entry in wv.get("stations") or []:
            kind = str((entry or {}).get("kind") or "")
            at = absolute(entry)
            if kind in ("Work Bench", "Furnace", "Anvil") and at and kind not in seen_kinds:
                seen_kinds.add(kind)  # list is nearest-first
                if self.mem["stations"].get(kind) != at:
                    self.mem["stations"][kind] = at
                    changed = True
        side = "right" if pos[0] >= 0 else "left"
        if abs(pos[0]) > int(self.mem["explored"].get(side) or 0) + 8:
            self.mem["explored"][side] = abs(pos[0])
            changed = True
        if changed:
            self.save()

    def record_result(self, name: str, args: dict[str, Any], status: str, reason: str, hud: dict[str, Any]) -> None:
        """Feed a finished skill's outcome back: failure streaks, looted chests, mined crystals, the arena."""
        key = Candidate(name, args, "").key
        pos = self._pos(hud)
        ok = status == "success"
        if ok:
            self._fails.pop(key, None)
        else:
            self._fails[key] = self._fails.get(key, 0) + 1
            if self._fails[key] >= 2:
                self._blocked[key] = self._decisions + BLOCK_COOLDOWN
                self._fails[key] = 0
        if name == "loot_chest" and pos:
            # Looted, or unreachable/locked: either way stop walking back to the nearest known chest.
            near = min(self.mem["chests"], default=None, key=lambda k: _dist(_xy(k), pos))
            if near is not None and (ok or "no chest" not in reason):
                self.mem["chests"][near] = True
        if name == "mine" and str(args.get("kind")) == "life crystal" and pos and self.mem["life_crystals"]:
            near = min(self.mem["life_crystals"], key=lambda c: _dist(c, pos))
            if ok or _dist(near, pos) <= 40:
                self.mem["life_crystals"].remove(near)
        if name == "mine" and str(args.get("kind")) == "ore" and not ok:
            self.mem["ore_attempts"] += 1
            if self.mem["ore_attempts"] >= ORE_ATTEMPT_LIMIT:
                self.mem["ore_given_up"] = True
        if name == "goto" and pos and self.mem["life_crystals"]:
            wv = hud.get("world_view") if isinstance(hud.get("world_view"), dict) else {}
            if not wv.get("life_crystals"):
                self.mem["life_crystals"] = [c for c in self.mem["life_crystals"] if _dist(c, pos) > 8]
        if name == "bunker" and ok and pos:
            self.mem["bunker"] = list(pos)
        if name == "explore" and args.get("purpose") == "crystals":
            self.mem["crystal_searches"] = int(self.mem.get("crystal_searches") or 0) + 1
        if name == "craft" and str(args.get("item", "")).endswith(" Pickaxe") and ok:
            self.mem["pickaxe_upgraded"] = True
        if name == "build_arena" and ok and pos:
            self.mem["arena"] = {"x": pos[0], "y": pos[1], "width": int(args.get("width") or ARENA_WIDTH)}
        if name == "craft" and str(args.get("item")) == SUMMON and not ok and "no " in reason and pos:
            # The remembered altar is gone or was never one: forget those near here.
            self.mem["altars"] = [a for a in self.mem["altars"] if _dist(a, pos) > 45]
        if name == "use" and str(args.get("item")) == SUMMON:
            self.mem["summoned"] = ok
        if name == "fight":
            self.mem["summoned"] = False
            if ok:
                self.mem["eye_defeated"] = True
        if name == "dig_down" and args.get("purpose") == "altar":
            self.mem["altar_digs"] += 1
        self.mem["log"] = (self.mem["log"] + [f"{name} {json.dumps(args, sort_keys=True)}: {status} - {reason}"])[-12:]
        self.save()

    # ------------------------------------------------------------------ planning
    def candidates(self, hud: dict[str, Any]) -> list[Candidate]:
        """Ordered next-skill options, best first. Empty means nothing to do (dead, or the goal is complete)."""
        self._decisions += 1
        if self.mem.get("eye_defeated") or hud.get("dead"):
            return []
        out = [c for c in self._plan(hud) if self._blocked.get(c.key, 0) < self._decisions]
        seen: set[str] = set()
        unique = []
        for c in out:
            if c.key not in seen:
                seen.add(c.key)
                unique.append(c)
        if not unique:
            # Everything sensible is on cooldown: move somewhere new rather than stand still.
            unique.append(Candidate("explore", {"dir": self._fresh_direction(), "tiles": 60}, "all planned steps failed recently; try new ground"))
        return unique[:6]

    def _plan(self, hud: dict[str, Any]) -> list[Candidate]:
        bag = hud.get("backpack") if isinstance(hud.get("backpack"), dict) else {}
        worn = [str(n) for n in (hud.get("equipment_now") or [])]
        craftable = {str((c or {}).get("name") or "") for c in (hud.get("craftable_now") or []) if isinstance(c, dict)}
        pos = self._pos(hud) or (0, 0)
        wv = hud.get("world_view") if isinstance(hud.get("world_view"), dict) else {}
        night = str(hud.get("time_phase") or "Day") == "Night"
        clock = _num(hud.get("clock"), 12.0)
        underground = pos[1] > 25
        mem = self.mem
        strat = self.strategy
        hp, max_hp = _num(hud.get("hp"), 100), _num(hud.get("max_hp"), 100)

        def have(name: str) -> int:
            return int(bag.get(name) or 0)

        def have_like(pattern: str) -> int:
            return sum(int(q or 0) for n, q in bag.items() if re.search(pattern, str(n), re.I))

        wood = have("Wood")
        stone = have("Stone Block")
        arrows = sum(have(n) for n in ARROW_NAMES)
        ore = have_like(r" Ore$")
        bars = have_like(r" Bar$")
        gel = have("Gel")
        torches = have("Torch")
        lens = have("Lens")
        metal_bow = next((b for b in METAL_BOWS if have(b)), "")
        any_bow = bool(metal_bow) or have("Wooden Bow") > 0
        bench = mem["stations"].get("Work Bench")
        furnace = mem["stations"].get("Furnace")
        anvil = mem["stations"].get("Anvil")
        home = bench or [0, 0]
        out: list[Candidate] = []

        def add(name: str, reason: str, **args: Any) -> None:
            out.append(Candidate(name, args, reason))

        def go(target: list[int], why: str, tol_y: int = 6) -> bool:
            """Queue a walk when the target is too far for a station skill to find by itself."""
            if _dist(target, pos) > 30:
                add("goto", why, x=int(target[0]), y=int(target[1]), tol_y=tol_y)
                return True
            return False

        def at_bench(item: str, reason: str, count: int = 1) -> None:
            if not go(home, f"walk to the Work Bench to craft {item}"):
                add("craft", reason, item=item, count=count, station="Work Bench")

        # ---- 0. A boss on screen outranks everything.
        boss = hud.get("boss")
        if isinstance(boss, dict) and boss:
            arena = mem.get("arena")
            args = {"x_min": arena["x"] - arena["width"] // 2, "x_max": arena["x"] + arena["width"] // 2} if arena else {}
            out.append(Candidate("fight", args, f"{boss.get('name', 'boss')} is here: fight it"))
            return out
        if mem.get("summoned"):
            add("fight", "the summoning item was just used: fight the Eye")

        # ---- 1. Free wins that are already in the bag.
        if have("Life Crystal") and _num(hud.get("max_hp"), 100) < 400:
            add("use", "permanently +20 max HP", item="Life Crystal")
        for piece in WOOD_ARMOR:
            if have(piece) and piece not in worn and not any(_slot_of(w) == _slot_of(piece) for w in worn):
                add("equip", "wear the armor that is sitting in the bag", item=piece)
        for name in bag:
            if _slot_of(str(name)) and not str(name).startswith("Wood ") and str(name) not in worn and have(str(name)):
                if not any(_slot_of(w) == _slot_of(str(name)) and not w.startswith("Wood ") for w in worn):
                    add("equip", "metal armor beats wood", item=str(name))

        heal = next((h for h in HEAL_ITEMS if have(h)), "")
        if heal and hp < 0.5 * max_hp and "Potion Sickness" not in json.dumps(hud.get("buffs") or [], default=str):
            add("use", f"HP is {int(hp)}/{int(max_hp)}: heal before doing anything else", item=heal)

        # ---- 1b. Night on the surface without the means to fight: wall in until morning.
        blocks = wood + stone + have("Dirt Block") + have("Mud Block")
        armed = any_bow and arrows >= 50
        bunker_usable = self._blocked.get(Candidate("bunker", {}, "").key, 0) < self._decisions
        if night and not underground and strat.night == "bunker" and bunker_usable and (not armed or hp < 0.4 * max_hp):
            walled = mem.get("bunker")
            if walled and _dist(walled, pos) <= 2:
                add("wait", "inside the bunker: nothing can reach you until morning", until="day")
            elif blocks >= BUNKER_BLOCKS:
                add("bunker", "night without a working ranged weapon (or badly hurt): wall yourself in")
            else:
                add("chop", "blocks to wall in for the night", wood=wood + 20)
            return out

        # ---- 2. Ready to summon? Only once the strategy's preparation is done or has been given up on.
        slots_worn = {_slot_of(w): w for w in worn if _slot_of(w)}
        metal_armor_on = len(slots_worn) == 3 and not any(w.startswith("Wood ") for w in slots_worn.values())
        ore_closed = bool(mem.get("ore_given_up"))
        crystals_closed = int(mem.get("crystal_searches") or 0) >= CRYSTAL_SEARCH_LIMIT
        prep_done = (
            (bool(metal_bow) or strat.bow != "metal" or ore_closed)
            and (metal_armor_on or strat.armor != "metal" or ore_closed)
            and (max_hp >= strat.max_hp or crystals_closed)
        )
        ready_weapon = any_bow and arrows >= (150 if metal_bow else 300)
        if have(SUMMON) and ready_weapon and prep_done:
            arena = mem.get("arena")
            if not arena:
                if underground:
                    add("surface", "climb out to build the arena")
                elif go(home, "build the arena on the familiar ground near spawn", tol_y=12):
                    pass
                elif wood + stone + have("Dirt Block") + have("Wood Platform") < 60:
                    add("chop", "blocks for the arena floor", wood=wood + 60)
                else:
                    add("build_arena", "a long flat runway to outrun the Eye's charges", width=strat.arena_width)
            elif _dist([arena["x"], arena["y"]], pos) > 12:
                add("goto", "stand in the middle of the arena", x=arena["x"], y=arena["y"], tol_y=4)
            elif not night:
                add("wait", "the Eye can only be summoned at night", until="night")
            elif _in_summon_window(clock):
                add("use", "summon the Eye of Cthulhu", item=SUMMON)
            else:
                add("wait", "too close to dawn to finish the fight; wait for the next night", until="day")
            return out

        # ---- 3. Wood, bench, bow, armor.
        if not bench and not have("Work Bench"):
            if wood < 10:
                add("chop", "wood for a Work Bench and first gear", wood=160)
            else:
                add("craft", "every other recipe needs a Work Bench", item="Work Bench", count=1)
        elif not bench:
            add("place", "put the Work Bench down", item="Work Bench")
        if out and not underground:
            return out + self._wood_fallback(wood)

        missing_armor = [] if strat.armor == "none" else [
            p for p in WOOD_ARMOR if not have(p) and not any(_slot_of(w) == _slot_of(p) for w in worn)]
        wood_needed = (0 if any_bow else 10) + (0 if have("Wooden Sword") else 7) \
            + sum({"Wood Helmet": 20, "Wood Breastplate": 30, "Wood Greaves": 25}[p] for p in missing_armor)
        if bench and wood < wood_needed and not underground:
            add("chop", "wood for bow, sword and armor", wood=wood_needed + 60)
        if bench and not any_bow and wood >= 10:
            at_bench("Wooden Bow", "a bow is the only safe way to damage the Eye")
        if bench and not have("Wooden Sword") and wood >= 7 and any_bow:
            at_bench("Wooden Sword", "melee backup with a real swing arc")
        for piece, cost in (("Wood Breastplate", 30), ("Wood Helmet", 20), ("Wood Greaves", 25)):
            if bench and piece in missing_armor and wood >= cost and any_bow:
                at_bench(piece, "armor before the first night")
        if out and not underground:
            return out + self._wood_fallback(wood)

        # ---- 4. Arrows (1 Wood + 1 Stone -> 25 at the Work Bench).
        arrow_goal = strat.arrows or (ARROWS_METAL_BOW if metal_bow else ARROWS_WOODEN_BOW)
        if arrows < arrow_goal and not underground:
            batches = (arrow_goal - arrows + 24) // 25
            if stone < min(batches, 12):
                add("mine", "stone for arrows and a furnace", kind="stone", count=40)
                add("dig_down", "no stone on the surface here: it is a few tiles down", depth=12)
            elif wood < min(batches, 12):
                add("chop", "wood for arrows", wood=wood + 80)
            else:
                at_bench("Wooden Arrow", "ammunition for the fight", count=min(batches, stone, wood) * 25)

        # ---- 5. Opportunistic pickups in view.
        if wv.get("life_crystals") and _num(hud.get("max_hp"), 100) < 400:
            add("mine", "a Life Crystal is in view (+20 max HP)", kind="life crystal", count=1)
        nearest_chest = self._nearest_unlooted(pos, 45)
        if nearest_chest:
            add("loot_chest", "an unopened chest is close (potions, gear, maybe a summoning item)")

        # ---- 6. Metal gear the strategy asks for: ore -> furnace -> bars -> anvil -> bow / armor / pickaxe.
        metal_slots = {_slot_of(str(n)) for n in list(bag) + worn
                       if _slot_of(str(n)) and not str(n).startswith("Wood ") and (str(n) in worn or have(str(n)))}
        armor_missing = {"head", "body", "legs"} - metal_slots
        want_bow = strat.bow == "metal" and not metal_bow
        want_armor = strat.armor == "metal" and bool(armor_missing)
        want_pick = strat.pickaxe and not mem.get("pickaxe_upgraded")
        ore_target = (ORE_TARGET if want_bow else (0 if anvil else 15)) \
            + (METAL_ARMOR_ORE if want_armor else 0) + (36 if want_pick else 0)
        if want_bow or want_armor or want_pick:
            wanted: list[tuple[str, str]] = []
            if want_pick:
                better = next((c for c in sorted(craftable) if c.endswith(" Pickaxe") and not have(c)), "")
                if better:
                    wanted.append((better, "a better pickaxe makes every later dig faster"))
            if want_bow:
                best_bow = next((b for b in METAL_BOWS if b in craftable), "")
                if best_bow:
                    wanted.append((best_bow, "a metal bow multiplies damage per arrow"))
            if want_armor:
                per_slot: dict[str, str] = {}
                for c in sorted(craftable):
                    if _slot_of(c) in armor_missing and not c.startswith("Wood "):
                        per_slot.setdefault(_slot_of(c), c)
                wanted += [(c, "metal armor: every charge of the Eye costs less health") for c in per_slot.values()]
            anvil_item = next((a for a in ANVILS if have(a)), "")
            if wanted:
                for item, why in wanted:
                    add("craft", why, item=item, count=1, station="Anvil")
            elif anvil_item and not underground:
                if not go(home, "place the anvil next to the Work Bench"):
                    add("place", "put the anvil down", item=anvil_item)
            elif bars >= 5 and not anvil:
                made = next((a for a in ANVILS if a in craftable), "")
                if made:
                    add("craft", "an anvil unlocks metal bows", item=made, count=1, station="Work Bench")
                elif not underground:
                    go(home, "walk home to craft an anvil")
            elif bars >= 7 and anvil and not underground:
                go(anvil, "walk to the anvil to craft a bow")
            if ore >= 3 and not underground:
                if have("Furnace"):
                    if not go(home, "place the furnace next to the Work Bench"):
                        add("place", "put the furnace down", item="Furnace")
                elif not furnace:
                    if torches < 3 and gel < 1:
                        add("hunt", "Gel for torches (a furnace needs 3)", npc="Slime", item="Gel", count=3, when="any")
                    elif torches < 3:
                        at_bench("Torch", "a furnace needs 3 torches", count=6)
                    elif stone < 20:
                        add("mine", "a furnace needs 20 stone", kind="stone", count=20 - stone)
                    else:
                        at_bench("Furnace", "smelt ore into bars")
                else:
                    bar = next((c for c in sorted(craftable) if c.endswith(" Bar")), "")
                    if bar:
                        add("craft", "smelt the ore", item=bar, count=40, station="Furnace")
                    else:
                        go(furnace, "walk to the furnace to smelt ore")
            if not mem.get("ore_given_up") and ore + bars * 3 < ore_target and not night:
                if wv.get("ores"):
                    add("mine", "ore in view: bars for the anvil and the metal gear in the plan", kind="ore", count=ore_target - ore)
                elif underground:
                    add("explore", "prospect sideways for ore", dir=self._fresh_direction(), tiles=40, mode="tunnel")
                    add("dig_down", "go deeper for ore", depth=25)
                elif arrows >= arrow_goal // 2:
                    add("dig_down", "ore is underground", depth=40)

        # ---- 6b. Life Crystals up to the strategy's health target.
        if max_hp < strat.max_hp and not crystals_closed and not wv.get("life_crystals"):
            known = min(mem["life_crystals"], default=None, key=lambda c: _dist(c, pos))
            if known is not None:
                add("goto", "a Life Crystal seen earlier (+20 max HP)", x=known[0], y=known[1], tol_y=6)
            elif underground:
                add("explore", "search these caves for a Life Crystal", dir=self._fresh_direction(), tiles=40,
                    mode="tunnel", purpose="crystals")
            elif not night and arrows >= arrow_goal // 2:
                add("dig_down", "Life Crystals are underground", depth=40)

        # ---- 7. Summoning item: 6 Lens crafted at a Demon/Crimson Altar.
        if not have(SUMMON):
            altar = min(mem["altars"], default=None, key=lambda a: _dist(a, pos))
            if lens >= 6 and altar:
                if _dist(altar, pos) > 30:
                    add("goto", "walk to the altar with the 6 lenses", x=altar[0], y=altar[1], tol_y=8)
                else:
                    add("craft", "6 Lens at the altar make the summoning item", item=SUMMON, count=1, station="Demon Altar")
            if lens < 6 and night and any_bow and arrows >= 50:
                if underground:
                    add("surface", "Demon Eyes fly on the surface at night")
                else:
                    add("hunt", "Demon Eyes drop Lens (6 needed)", npc="Demon Eye", item="Lens", count=6)
            if not altar:
                if hud.get("evil_biome") and not underground and mem["altar_digs"] < 6:
                    out.append(Candidate("dig_down", {"depth": 45, "purpose": "altar"}, "inside the evil biome: altars sit in its caves below"))
                if underground and hud.get("evil_biome"):
                    add("explore", "search this evil-biome cave for an altar", dir=self._fresh_direction(), tiles=50, mode="tunnel")
                if underground:
                    add("surface", "nothing here; continue the altar search on the surface")
                else:
                    first = self._fresh_direction()
                    other = "left" if first == "right" else "right"
                    add("explore", "walk until the Corruption/Crimson and an altar come into view", dir=first, tiles=150)
                    add("explore", "search the other way for the evil biome", dir=other, tiles=150)
            elif lens < 6 and not night:
                if underground:
                    add("surface", "get back up before night to hunt Demon Eyes")
                elif arrows >= arrow_goal and (metal_bow or mem.get("ore_given_up")):
                    add("wait", "everything else is ready; lenses come at night", until="night")

        # ---- 8. Always have something useful to do.
        if underground and not out:
            add("surface", "nothing left to do down here")
        return out + ([] if underground else self._wood_fallback(wood))

    def _wood_fallback(self, wood: int) -> list[Candidate]:
        return [Candidate("chop", {"wood": wood + 60}, "spare wood never hurts (arrows, arena floor)")] if wood < 300 else []

    def _fresh_direction(self) -> str:
        explored = self.mem.get("explored") or {}
        return "right" if int(explored.get("right") or 0) <= int(explored.get("left") or 0) else "left"

    def _nearest_unlooted(self, pos: tuple[int, int], radius: int) -> list[int] | None:
        open_chests = [_xy(k) for k, looted in self.mem["chests"].items() if not looted]
        near = min(open_chests, default=None, key=lambda c: _dist(c, pos))
        return near if near is not None and _dist(near, pos) <= radius else None

    # ------------------------------------------------------------------ prompt
    def briefing(self, hud: dict[str, Any]) -> str:
        """Compact state summary for the decision model: progress ledger plus what memory holds."""
        bag = hud.get("backpack") if isinstance(hud.get("backpack"), dict) else {}
        keep = {n: q for n, q in bag.items() if re.search(r"Wood|Stone|Arrow|Bow|Sword|Ore|Bar|Lens|Gel|Torch|Potion|Crystal|Eye|Platform|Anvil|Furnace|Bench", str(n))}
        m = self.mem
        return json.dumps({
            "goal": "defeat the Eye of Cthulhu",
            "long_term_plan": {"strategy": self.strategy.as_dict(), "written_by": m.get("strategy_author") or "reference",
                               "steps": [f"{r['n']}. [{r['status']}] {r['step']}" for r in self.roadmap(hud)]},
            "hp": f"{hud.get('hp')}/{hud.get('max_hp')}", "defense": hud.get("defense"),
            "time": f"{hud.get('time_phase')} {hud.get('clock')}h", "pos_from_spawn": hud.get("pos"),
            "in_evil_biome": bool(hud.get("evil_biome")), "wearing": hud.get("equipment_now"), "key_items": keep,
            "known": {"altars": m["altars"][:3], "stations": m["stations"], "arena": m["arena"],
                      "unlooted_chests": sum(1 for v in m["chests"].values() if not v),
                      "explored_tiles": m["explored"], "deaths": len(m["deaths"])},
            "recent_skill_results": m["log"][-6:],
        }, sort_keys=True, default=str)


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _xy(key: str) -> list[int]:
    x, y = key.split(",")
    return [int(x), int(y)]


def _dist(a: Any, b: Any) -> float:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _slot_of(name: str) -> str:
    """Armor slot of an item name, or '' when it is not armor."""
    if re.search(r"(Helmet|Hat|Mask|Headgear|Hood)$", name):
        return "head"
    if re.search(r"(Breastplate|Chainmail|Mail|Shirt|Robe)$", name):
        return "body"
    if re.search(r"(Greaves|Leggings|Pants)$", name):
        return "legs"
    return ""


def _in_summon_window(clock: float) -> bool:
    start, end = SUMMON_HOURS
    return clock >= start or clock < end


def new_skill_id_seed() -> int:
    """Skill ids must never repeat across server restarts, or the mod would ignore the first command."""
    return int(time.time()) % 1_000_000_000
