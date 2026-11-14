"""Deterministic tech-graph planner for one goal: defeat the Eye of Cthulhu in real Terraria (normal mode).

The tModLoader bridge executes *skills* (closed-loop programs that play with real swing timing and tool power; see
``mod/TerrariaRLBridge/Skills.cs``). This module decides which skill to run next. It reads only the sanitised,
human-visible HUD, keeps a durable on-disk memory of what has been found (altars, chests, stations, the arena) in
spawn-relative tile coordinates, and returns an ordered list of candidate skills with reasons. The caller either
runs the top candidate (autopilot) or lets a decision model pick among them.

Route: wood -> Work Bench -> bow + sword + wood armor -> stone -> arrows -> (ore -> furnace -> anvil -> metal bow)
-> 6 Lens at night + find a Demon/Crimson Altar -> Suspicious Looking Eye -> runway arena -> night -> summon -> fight.
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
SUMMON_HOURS = (19.5, 1.5)  # summon between 7:30 PM and 1:30 AM so the Eye cannot escape at dawn


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

        # ---- 2. Ready to summon?
        ready_weapon = any_bow and arrows >= (150 if metal_bow else 300)
        if have(SUMMON) and ready_weapon:
            arena = mem.get("arena")
            if not arena:
                if underground:
                    add("surface", "climb out to build the arena")
                elif go(home, "build the arena on the familiar ground near spawn", tol_y=12):
                    pass
                elif wood + stone + have("Dirt Block") + have("Wood Platform") < 60:
                    add("chop", "blocks for the arena floor", wood=wood + 60)
                else:
                    add("build_arena", "a long flat runway to outrun the Eye's charges", width=ARENA_WIDTH)
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

        missing_armor = [p for p in WOOD_ARMOR if not have(p) and not any(_slot_of(w) == _slot_of(p) for w in worn)]
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
        arrow_goal = ARROWS_METAL_BOW if metal_bow else ARROWS_WOODEN_BOW
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

        # ---- 6. Metal bow: ore -> furnace -> bars -> anvil -> bow.
        if not metal_bow:
            best_bow = next((b for b in METAL_BOWS if b in craftable), "")
            anvil_item = next((a for a in ANVILS if have(a)), "")
            if best_bow:
                add("craft", "a metal bow multiplies damage per arrow", item=best_bow, count=1, station="Anvil")
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
            if not mem.get("ore_given_up") and ore + bars * 3 < ORE_TARGET and not night:
                if wv.get("ores"):
                    add("mine", "ore in view: bars for an anvil and a better bow", kind="ore", count=ORE_TARGET - ore)
                elif underground:
                    add("explore", "prospect sideways for ore", dir=self._fresh_direction(), tiles=40, mode="tunnel")
                    add("dig_down", "go deeper for ore", depth=25)
                elif arrows >= arrow_goal // 2:
                    add("dig_down", "ore is underground", depth=40)

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
