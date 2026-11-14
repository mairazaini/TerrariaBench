"""Deterministic headless world (the surrogate behind the MCP surface).

This is an abstract, symbolic tile world with Terraria-shaped mechanics: wood,
ore tiers, housing, caverns, events and two bosses. It is *not* the game; the
real game is reached through ``mod/TerrariaRLBridge``. The surrogate exists so
that calibration, exploit regression and RL rollouts are byte-reproducible and
cheap. Everything stochastic draws from a purpose-keyed stream (provenance.py);
wall-clock time is never read.
"""
from __future__ import annotations

import heapq
from collections import deque

from .provenance import ProvenanceError, pinned_seed_hash, sha256_bytes, stream
from .tasks import Task

W, H = 200, 100
CAVERN_Y, UNDERWORLD_Y = 50, 88
TICKS_PER_WINDOW = 27_000  # 12 simulated hours at 60 Hz
RHO = 0.15                 # salvage rate for raw inventory
DEATH_OPS_PENALTY = 6

(AIR, DIRT, STONE, WOOD, PLATFORM, COPPER, IRON, SILVER, GOLD, ASH, TREE, WITHERED, WORKBENCH,
 FURNACE, ANVIL, CAMPFIRE, TORCH, CHEST, CRYSTAL, DOOR, LAVA) = range(21)
TILE_NAMES = ("air", "dirt", "stone", "wood_block", "platform", "copper_ore", "iron_ore", "silver_ore",
              "gold_ore", "ash", "tree", "withered_tree", "workbench", "furnace", "anvil", "campfire",
              "torch", "chest", "life_crystal", "door", "lava")
TILE_CHARS = ".dsW=civgaTtBFAC*HLD~"
SOLID = frozenset({DIRT, STONE, WOOD, COPPER, IRON, SILVER, GOLD, ASH})
BARRIER = SOLID | {DOOR}
ORE_TILE = {"copper_ore": COPPER, "iron_ore": IRON, "silver_ore": SILVER, "gold_ore": GOLD}
TILE_ORE = {v: k for k, v in ORE_TILE.items()}
ORE_PICK = {SILVER: 2, GOLD: 2}  # minimum pickaxe tier
PLACEABLE = {"workbench": WORKBENCH, "furnace": FURNACE, "anvil": ANVIL, "campfire": CAMPFIRE,
             "torch": TORCH, "door": DOOR}
TIER_NAMES = ("Copper", "Wood", "Iron", "Silver", "Gold")


def _item(kind, v, **kw):
    return {"kind": kind, "v": v, "tier": 0, **kw}


ITEMS = {
    "wood": _item("raw", 0.2), "stone": _item("raw", 0.1), "gel": _item("raw", 0.3),
    "lens": _item("raw", 0.5), "mushroom": _item("raw", 0.4),
    "copper_ore": _item("raw", 0.6), "iron_ore": _item("raw", 1.0), "silver_ore": _item("raw", 1.5),
    "gold_ore": _item("raw", 2.0), "copper_bar": _item("raw", 1.5), "iron_bar": _item("raw", 2.5),
    "silver_bar": _item("raw", 3.5), "gold_bar": _item("raw", 4.5),
    "torch": _item("supply", 0.2), "wood_platform": _item("supply", 0.1), "wooden_arrow": _item("supply", 0.02),
    "door": _item("supply", 1.2), "workbench": _item("supply", 2.0), "furnace": _item("supply", 3.0),
    "anvil": _item("supply", 8.0), "campfire": _item("supply", 3.0),
    "lesser_healing_potion": _item("potion", 3.0), "ironskin_potion": _item("potion", 3.0),
    "suspicious_looking_eye": _item("summon", 4.0), "guide_voodoo_doll": _item("summon", 4.0),
    "copper_shortsword": _item("weapon", 1.0, dps=8, ranged=False),
    "wooden_sword": _item("weapon", 1.4, tier=1, dps=11, ranged=False),
    "wooden_bow": _item("weapon", 2.0, tier=1, dps=12, ranged=True),
    "wood_armor": _item("armor", 6.0, tier=1, defense=3),
    "copper_pickaxe": _item("pickaxe", 1.0, ticks=25),
}
for _n, (_metal, _dps, _def, _ticks) in enumerate((("iron", 20, 9, 14), ("silver", 28, 13, 10), ("gold", 36, 16, 7)), 2):
    _bar = ITEMS[f"{_metal}_bar"]["v"]
    ITEMS[f"{_metal}_broadsword"] = _item("weapon", 4 * _bar, tier=_n, dps=_dps, ranged=False)
    ITEMS[f"{_metal}_bow"] = _item("weapon", 4 * _bar, tier=_n, dps=_dps + 2, ranged=True)
    ITEMS[f"{_metal}_armor"] = _item("armor", 6 * _bar, tier=_n, defense=_def)
    ITEMS[f"{_metal}_pickaxe"] = _item("pickaxe", 5 * _bar, tier=_n, ticks=_ticks)

# name -> (output count, ingredients, required station or None)
RECIPES = {
    "workbench": (1, {"wood": 10}, None),
    "wood_platform": (2, {"wood": 1}, None),
    "torch": (3, {"wood": 1, "gel": 1}, None),
    "campfire": (1, {"wood": 10, "torch": 1}, None),
    "door": (1, {"wood": 6}, "workbench"),
    "wooden_sword": (1, {"wood": 7}, "workbench"),
    "wooden_bow": (1, {"wood": 10}, "workbench"),
    "wooden_arrow": (25, {"wood": 1, "stone": 1}, "workbench"),
    "wood_armor": (1, {"wood": 30}, "workbench"),
    "furnace": (1, {"stone": 20, "wood": 4, "torch": 3}, "workbench"),
    "lesser_healing_potion": (1, {"gel": 2, "mushroom": 1}, "workbench"),
    "suspicious_looking_eye": (1, {"lens": 6}, "workbench"),
    "copper_bar": (1, {"copper_ore": 2}, "furnace"),
    "iron_bar": (1, {"iron_ore": 2}, "furnace"),
    "silver_bar": (1, {"silver_ore": 2}, "furnace"),
    "gold_bar": (1, {"gold_ore": 2}, "furnace"),
    "anvil": (1, {"iron_bar": 3}, "workbench"),
}
for _metal in ("iron", "silver", "gold"):
    RECIPES[f"{_metal}_broadsword"] = (1, {f"{_metal}_bar": 4}, "anvil")
    RECIPES[f"{_metal}_bow"] = (1, {f"{_metal}_bar": 4}, "anvil")
    RECIPES[f"{_metal}_armor"] = (1, {f"{_metal}_bar": 6}, "anvil")
    RECIPES[f"{_metal}_pickaxe"] = (1, {f"{_metal}_bar": 5, "wood": 2}, "anvil")

# ------------------------------------------------------------ progression value
MILESTONE_VALUE = {
    "T1": 30, "T2": 60, "T3": 80, "T4": 100, "SHELTER": 25, "ARENA": 40, "WORKBENCH": 4, "FURNACE": 10,
    "ANVIL": 12, "CAVERN": 25, "UNDERWORLD": 45, "CRYSTAL1": 25, "CRYSTAL2": 15, "CRYSTAL3": 10,
    "HELLBRIDGE": 40, "SIEGE": 120, "BLOODMOON": 50, "SLIME_RAIN": 30, "KILL:eye": 120,
    "KILL:wall_of_flesh": 200,
}
BOSSES = {
    "eye": {"title": "Eye of Cthulhu", "hp": 2800, "dmg": 30, "credit": 180.0},
    "wall_of_flesh": {"title": "Wall of Flesh", "hp": 8000, "dmg": 50, "credit": 400.0},
}
GEAR_VALUE = (0, 6, 14, 22, 30)
KILL_VALUE, KILL_CAP = 0.2, 5   # per family: grinding is capped (exploit 4)
MOBS = {  # family -> (hp, dmg)
    "slime": (25, 6), "rain_slime": (25, 7), "zombie": (45, 14), "demon_eye": (60, 18),
    "blood_zombie": (50, 20), "goblin": (70, 16), "bat": (20, 10), "skeleton": (60, 20), "demon": (110, 32),
}
BLOODMOON_KILLS, SLIME_RAIN_KILLS, GOBLIN_ARMY_SIZE = 8, 10, 24

_TEMPLATES: dict[int, dict] = {}


def _template(seed: int) -> dict:
    """World template, keyed by the seed only (all task lenses share it)."""
    if seed in _TEMPLATES:
        return _TEMPLATES[seed]
    r = stream(seed, "terrain")
    surf, h = [], 24
    for x in range(W):
        if abs(x - W // 2) > 8 and r.random() < 0.35:
            h = min(28, max(20, h + r.choice((-1, 1))))
        surf.append(h)
    g = [bytearray(W) for _ in range(H)]
    for y in range(H):
        row = g[y]
        for x in range(W):
            s = surf[x]
            if y < s:
                t = AIR
            elif y < s + 7:
                t = DIRT
            elif y < UNDERWORLD_Y:
                t = STONE
            elif y < 90:
                t = ASH
            elif y < 96:
                t = AIR
            elif y < 98:
                t = ASH
            else:
                t = LAVA
            row[x] = t
    cave: list = []

    def carve(x, y):
        for cx in (x, x + 1):
            for cy in (y, y + 1):
                if 2 <= cx < W - 2 and 30 <= cy < 90 and g[cy][cx] != AIR:
                    g[cy][cx] = AIR
                    cave.append((cx, cy))

    # main descent: guaranteed route from near spawn to the underworld
    r = stream(seed, "descent")
    dx0 = r.choice((-14, 14))
    x, y = W // 2 + dx0, surf[W // 2 + dx0]
    descent = []
    while y < 89:
        g[y][x] = AIR
        if y >= 30:
            carve(x, y)
        descent.append((x, y))
        if r.random() < 0.62:
            y += 1
        else:
            x = min(W - 12, max(12, x + r.choice((-1, 1))))
    r = stream(seed, "tunnels")
    for _ in range(10):
        x, y = r.randint(10, W - 10), r.randint(36, 84)
        d = r.choice((-1, 1))
        for _ in range(60):
            carve(x, y)
            x = min(W - 4, max(3, x + (d if r.random() < 0.8 else 0)))
            y = min(85, max(32, y + r.choice((-1, 0, 0, 1))))

    def blob(cx, cy, n, tile, rr):
        x, y = cx, cy
        for _ in range(n * 3):
            if 2 <= x < W - 2 and 28 <= y < 87 and g[y][x] == STONE:
                g[y][x] = tile
                n -= 1
                if n <= 0:
                    return
            x += rr.choice((-1, 0, 1))
            y += rr.choice((-1, 0, 1))

    r = stream(seed, "ores")
    near = lambda depth: min(descent, key=lambda p: abs(p[1] - depth))
    for tile, depths, count, lo, hi in ((COPPER, (34, 42), 14, 30, 52), (IRON, (40, 46, 56, 64), 12, 38, 72),
                                        (SILVER, (60, 72), 8, 52, 82), (GOLD, (70, 80), 8, 60, 86)):
        for depth in depths:
            px, py = near(depth)
            blob(px + r.choice((-3, 3)), py, r.randint(9, 12), tile, r)
        for _ in range(count):
            blob(r.randint(6, W - 6), r.randint(lo, hi), r.randint(6, 10), tile, r)
    r = stream(seed, "features")
    spots = [p for p in cave if g[p[1]][p[0]] == AIR]
    for depth in (55, 68):
        px, py = near(depth)
        fx = px + r.choice((-2, 3))
        g[py][fx] = CRYSTAL
    for _ in range(4):
        fx, fy = r.choice([p for p in spots if p[1] >= 52])
        g[fy][fx] = CRYSTAL
    for _ in range(8):
        fx, fy = r.choice(spots)
        if g[fy][fx] == AIR:
            g[fy][fx] = CHEST
    px, py = near(44)
    g[py][px + r.choice((-2, 3))] = CHEST
    traps = set()
    for _ in range(16):
        p = r.choice(spots)
        if g[p[1]][p[0]] == AIR:
            traps.add(p)
    trap_free = set(descent[:12])
    traps -= trap_free
    r = stream(seed, "trees")
    trees, x = {}, 4
    while x < W - 4:
        if abs(x - W // 2) > 5 and abs(x - (W // 2 + dx0)) > 1:
            hgt = r.randint(5, 9)
            trees[x] = hgt
            for k in range(1, hgt + 1):
                g[surf[x] - k][x] = TREE
        x += r.randint(3, 6)
    wither = sorted(trees)
    stream(seed, "corruption").shuffle(wither)
    spawn = (W // 2, surf[W // 2] - 1)
    digest = sha256_bytes(b"".join(bytes(row) for row in g) + repr((sorted(traps), sorted(trees.items()), wither, spawn)).encode())
    _TEMPLATES[seed] = {"grid": g, "surf": surf, "cave": cave, "traps": traps, "trees": trees, "wither": wither,
                        "spawn": spawn, "hash": digest}
    return _TEMPLATES[seed]


def world_hash(seed: int) -> str:
    return _template(seed)["hash"]


class World:
    def __init__(self, task: Task, seed: int):
        tpl = _template(seed)
        pin = pinned_seed_hash(seed)
        if pin is not None and pin != tpl["hash"]:
            raise ProvenanceError(f"seed data for seed {seed} does not match its pinned hash; refusing to arm")
        self.task, self.seed = task, seed
        self.g = [bytearray(row) for row in tpl["grid"]]
        self.surf, self.cave = tpl["surf"], tpl["cave"]
        self.trees, self.wither_order, self.withered = dict(tpl["trees"]), list(tpl["wither"]), set()
        self.traps, self.known_traps = set(tpl["traps"]), set()
        self.spawn = self.pos = self.home = tpl["spawn"]
        self.placed, self.platforms, self.lights, self.known = set(), set(), set(), set()
        self.inv = {"copper_shortsword": 1, "copper_pickaxe": 1, "torch": 6}
        for k, v in task.kit.items():
            self.inv[k] = self.inv.get(k, 0) + v
        self.eq = {"weapon": "copper_shortsword", "armor": None, "pickaxe": "copper_pickaxe"}
        picks = [k for k in self.inv if ITEMS[k]["kind"] == "pickaxe"]
        self.eq["pickaxe"] = max(picks, key=lambda k: ITEMS[k]["tier"])
        for k in task.equip:
            self.eq[ITEMS[k]["kind"]] = k
        self.max_hp = self.hp = task.start_max_hp
        self.mana = 20
        self.window, self.tick, self.ops_penalty, self.done = 1, 0, 0, False
        self.milestones = set(task.start_milestones)
        self.kills: dict[str, int] = {}
        self.boss, self.boss_min, self.army = None, {}, None
        self.rosters: dict[str, list] = {}
        self.deaths, self.debt, self.died, self.window_deaths, self.last_death = 0, 0.0, False, 0, ""
        self.sick_until, self.buffs, self.stance = -1, {}, None
        self.crystals, self.max_depth, self.opened = 0, self.pos[1], set()
        self.notes: list = []
        self.end_sheltered = False
        self.phi0 = 0.0
        self._begin_window()
        self.refresh()
        self.phi0 = self.phi()

    # ------------------------------------------------------------ basic state
    @property
    def phase(self) -> str:
        return "Day" if self.window % 2 == 1 else "Night"

    @property
    def abs_tick(self) -> int:
        return (self.window - 1) * TICKS_PER_WINDOW + self.tick

    @property
    def defense(self) -> int:
        d = ITEMS[self.eq["armor"]]["defense"] if self.eq["armor"] else 0
        return d + (8 if self.buffs.get("ironskin", -1) > self.abs_tick else 0)

    @property
    def tech_tier(self) -> int:
        return max([n for n in range(1, 5) if f"T{n}" in self.milestones] or [0])

    @property
    def pick_ticks(self) -> int:
        return ITEMS[self.eq["pickaxe"]]["ticks"]

    def zone(self, pos=None) -> str:
        x, y = pos or self.pos
        if y >= UNDERWORLD_Y:
            return "underworld"
        if y >= CAVERN_Y:
            return "cavern"
        return "underground" if y > self.surf[x] + 3 else "surface"

    def has(self, item, n=1) -> bool:
        return self.inv.get(item, 0) >= n

    def slots(self) -> int:
        return sum((c + 998) // 999 for c in self.inv.values())

    def give(self, item, n=1) -> int:
        if n <= 0:
            return 0
        if item not in self.inv and self.slots() >= 50:
            return 0  # inventory full: overflow is lost
        self.inv[item] = self.inv.get(item, 0) + n
        return n

    def take(self, item, n=1):
        left = self.inv.get(item, 0) - n
        if left > 0:
            self.inv[item] = left
        else:
            self.inv.pop(item, None)
            for slot, name in self.eq.items():
                if name == item:
                    self.eq[slot] = None
            if self.eq["pickaxe"] is None:
                self.eq["pickaxe"] = "copper_pickaxe" if "copper_pickaxe" in self.inv else None
                if self.eq["pickaxe"] is None:  # the starter pickaxe is soulbound
                    self.inv["copper_pickaxe"] = 1
                    self.eq["pickaxe"] = "copper_pickaxe"

    def event(self, kind):
        """The active scripted event of this kind in the current window, if any."""
        for ev in self.task.events:
            if ev["kind"] == kind and ev["w"] <= self.window < ev["w"] + ev.get("dur", 1):
                return ev
        return None

    def set_tile(self, x, y, t, placed=False):
        p = (x, y)
        self.g[y][x] = t
        self.platforms.discard(p)
        self.lights.discard(p)
        self.placed.discard(p)
        if placed:
            self.placed.add(p)
        if t == PLATFORM:
            self.platforms.add(p)
        elif t in (TORCH, CAMPFIRE):
            self.lights.add(p)

    def light(self, x, y) -> float:
        sun = 1.0 if self.phase == "Day" else 0.2
        best = max(0.0, sun - 0.2 * max(0, y - self.surf[x]))
        for lx, ly in self.lights:
            d = abs(lx - x) + abs(ly - y)
            if d <= 8:
                best = max(best, 1.0 - d / 9.0)
        return round(best, 2)

    def near_tile(self, tile, radius=4) -> bool:
        px, py = self.pos
        for y in range(max(0, py - radius), min(H, py + radius + 1)):
            row = self.g[y]
            for x in range(max(0, px - radius), min(W, px + radius + 1)):
                if row[x] == tile:
                    return True
        return False

    # ------------------------------------------------------------- structures
    def enclosure(self, x, y, limit=400):
        """Flood fill through everything that is not a wall or a door. Returns the
        interior tile set, or None if the space leaks (exploit 9: a platform gap,
        a missing block or an open side all leak)."""
        if self.g[y][x] in BARRIER:
            return None
        seen, queue = {(x, y)}, deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if not (0 < nx < W - 1 and 0 < ny < H - 1):
                    return None
                if (nx, ny) in seen or self.g[ny][nx] in BARRIER:
                    continue
                seen.add((nx, ny))
                if len(seen) > limit:
                    return None
                queue.append((nx, ny))
        return seen

    def housing(self) -> dict:
        room = self.enclosure(*self.pos)
        if room is None:
            return {"valid": False, "enclosed": False, "reasons": ["not enclosed: the space around you leaks to the outside"]}
        tiles = {self.g[y][x] for x, y in room}
        reasons = []
        if len(room) < 20:
            reasons.append(f"too small: {len(room)} interior tiles, need 20")
        if not tiles & {TORCH, CAMPFIRE}:
            reasons.append("no light source inside")
        if WORKBENCH not in tiles:
            reasons.append("no workbench inside")
        return {"valid": not reasons, "enclosed": True, "interior_tiles": len(room), "reasons": reasons}

    def _runs(self, min_len, underworld):
        rows: dict[int, list] = {}
        for x, y in self.platforms:
            if (y >= UNDERWORLD_Y) == underworld:
                rows.setdefault(y, []).append(x)
        out = []
        for y, xs in rows.items():
            xs.sort()
            start = prev = xs[0]
            for x in xs[1:] + [None]:
                if x is None or x != prev + 1:
                    if prev - start + 1 >= min_len:
                        out.append((y, start, prev))
                    start = x
                prev = x
        return out

    def arena(self) -> dict:
        runs = sorted(self._runs(24, False))
        for i, (y0, a0, b0) in enumerate(runs):
            lo, hi, ys = a0, b0, [y0]
            for y1, a1, b1 in runs[i + 1:]:
                if y1 - y0 > 14 or y1 == ys[-1]:
                    continue
                if min(hi, b1) - max(lo, a1) + 1 >= 24:
                    lo, hi = max(lo, a1), min(hi, b1)
                    ys.append(y1)
            if len(ys) >= 3:
                cx, cy = (lo + hi) // 2, ys[len(ys) // 2]
                fire = any(self.g[ly][lx] == CAMPFIRE and abs(lx - cx) <= 30 and abs(ly - cy) <= 16 for lx, ly in self.lights)
                return {"valid": fire, "rows": len(ys), "x_range": [lo, hi], "y_range": [ys[0], ys[-1]],
                        "reasons": [] if fire else ["no campfire within reach of the platforms"]}
        return {"valid": False, "rows": len({r[0] for r in runs}), "reasons": ["need 3 platform rows, each 24+ wide and overlapping"]}

    def on_arena(self) -> bool:
        a = self.arena()
        if not a["valid"]:
            return False
        x, y = self.pos
        return a["x_range"][0] - 2 <= x <= a["x_range"][1] + 2 and a["y_range"][0] - 5 <= y <= a["y_range"][1] + 5

    def bridge_len(self, here=False) -> int:
        best = 0
        for y, a, b in self._runs(1, True):
            if here and not (a - 1 <= self.pos[0] <= b + 1 and abs(y - self.pos[1]) <= 2):
                continue
            best = max(best, b - a + 1)
        return best

    def refresh(self):
        m = self.milestones
        wt = ITEMS[self.eq["weapon"]]["tier"] if self.eq["weapon"] else 0
        at = ITEMS[self.eq["armor"]]["tier"] if self.eq["armor"] else 0
        for n in range(1, 5):
            if wt >= n and at >= n:
                m.add(f"T{n}")
        self.max_depth = max(self.max_depth, self.pos[1])
        if self.max_depth >= CAVERN_Y:
            m.add("CAVERN")
        if self.max_depth >= UNDERWORLD_Y:
            m.add("UNDERWORLD")
        if self.housing()["valid"]:
            m.add("SHELTER")
            self.home = self.pos
        if "ARENA" not in m and self.platforms and self.arena()["valid"]:
            m.add("ARENA")
        if "HELLBRIDGE" not in m and self.platforms and self.bridge_len() >= 60:
            m.add("HELLBRIDGE")
        if self.kills.get("blood_zombie", 0) >= BLOODMOON_KILLS:
            m.add("BLOODMOON")
        if self.kills.get("rain_slime", 0) >= SLIME_RAIN_KILLS:
            m.add("SLIME_RAIN")
        if self.army is not None and not self.army:
            m.add("SIEGE")
            self.army = None

    # -------------------------------------------------------------- valuation
    def phi(self, with_debt=True) -> float:
        """Net Progression Value: irreversible milestones + boss HP high-water
        credit + equipped gear + salvage-valued inventory - death debt."""
        v = 0.0
        for m in self.milestones:
            if m in MILESTONE_VALUE:
                v += MILESTONE_VALUE[m]
            elif m.startswith("CRAFT:"):  # set indicator: one credit per unique item, ever
                v += 1 + 2 * ITEMS[m[6:]]["tier"]
        for b, ratio in self.boss_min.items():
            v += BOSSES[b]["credit"] * (1.0 - ratio)
        for slot, scale in (("weapon", 1.0), ("armor", 1.0), ("pickaxe", 0.5)):
            if self.eq[slot]:
                v += GEAR_VALUE[ITEMS[self.eq[slot]]["tier"]] * scale
        v += RHO * sum(ITEMS[k]["v"] * c for k, c in self.inv.items())
        v += KILL_VALUE * sum(min(c, KILL_CAP) for c in self.kills.values())
        return round(v - (self.debt if with_debt else 0.0), 6)

    # ------------------------------------------------------------------ mobs
    def roster(self, zone=None) -> list:
        zone = zone or self.zone()
        if zone not in self.rosters:
            r = stream(self.seed, "mobs", self.window, zone)
            spec = []
            if zone == "surface":
                if self.phase == "Day":
                    spec = [("rain_slime", 16)] if self.event("slime_rain") else [("slime", 2)]
                elif self.event("blood_moon"):
                    spec = [("blood_zombie", 12), ("demon_eye", 2)]
                elif self.event("spawn_spike"):
                    spec = [("zombie", 8), ("demon_eye", 5)]
                else:
                    spec = [("zombie", 3), ("demon_eye", 2)]
            elif zone == "underground":
                spec = [("bat", 2)]
            elif zone == "cavern":
                spec = [("skeleton", 3)]
            else:
                spec = [("demon", 2)]
            chill = zone != "surface" and self.event("cave_chill")
            spike = zone == "surface" and self.phase == "Night" and self.event("spawn_spike")
            mobs = []
            for family, count in spec:
                hp, dmg = MOBS[family]
                for _ in range(count * (2 if chill else 1)):
                    mobs.append({"id": f"{zone[:2]}{self.window}_{len(mobs)}", "family": family,
                                 "hp": int(hp * r.uniform(0.9, 1.1)), "dmg": round(dmg * (1.5 if chill else 1.2 if spike else 1.0), 1)})
            self.rosters[zone] = mobs
        mobs = self.rosters[zone]
        return mobs + self.army if zone == "surface" and self.army else mobs

    def threat_multiplier(self) -> float:
        zone = self.zone()
        if zone == "surface":
            if self.phase == "Night" and self.event("blood_moon"):
                return 2.8
            if self.phase == "Night" and self.event("spawn_spike"):
                return 2.6 * 1.2
            return 1.0
        return 3.0 if self.event("cave_chill") else 1.0

    def _kill(self, mob):
        fam = mob["family"]
        self.kills[fam] = self.kills.get(fam, 0) + 1
        r = stream(self.seed, "drop", mob["id"])
        if "slime" in fam:
            self.give("gel", 1)
        elif fam == "demon_eye" and r.random() < 0.6:
            self.give("lens", 1)
        elif fam == "skeleton":
            self.give("wooden_arrow", 5)
        for pool in ([self.army] if self.army is not None else []) + list(self.rosters.values()):
            if mob in pool:
                pool.remove(mob)

    def hurt(self, dmg, cause) -> bool:
        self.hp -= max(0, int(round(dmg)))
        if self.hp <= 0:
            self._die(cause)
            return True
        return False

    def _die(self, cause):
        self.deaths += 1
        self.window_deaths += 1
        self.died = True
        self.last_death = cause
        for k in [k for k in self.inv if ITEMS[k]["kind"] == "raw"]:
            self.take(k, (self.inv[k] + 1) // 2)
        # A death forfeits up to death_fee of progression earned this episode; it
        # can erase gains but never drives the ledger below the starting value.
        gain = self.phi(with_debt=False) - self.phi0 - self.debt
        self.debt += min(self.task.death_fee, max(0.0, gain))
        self.boss = None
        self.stance = None
        self.pos = self.home
        self.hp = max(1, self.max_hp // 2)
        self.ops_penalty = DEATH_OPS_PENALTY

    def harass(self) -> int:
        """Chip damage taken while working in the open among live hostiles."""
        if self.died or self.enclosure(*self.pos) is not None:
            return 0
        mobs = sorted(self.roster(), key=lambda m: -m["dmg"])[:13]
        dmg = 0.12 * sum(max(1.0, m["dmg"] - self.defense / 2) for m in mobs)
        if dmg < 1:
            return 0
        before = self.hp
        self.hurt(dmg, "overrun while working in the open")
        return before - max(self.hp, 0) if not self.died else before

    # ------------------------------------------------------------- movement
    def _cost(self, y, nx, ny):
        t = self.g[ny][nx]
        if t == LAVA or (t in SOLID and (nx, ny) in self.placed):
            return None  # lava and player-built walls are never dug through; use a door
        c = 5 + (self.pick_ticks if t in SOLID else 0) + (5 if ny < y else 0)
        if (nx, ny) in self.known_traps:
            c += 3000
        return c

    def find_path(self, gx, gy, max_expand=5000):
        gx, gy = min(W - 2, max(1, gx)), min(H - 3, max(1, gy))
        start, goal = self.pos, (gx, gy)
        h0 = 7 * (abs(gx - start[0]) + abs(gy - start[1]))
        heap, came, best_g, best, n = [(h0, 0, start)], {}, {start: 0}, (h0, start), 0
        while heap and n < max_expand:
            _, gc, cur = heapq.heappop(heap)
            if cur == goal:
                break
            if gc > best_g[cur]:
                continue
            n += 1
            x, y = cur
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (1 <= nx <= W - 2 and 1 <= ny <= H - 3):
                    continue
                c = self._cost(y, nx, ny)
                if c is None:
                    continue
                ng = gc + c
                if ng < best_g.get((nx, ny), 1 << 30):
                    best_g[(nx, ny)] = ng
                    came[(nx, ny)] = cur
                    hh = 7 * (abs(gx - nx) + abs(gy - ny))
                    if (hh, (nx, ny)) < best:
                        best = (hh, (nx, ny))
                    heapq.heappush(heap, (ng + hh, ng, (nx, ny)))
        end = goal if goal in came or goal == start else best[1]
        path = []
        while end != start:
            path.append(end)
            end = came[end]
        return path[::-1], goal

    def walk(self, gx, gy, budget=300) -> dict:
        """A* toward the goal, digging where needed, for at most ``budget`` ticks."""
        path, goal = self.find_path(gx, gy)
        used, steps, trap, dug = 0, 0, None, 0
        for nx, ny in path:
            c = self._cost(self.pos[1], nx, ny)
            if c >= 3000:
                c -= 3000
            if used + c > budget:
                break
            t = self.g[ny][nx]
            if t in SOLID:
                self.dig_yield(nx, ny)
                self.set_tile(nx, ny, AIR)
                dug += 1
            used += c
            steps += 1
            self.pos = (nx, ny)
            if (nx, ny) in self.traps and (nx, ny) not in self.known_traps:
                self.known_traps.add((nx, ny))
                trap = max(10, 35 - self.defense)
                break
        self.tick += used
        self.max_depth = max(self.max_depth, self.pos[1])
        out = {"ticks": used, "steps": steps, "dug": dug, "arrived": self.pos == goal, "position": list(self.pos)}
        if trap:
            out["trap_damage"] = trap
            self.hurt(trap, "a hidden trap")
        return out

    def dig_yield(self, x, y):
        """Natural ore and stone drop an item; anything the player placed drops
        nothing (exploit 3: place/break farming has no mining value)."""
        t = self.g[y][x]
        if (x, y) in self.placed:
            return
        if t == STONE:
            self.give("stone", 1)
        elif t in TILE_ORE and ITEMS[self.eq["pickaxe"]]["tier"] >= ORE_PICK.get(t, 0):
            self.give(TILE_ORE[t], 1)

    def reveal(self, radius, pos=None) -> list:
        px, py = pos or self.pos
        found = []
        for y in range(max(1, py - radius), min(H - 1, py + radius + 1)):
            row = self.g[y]
            for x in range(max(1, px - radius), min(W - 1, px + radius + 1)):
                if row[x] in TILE_ORE or row[x] in (CHEST, CRYSTAL):
                    if (x, y) not in self.known:
                        self.known.add((x, y))
                        found.append((x, y))
        for p in self.traps:
            if abs(p[0] - px) <= radius // 2 and abs(p[1] - py) <= radius // 2:
                self.known_traps.add(p)
        return found

    def nearest_known(self, tiles, maxd=40):
        px, py = self.pos
        best = None
        for x, y in self.known:
            if self.g[y][x] in tiles:
                d = abs(x - px) + abs(y - py)
                if d <= maxd and (best is None or (d, x, y) < best):
                    best = (d, x, y)
        return best

    # ---------------------------------------------------------------- combat
    def targets(self) -> list:
        out = [{"id": f"boss_{self.boss['kind']}", "family": "boss", "name": BOSSES[self.boss["kind"]]["title"],
                "hp": int(self.boss["hp"]), "max_hp": BOSSES[self.boss["kind"]]["hp"]}] if self.boss else []
        return out + [dict(m) for m in self.roster()]

    def spawn_boss(self, kind):
        self.boss = {"kind": kind, "hp": float(BOSSES[kind]["hp"]), "advance": 0}
        self.boss_min.setdefault(kind, 1.0)

    def burst(self, target_id=None, weapon=None, kite=0.0, dodge=0.0, vertical=False) -> dict:
        wname = weapon or self.eq["weapon"]
        it = ITEMS.get(wname or "")
        if not it or it["kind"] != "weapon" or not self.has(wname):
            return {"ok": False, "error": f"no usable weapon '{wname}'"}
        if it["ranged"] and not self.has("wooden_arrow", 10):
            return {"ok": False, "error": "out of arrows (a burst with a bow needs 10)"}
        mobs = self.roster()
        on_boss = self.boss is not None and (target_id is None or str(target_id).startswith("boss"))
        if not on_boss:
            if str(target_id or "").startswith("boss"):
                return {"ok": False, "error": "no boss is present"}
            if not mobs:
                return {"ok": False, "error": "no hostile in range"}
            first = next((m for m in mobs if m["id"] == target_id), None)
            if target_id is not None and first is None:
                return {"ok": False, "error": f"unknown target '{target_id}'"}
        kite, dodge = min(40.0, max(0.0, float(kite))), min(1.0, max(0.0, float(dodge)))
        r = stream(self.seed, "burst", self.window, self.tick, self.hp)
        dealt = it["dps"] * 12 * (1 - 0.3 * dodge) * r.uniform(0.92, 1.08)
        if it["ranged"]:
            self.take("wooden_arrow", 10)
        on_bridge = self.pos[1] >= UNDERWORLD_Y and self.bridge_len(here=True) >= 20
        stance = bool(vertical) and (on_bridge or self.on_arena())
        exposure = (max(0.3, 1 - kite / 16) if it["ranged"] else 1.0) * (1 - 0.5 * dodge) * (0.6 if stance else 1.0)
        out = {"ok": True, "weapon": wname, "damage_dealt": int(dealt), "kills": []}
        if on_boss:
            b, spec = self.boss, BOSSES[self.boss["kind"]]
            dmg = spec["dmg"]
            if b["kind"] == "eye" and b["hp"] <= 0.5 * spec["hp"]:
                dmg *= 1.6  # silent phase-2 enrage: it charges, so distance alone stops helping
                if dodge < 0.5 and not stance:
                    exposure = max(exposure, 0.6)
            if b["kind"] == "wall_of_flesh":
                b["advance"] += 5
                if self.bridge_len(here=True) - b["advance"] < 0:
                    exposure *= 3.0  # out of bridge: cornered
                    out["cornered"] = True
            b["hp"] -= dealt
            ratio = max(0.0, b["hp"] / spec["hp"])
            self.boss_min[b["kind"]] = min(self.boss_min[b["kind"]], ratio)
            taken = max(1.0, dmg - self.defense / 2) * 1.2 * exposure
            out.update(target=f"boss_{b['kind']}", target_hp=max(0, int(b["hp"])), target_max_hp=spec["hp"])
            if b["hp"] <= 0:
                self.milestones.add(f"KILL:{b['kind']}")
                out["kills"].append(f"boss_{b['kind']}")
                self.boss, taken = None, taken * 0.5
        else:
            order = [first] if target_id is not None else []
            order += sorted((m for m in mobs if m not in order), key=lambda m: (m["hp"], m["id"]))
            left = dealt
            for m in order:  # overkill carries over to the next hostile
                if left <= 0:
                    break
                hit = min(left, m["hp"])
                m["hp"] -= hit
                left -= hit
                if m["hp"] <= 0:
                    out["kills"].append(m["id"])
                    self._kill(m)
            alive = sorted(self.roster(), key=lambda m: -m["dmg"])[:2]
            taken = sum(max(1.0, m["dmg"] - self.defense / 2) for m in alive) * 0.6 * exposure
            if self.stance == "funnel":
                taken *= 0.6
            out["hostiles_left"] = len(self.roster())
        if taken >= 1 and any(self.g[ly][lx] == CAMPFIRE and abs(lx - self.pos[0]) + abs(ly - self.pos[1]) <= 24 for lx, ly in self.lights):
            taken = max(0.0, taken - 6)
        self.tick += 180
        out["damage_taken"] = int(round(taken))
        if self.hurt(taken, "slain in combat"):
            out["died"] = True
        return out

    # ------------------------------------------------------------ settlement
    def _begin_window(self):
        self.tick, self.died, self.window_deaths, self.stance, self.rosters = 0, False, 0, None, {}
        for ev in self.task.events:
            if ev["w"] != self.window:
                continue
            if ev["kind"] == "goblin_siege":
                r = stream(self.seed, "army")
                hp, dmg = MOBS["goblin"]
                self.army = [{"id": f"g_{i}", "family": "goblin", "hp": int(hp * r.uniform(0.9, 1.1)), "dmg": dmg}
                             for i in range(GOBLIN_ARMY_SIZE)]
            elif ev["kind"] == "eye_spawn" and "KILL:eye" not in self.milestones and self.boss is None:
                self.spawn_boss("eye")

    def settle(self) -> dict:
        """Play out the rest of the window with the player standing where they are."""
        notes = []
        self.end_sheltered = self.housing()["valid"]
        enclosed = self.enclosure(*self.pos) is not None
        died_before = self.died
        if self.boss:  # exploit 8: a live boss fight at settlement resolves as a loss
            spec = BOSSES[self.boss["kind"]]
            guard = 0.5 if (self.on_arena() or (self.pos[1] >= UNDERWORLD_Y and self.bridge_len(here=True) >= 20)) else 1.0
            notes.append(f"The {spec['title']} overwhelmed you and left.")
            self.boss = None
            self.hurt(max(1.0, spec["dmg"] - self.defense / 2) * 5 * guard, f"the {spec['title']}")
        if not self.died and not enclosed:
            k = 0.5 if (self.phase == "Day" and self.zone() == "surface") else 2.0
            mobs = sorted(self.roster(), key=lambda m: -m["dmg"])[:6]
            self.hurt(k * sum(max(1.0, m["dmg"] - self.defense / 2) for m in mobs), "caught in the open")
        if self.died and not died_before:
            notes.append(f"You were slain ({self.last_death}).")
        if not self.died:
            self.hp = min(self.max_hp, self.hp + int(self.max_hp * (0.4 if enclosed else 0.1)))
        alive = [x for x in self.wither_order if x in self.trees and x not in self.withered]
        for x in alive[:self.task.corruption_rate]:  # silent corruption spread
            self.withered.add(x)
            for k in range(1, self.trees[x] + 1):
                if self.g[self.surf[x] - k][x] == TREE:
                    self.g[self.surf[x] - k][x] = WITHERED
        if self.army is not None and not self.army:
            notes.append("The goblin army has been defeated!")
        self.refresh()
        out = {"window": self.window, "died": self.died, "deaths": self.window_deaths, "sheltered": self.end_sheltered, "notes": notes}
        self.window += 1
        if self.window > self.task.windows:
            self.window, self.done = self.task.windows, True
        else:
            self._begin_window()
        return out

    def alerts(self) -> tuple[str, list]:
        """Announced and rumored channels. Silent events never appear here."""
        w, out, threat = self.window, [], []
        if self.boss:
            threat.append((9, "BOSS_ACTIVE"))
            out.append({"channel": "announced", "text": f"{BOSSES[self.boss['kind']]['title']} has awoken!"})
        if self.army:
            threat.append((8, "SIEGE_ACTIVE"))
            out.append({"channel": "announced", "text": f"A goblin army is attacking! {len(self.army)} remain."})
        for ev in self.task.events:
            k, ew, lead = ev["kind"], ev["w"], ev["w"] - w
            if k == "goblin_siege" and ev.get("announce", ew) <= w < ew:
                threat.append((5, "ANNOUNCED_GOBLIN_ARMY"))
                out.append({"channel": "announced", "text": f"A goblin army is approaching! It arrives in {lead} window(s)."})
            elif k == "blood_moon":
                if w == ew:
                    threat.append((7, "BLOOD_MOON"))
                    out.append({"channel": "announced", "text": "The Blood Moon is rising..."})
                elif ev.get("announce", ew) <= w < ew:
                    threat.append((5, "ANNOUNCED_BLOOD_MOON"))
                    out.append({"channel": "announced", "text": f"The moon looks wrong. A Blood Moon rises in {lead} window(s)."})
            elif k == "slime_rain":
                if w == ew:
                    threat.append((6, "SLIME_RAIN"))
                    out.append({"channel": "announced", "text": "Slime is falling from the sky!"})
                elif ev.get("announce", ew) <= w < ew:
                    threat.append((4, "ANNOUNCED_SLIME_RAIN"))
                    out.append({"channel": "announced", "text": f"The sky looks strange. Slime rain starts in {lead} window(s)."})
            elif k == "eye_spawn" and 1 <= lead <= 2 and "KILL:eye" not in self.milestones and not self.boss:
                threat.append((3, "RUMORED_BOSS"))
                out.append({"channel": "rumored", "text": "You feel an evil presence watching you..."})
            elif k == "cave_chill" and -ev.get("dur", 1) < lead <= 1:
                threat.append((2, "RUMORED_CAVE"))
                out.append({"channel": "rumored", "text": "Chilling winds blow from the depths..."})
        return (max(threat)[1] if threat else "CALM"), out

    def state_hash(self) -> str:
        blob = b"".join(bytes(row) for row in self.g) + repr((
            self.pos, self.hp, self.max_hp, sorted(self.inv.items()), sorted(self.eq.items(), key=str),
            sorted(self.milestones), sorted(self.kills.items()), sorted(self.boss_min.items()), self.deaths,
            self.debt, self.window, self.tick, sorted(self.placed), sorted(self.known))).encode()
        return sha256_bytes(blob)
