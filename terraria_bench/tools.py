"""The 24-tool MCP surface, windowed operation metering, and sealed episode ledger.

Every tool call returns a dict with ``header`` (the deterministic one-line status
banner) and ``text`` (header + JSON body). All tools except ``terraria_status``
spend 1 operation from the current 12-hour window budget. Spending the final
operation in a window or calling ``end_window`` settles the window and evaluates
any hidden SLA checkpoints scheduled for that window.
"""
from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

from .engine import (
    AIR,
    ANVIL,
    BOSSES,
    CAMPFIRE,
    CAVERN_Y,
    CHEST,
    COPPER,
    CRYSTAL,
    DIRT,
    DOOR,
    FURNACE,
    GOLD,
    H,
    IRON,
    ITEMS,
    LAVA,
    ORE_PICK,
    ORE_TILE,
    PLACEABLE,
    PLATFORM,
    RECIPES,
    RHO,
    SILVER,
    SOLID,
    STONE,
    TICKS_PER_WINDOW,
    TIER_NAMES,
    TILE_CHARS,
    TILE_NAMES,
    TILE_ORE,
    TORCH,
    TREE,
    UNDERWORLD_Y,
    W,
    WITHERED,
    WOOD,
    WORKBENCH,
    World,
    world_hash,
)
from .grader import sla_met
from .provenance import run_manifest, seal, stream
from .tasks import TASKS, Task

TOOL_NAMES: tuple[str, ...] = (
    # Family 1: Status & Time (2)
    "terraria_status",
    "end_window",
    # Family 2: Observation & Spatial Inspection (5)
    "observe_viewport",
    "scan_biome_and_ores",
    "inspect_inventory",
    "list_craftable_recipes",
    "inspect_housing_and_arena",
    # Family 3: Gathering, Mining & Navigation (5)
    "navigate_to",
    "chop_trees",
    "mine_vein",
    "explore_cavern",
    "loot_container_or_crystal",
    # Family 4: Crafting, Building & Blueprint Synthesis (6)
    "craft_item",
    "place_station_or_furniture",
    "build_blueprint",
    "equip_item",
    "consume_buff_or_heal",
    "manage_inventory_trash",
    # Family 5: Combat & Boss Encounter Control (6)
    "summon_or_trigger_boss",
    "engage_combat_burst",
    "retreat_and_stabilize",
    "deploy_combat_arena_tactics",
    "query_threat_telemetry",
    "record_tactical_checkpoint",
)
assert len(TOOL_NAMES) == 24

# Palette RGB bytes for deterministic stdlib PNG viewport generation (21 tile types + player)
_PALETTE_RGB = (
    (18, 22, 38), (110, 74, 45), (120, 124, 132), (156, 108, 60), (190, 148, 90),
    (184, 105, 58), (175, 165, 160), (192, 205, 218), (242, 202, 68), (72, 66, 70),
    (46, 138, 56), (92, 78, 88), (168, 118, 62), (140, 80, 50), (130, 140, 150),
    (245, 130, 40), (255, 220, 90), (210, 165, 55), (235, 65, 115), (135, 92, 48),
    (245, 70, 20), (60, 230, 255),
)


def _encode_png(indices: list[bytes], width: int, height: int) -> str:
    """Deterministic 8-bit indexed-RGB PNG encoded with stdlib struct + zlib."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = bytearray()
    for row in indices:
        raw.append(0)
        for idx in row:
            raw.extend(_PALETTE_RGB[idx if idx < len(_PALETTE_RGB) else 0])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "terraria_status": {"cost": 0, "description": "Free status probe (0 ops). Returns vitals, window/tick, and alerts.", "params": {}},
    "end_window": {"cost": 1, "description": "Advance to the end of the current 12-hour window and run settlement.", "params": {}},
    "observe_viewport": {"cost": 1, "description": "Return 61x35 ego-centric ASCII grid, lighting, entities, and Base64 PNG.", "params": {"width": "int=61", "height": "int=35", "include_png": "bool=True"}},
    "scan_biome_and_ores": {"cost": 1, "description": "Scan nearby exposed ores, chests, life crystals, and known hazards.", "params": {"radius": "int=28"}},
    "inspect_inventory": {"cost": 1, "description": "List inventory slots, equipped loadout, and salvage vs crafted value.", "params": {}},
    "list_craftable_recipes": {"cost": 1, "description": "Return currently craftable items and missing station/material prerequisites.", "params": {}},
    "inspect_housing_and_arena": {"cost": 1, "description": "Evaluate flood-fill housing validity, 3-tier boss arena, and hellbridge.", "params": {}},
    "navigate_to": {"cost": 1, "description": "A* pathfind and dig toward (x, y) for up to 300 ticks.", "params": {"x": "int", "y": "int", "budget": "int=300"}},
    "chop_trees": {"cost": 1, "description": "Harvest nearest living surface trees within radius.", "params": {"radius": "int=28", "max_trees": "int=3"}},
    "mine_vein": {"cost": 1, "description": "Pathfind to and excavate a target ore or stone cluster.", "params": {"x": "int|None=None", "y": "int|None=None", "tile_type": "str='any'", "max_tiles": "int=8"}},
    "explore_cavern": {"cost": 1, "description": "Frontier subterranean descent with automatic torch placement at light < 0.25.", "params": {"target_depth": "int=58", "auto_torch": "bool=True"}},
    "loot_container_or_crystal": {"cost": 1, "description": "Loot a discovered chest or consume a Life Crystal.", "params": {"x": "int|None=None", "y": "int|None=None", "kind": "str='any'"}},
    "craft_item": {"cost": 1, "description": "Craft item_id (quantity) if within 4 tiles of required station.", "params": {"item_id": "str", "quantity": "int=1"}},
    "place_station_or_furniture": {"cost": 1, "description": "Place station, block, platform, door, torch, or campfire at (dx, dy).", "params": {"item_id": "str", "dx": "int=0", "dy": "int=0"}},
    "build_blueprint": {"cost": 1, "description": "Construct a shelter, multi-tier arena, hellbridge, or 2D tile matrix blueprint.", "params": {"preset": "str='shelter'", "matrix": "list|None=None", "origin_dx": "int=-4", "origin_dy": "int=-4"}},
    "equip_item": {"cost": 1, "description": "Equip a weapon, armor piece, or pickaxe from inventory.", "params": {"item_id": "str"}},
    "consume_buff_or_heal": {"cost": 1, "description": "Use a healing or buff potion (respects 60s Potion Sickness).", "params": {"item_id": "str='lesser_healing_potion'"}},
    "manage_inventory_trash": {"cost": 1, "description": "Discard quantity of item_id from inventory.", "params": {"item_id": "str", "quantity": "int=1"}},
    "summon_or_trigger_boss": {"cost": 1, "description": "Use a boss summoning item (eye or wall_of_flesh).", "params": {"boss": "str='eye'"}},
    "engage_combat_burst": {"cost": 1, "description": "Run 180-tick reactive combat burst against boss or hostiles.", "params": {"target_id": "str|None=None", "weapon": "str|None=None", "kite_distance": "float=12.0", "dodge_priority": "float=0.5", "vertical_platform_use": "bool=True"}},
    "retreat_and_stabilize": {"cost": 1, "description": "Evade toward shelter/arena, enter funnel stance, and optionally heal.", "params": {"use_potion": "bool=True"}},
    "deploy_combat_arena_tactics": {"cost": 1, "description": "Deploy tactical platforms/campfire or grapple to arena tier.", "params": {"tactic": "str='grapple_tier'"}},
    "query_threat_telemetry": {"cost": 1, "description": "Return boss HP high-water mark, phase state, and hostile vectors.", "params": {}},
    "record_tactical_checkpoint": {"cost": 1, "description": "Log structured tactical hypothesis and plan to episode provenance.", "params": {"hypothesis": "str", "plan": "str=''", "confidence": "float=0.8"}},
}


class Episode:
    """Metered tool session over a seeded World that records and seals the episode ledger."""

    def __init__(self, task: str | Task, seed: int, redact_seed: bool = False):
        self.task: Task = TASKS[task] if isinstance(task, str) else task
        self.seed = int(seed)
        self.redact_seed = redact_seed
        self.world = World(self.task, self.seed)
        self.window_budget = self.task.budget
        self.ops_left = self.window_budget
        self.attended = False
        self.windows_log: list[dict[str, Any]] = []
        self.actions_log: list[dict[str, Any]] = []
        self._sealed_ledger: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.world.done

    def header(self) -> str:
        w = self.world
        threat, _ = w.alerts()
        tier = w.tech_tier
        tier_label = f"{tier} ({TIER_NAMES[tier]})"
        seed_part = "World: locked | " if self.redact_seed else f"Seed: {self.seed} | "
        return (
            f"[{seed_part}Win: {w.window}/{self.task.windows} ({w.phase}) | "
            f"Ops: {self.ops_left}/{self.window_budget} | HP: {w.hp}/{w.max_hp} | "
            f"Mana: {w.mana}/20 | Def: {w.defense} | NetProg: {w.phi():.1f} | "
            f"TechTier: {tier_label} | Threat: {threat}]"
        )

    def _settle_current_window(self) -> dict[str, Any]:
        w_idx = self.world.window
        used = self.window_budget - self.ops_left
        res = self.world.settle()
        sla_checks = [[kind, sla_met(self.world, kind, res["sheltered"])] for dw, kind in self.task.sla if dw == w_idx]
        rec = {
            "w": w_idx,
            "attended": self.attended,
            "ops_used": used,
            "phi": self.world.phi(),
            "deaths": res["deaths"],
            "sheltered": res["sheltered"],
            "sla": sla_checks,
            "notes": res["notes"],
        }
        self.windows_log.append(rec)
        if not self.world.done:
            self.window_budget = max(1, self.task.budget - self.world.ops_penalty)
            self.world.ops_penalty = 0
            self.ops_left = self.window_budget
            self.attended = False
        else:
            self.ops_left = 0
        return rec

    def step(self, tool_name: str, **kwargs) -> dict[str, Any]:
        # Accept the typo alias from the prompt ('engage_combat_ burst') cleanly
        if tool_name == "engage_combat_ burst":
            tool_name = "engage_combat_burst"
        if tool_name not in TOOL_SCHEMAS:
            return self._wrap({"ok": False, "error": f"unknown tool '{tool_name}'"})
        if self.world.done:
            return self._wrap({"ok": False, "error": "episode already finished", "done": True})

        self.attended = True
        cost = TOOL_SCHEMAS[tool_name]["cost"]
        self.actions_log.append({"w": self.world.window, "tool": tool_name, "args": kwargs})

        if tool_name == "terraria_status":
            body = self._tool_terraria_status()
            return self._wrap(body)

        self.ops_left = max(0, self.ops_left - cost)
        if tool_name == "end_window":
            settled = self._settle_current_window()
            return self._wrap({"ok": True, "window_settled": settled, "done": self.world.done})

        handler = getattr(self, f"_tool_{tool_name}")
        body = handler(**kwargs)
        if self.ops_left <= 0 and not self.world.done:
            body["window_settled"] = self._settle_current_window()
            body["done"] = self.world.done
        return self._wrap(body)

    def _wrap(self, body: dict[str, Any]) -> dict[str, Any]:
        hdr = self.header()
        payload = {"header": hdr, **body}
        payload["text"] = f"{hdr}\n{json.dumps(body, sort_keys=True)}"
        return payload

    def finalize(self, calibration: dict | None = None) -> dict[str, Any]:
        """Settle any remaining windows (unattended if quit early) and seal the episode ledger."""
        if self._sealed_ledger is not None:
            return self._sealed_ledger
        while not self.world.done:
            self._settle_current_window()
        raw = {
            "task": self.task.id,
            "seed": self.seed,
            "world_hash": world_hash(self.seed),
            "state_hash": self.world.state_hash(),
            "phi0": self.world.phi0,
            "phi_final": self.world.phi(),
            "windows": self.windows_log,
            "milestones": sorted(self.world.milestones),
            "deaths": self.world.deaths,
            "manifest": run_manifest(calibration),
        }
        self._sealed_ledger = seal(raw)
        return self._sealed_ledger

    # --------------------------------------------------------- Family 1: Status
    def _tool_terraria_status(self) -> dict[str, Any]:
        w = self.world
        threat, alerts = w.alerts()
        return {
            "ok": True,
            "window": w.window,
            "total_windows": self.task.windows,
            "phase": w.phase,
            "tick": w.tick,
            "ticks_per_window": TICKS_PER_WINDOW,
            "ops_left": self.ops_left,
            "ops_budget": self.window_budget,
            "position": list(w.pos),
            "zone": w.zone(),
            "light": w.light(*w.pos),
            "hp": w.hp,
            "max_hp": w.max_hp,
            "mana": w.mana,
            "defense": w.defense,
            "tech_tier": w.tech_tier,
            "equipped": dict(w.eq),
            "potion_sickness_ticks": max(0, w.sick_until - w.abs_tick),
            "ironskin_ticks": max(0, w.buffs.get("ironskin", -1) - w.abs_tick),
            "threat": threat,
            "alerts": alerts,
        }

    # ---------------------------------------------------- Family 2: Observation
    def _tool_observe_viewport(self, width: int = 61, height: int = 35, include_png: bool = True) -> dict[str, Any]:
        w = self.world
        width = min(81, max(15, int(width) | 1))
        height = min(51, max(11, int(height) | 1))
        w.reveal(max(width, height) // 2)
        px, py = w.pos
        hw, hh = width // 2, height // 2
        ascii_rows: list[str] = []
        idx_rows: list[bytes] = []
        for dy in range(-hh, hh + 1):
            y = min(H - 1, max(0, py + dy))
            row = w.g[y]
            chars = []
            idxs = bytearray()
            for dx in range(-hw, hw + 1):
                x = min(W - 1, max(0, px + dx))
                if dx == 0 and dy == 0:
                    chars.append("@")
                    idxs.append(21)
                else:
                    t = row[x]
                    chars.append(TILE_CHARS[t])
                    idxs.append(t)
            ascii_rows.append("".join(chars))
            idx_rows.append(bytes(idxs))
        out: dict[str, Any] = {
            "ok": True,
            "center": [px, py],
            "shape": [width, height],
            "zone": w.zone(),
            "light_at_player": w.light(px, py),
            "ascii_grid": ascii_rows,
            "entities": w.targets(),
        }
        if include_png:
            out["viewport_png_b64"] = _encode_png(idx_rows, width, height)
        return out

    def _tool_scan_biome_and_ores(self, radius: int = 28) -> dict[str, Any]:
        w = self.world
        radius = min(60, max(6, int(radius)))
        w.reveal(radius)
        px, py = w.pos
        counts: dict[str, int] = {}
        nearest: dict[str, dict[str, int]] = {}
        for y in range(max(1, py - radius), min(H - 1, py + radius + 1)):
            row = w.g[y]
            for x in range(max(1, px - radius), min(W - 1, px + radius + 1)):
                t = row[x]
                if t in TILE_ORE or t in (CHEST, CRYSTAL, LAVA):
                    name = TILE_NAMES[t]
                    counts[name] = counts.get(name, 0) + 1
                    d = abs(x - px) + abs(y - py)
                    if name not in nearest or d < nearest[name]["dist"]:
                        nearest[name] = {"x": x, "y": y, "dx": x - px, "dy": y - py, "dist": d}
        living_trees = [x for x in w.trees if x not in w.withered and abs(x - px) <= radius]
        known_traps = [[tx, ty] for tx, ty in sorted(w.known_traps) if abs(tx - px) <= radius and abs(ty - py) <= radius]
        return {
            "ok": True,
            "zone": w.zone(),
            "surface_y": w.surf[px],
            "depth": py,
            "max_depth": w.max_depth,
            "living_trees_nearby": len(living_trees),
            "counts": counts,
            "nearest": nearest,
            "known_traps": known_traps,
        }

    def _tool_inspect_inventory(self) -> dict[str, Any]:
        w = self.world
        items = {
            k: {
                "count": c,
                "kind": ITEMS[k]["kind"],
                "tier": ITEMS[k]["tier"],
                "salvage_value": round(RHO * ITEMS[k]["v"] * c, 4),
            }
            for k, c in sorted(w.inv.items())
        }
        return {
            "ok": True,
            "slots_used": w.slots(),
            "slots_max": 50,
            "equipped": dict(w.eq),
            "tech_tier": w.tech_tier,
            "total_salvage_value": round(RHO * sum(ITEMS[k]["v"] * c for k, c in w.inv.items()), 4),
            "items": items,
        }

    def _tool_list_craftable_recipes(self) -> dict[str, Any]:
        w = self.world
        craftable = []
        missing = {}
        for name, (out_n, reqs, station) in sorted(RECIPES.items()):
            station_ok = station is None or w.near_tile(PLACEABLE[station], radius=4)
            short = {ing: need - w.inv.get(ing, 0) for ing, need in reqs.items() if w.inv.get(ing, 0) < need}
            if station_ok and not short:
                max_batches = min(w.inv.get(ing, 0) // need for ing, need in reqs.items())
                craftable.append({"item_id": name, "output_per_batch": out_n, "max_batches": max_batches, "station": station})
            else:
                missing[name] = {"missing_station": None if station_ok else station, "missing_ingredients": short}
        return {"ok": True, "craftable": craftable, "missing": missing}

    def _tool_inspect_housing_and_arena(self) -> dict[str, Any]:
        w = self.world
        w.refresh()
        return {
            "ok": True,
            "housing": w.housing(),
            "arena": w.arena(),
            "on_arena": w.on_arena(),
            "hellbridge_len": w.bridge_len(),
            "on_hellbridge": w.pos[1] >= UNDERWORLD_Y and w.bridge_len(here=True) >= 20,
            "home": list(w.home),
            "milestones": sorted(w.milestones),
        }

    # -------------------------------------- Family 3: Gathering & Navigation
    def _tool_navigate_to(self, x: int, y: int, budget: int = 300) -> dict[str, Any]:
        w = self.world
        budget = min(300, max(10, int(budget)))
        res = w.walk(int(x), int(y), budget=budget)
        w.reveal(8)
        chip = w.harass()
        w.refresh()
        return {"ok": True, **res, "harass_damage": chip}

    def _tool_chop_trees(self, radius: int = 28, max_trees: int = 3) -> dict[str, Any]:
        w = self.world
        radius = min(80, max(4, int(radius)))
        max_trees = min(6, max(1, int(max_trees)))
        px = w.pos[0]
        candidates = sorted(
            (abs(tx - px), tx) for tx in w.trees if abs(tx - px) <= radius and w.g[w.surf[tx] - 1][tx] in (TREE, WITHERED)
        )
        if not candidates:
            return {"ok": False, "error": "no trees within radius", "wood_gained": 0}
        chopped, wood_gained, gel_gained = 0, 0, 0
        for _, tx in candidates[:max_trees]:
            ty = w.surf[tx] - 1
            w.walk(tx, ty, budget=120)
            hgt = w.trees[tx]
            is_withered = tx in w.withered
            for k in range(1, hgt + 1):
                if w.g[w.surf[tx] - k][tx] in (TREE, WITHERED):
                    w.set_tile(tx, w.surf[tx] - k, AIR)
            w.trees.pop(tx, None)
            w.withered.discard(tx)
            gain = hgt * (1 if is_withered else 4)
            wood_gained += w.give("wood", gain)
            if not is_withered:
                gel_gained += w.give("gel", 1)
            chopped += 1
            w.tick += 60
        chip = w.harass()
        w.refresh()
        return {"ok": True, "trees_chopped": chopped, "wood_gained": wood_gained, "gel_gained": gel_gained,
                "position": list(w.pos), "harass_damage": chip}

    def _tool_mine_vein(self, x: int | None = None, y: int | None = None, tile_type: str = "any", max_tiles: int = 8) -> dict[str, Any]:
        w = self.world
        max_tiles = min(12, max(1, int(max_tiles)))
        w.reveal(20)
        if tile_type in ORE_TILE:
            wanted = {ORE_TILE[tile_type]}
        elif tile_type == "stone":
            wanted = {STONE}
        else:
            max_pick = ITEMS[w.eq["pickaxe"]]["tier"]
            wanted = {t for t in TILE_ORE if max_pick >= ORE_PICK.get(t, 0)} | {STONE}

        if x is not None and y is not None:
            tx, ty = int(x), int(y)
        else:
            found = w.nearest_known(wanted & set(TILE_ORE.keys()), maxd=50) or w.nearest_known(wanted, maxd=50)
            if found is None:
                # Scan immediate vicinity for stone/ore
                px, py = w.pos
                best = None
                for cy in range(max(1, py - 18), min(H - 2, py + 19)):
                    for cx in range(max(1, px - 18), min(W - 2, px + 19)):
                        if w.g[cy][cx] in wanted and (cx, cy) not in w.placed:
                            d = abs(cx - px) + abs(cy - py)
                            if best is None or d < best[0]:
                                best = (d, cx, cy)
                if best is None:
                    return {"ok": False, "error": f"no reachable vein matching '{tile_type}'"}
                _, tx, ty = best
            else:
                _, tx, ty = found

        nav = w.walk(tx, ty, budget=240)
        px, py = w.pos
        mined = 0
        before_inv = dict(w.inv)
        for cy in range(max(1, py - 4), min(H - 2, py + 5)):
            for cx in range(max(1, px - 4), min(W - 2, px + 5)):
                if mined >= max_tiles:
                    break
                t = w.g[cy][cx]
                if t in wanted:
                    if t in ORE_PICK and ITEMS[w.eq["pickaxe"]]["tier"] < ORE_PICK[t]:
                        continue
                    w.dig_yield(cx, cy)
                    w.set_tile(cx, cy, AIR)
                    mined += 1
                    w.tick += w.pick_ticks
        w.reveal(10)
        chip = w.harass()
        w.refresh()
        gained = {k: w.inv[k] - before_inv.get(k, 0) for k in w.inv if w.inv[k] > before_inv.get(k, 0)}
        return {"ok": True, "mined_tiles": mined, "gained": gained, "position": list(w.pos), "nav": nav, "harass_damage": chip}

    def _tool_explore_cavern(self, target_depth: int = 58, auto_torch: bool = True) -> dict[str, Any]:
        w = self.world
        target_depth = min(H - 4, max(30, int(target_depth)))
        # Pick a cave air pocket near target_depth closest in x to player
        candidates = [(abs(cy - target_depth) * 2 + abs(cx - w.pos[0]), cx, cy) for cx, cy in w.cave if cy >= min(target_depth, w.pos[1] + 6)]
        if candidates:
            _, gx, gy = min(candidates)
        else:
            gx, gy = w.pos[0], target_depth
        nav = w.walk(gx, gy, budget=300)
        revealed = w.reveal(16)
        torch_placed = False
        if auto_torch and w.light(*w.pos) < 0.25 and w.has("torch") and w.g[w.pos[1]][w.pos[0]] == AIR:
            w.take("torch", 1)
            w.set_tile(*w.pos, TORCH, placed=True)
            torch_placed = True
        if w.pos[1] >= 36:
            w.give("mushroom", 1)
        chip = w.harass()
        w.refresh()
        return {"ok": True, "position": list(w.pos), "zone": w.zone(), "max_depth": w.max_depth,
                "newly_revealed": len(revealed), "torch_placed": torch_placed, "nav": nav, "harass_damage": chip}

    def _tool_loot_container_or_crystal(self, x: int | None = None, y: int | None = None, kind: str = "any") -> dict[str, Any]:
        w = self.world
        w.reveal(18)
        wanted = {CRYSTAL} if kind == "life_crystal" else {CHEST} if kind == "chest" else {CHEST, CRYSTAL}
        if x is not None and y is not None:
            tx, ty = int(x), int(y)
        else:
            found = w.nearest_known(wanted, maxd=60)
            if found is None:
                return {"ok": False, "error": f"no discovered {kind} within range"}
            _, tx, ty = found

        if (tx, ty) in w.opened or w.g[ty][tx] not in wanted:
            return {"ok": False, "error": "container or crystal already looted or missing"}

        w.walk(tx, ty, budget=240)
        if abs(w.pos[0] - tx) + abs(w.pos[1] - ty) > 4:
            return {"ok": False, "error": "could not reach target in one operation", "position": list(w.pos)}

        t = w.g[ty][tx]
        w.opened.add((tx, ty))
        w.set_tile(tx, ty, AIR)
        loot: dict[str, int] = {}
        if t == CRYSTAL:
            w.crystals += 1
            w.max_hp += 20
            w.hp = min(w.max_hp, w.hp + 20)
            if w.crystals <= 3:
                w.milestones.add(f"CRYSTAL{w.crystals}")
            kind_looted = "life_crystal"
        else:
            r = stream(w.seed, "chest", tx, ty)
            loot["lesser_healing_potion"] = w.give("lesser_healing_potion", 2)
            loot["torch"] = w.give("torch", 6)
            loot["wooden_arrow"] = w.give("wooden_arrow", 40)
            bar = r.choice(("iron_bar", "silver_bar"))
            loot[bar] = w.give(bar, 2)
            if r.random() < 0.5:
                loot["ironskin_potion"] = w.give("ironskin_potion", 1)
            kind_looted = "chest"
        chip = w.harass()
        w.refresh()
        return {"ok": True, "looted": kind_looted, "target": [tx, ty], "loot": loot, "crystals": w.crystals,
                "max_hp": w.max_hp, "harass_damage": chip}

    # ------------------------------------------- Family 4: Crafting & Building
    def _tool_craft_item(self, item_id: str, quantity: int = 1) -> dict[str, Any]:
        w = self.world
        quantity = max(1, int(quantity))
        if item_id not in RECIPES:
            return {"ok": False, "error": f"unknown recipe '{item_id}'"}
        out_n, reqs, station = RECIPES[item_id]
        if station is not None and not w.near_tile(PLACEABLE[station], radius=4):
            return {"ok": False, "error": f"must be within 4 tiles of placed '{station}'"}
        for ing, need in reqs.items():
            if not w.has(ing, need * quantity):
                return {"ok": False, "error": f"insufficient '{ing}': need {need * quantity}, have {w.inv.get(ing, 0)}"}
        for ing, need in reqs.items():
            w.take(ing, need * quantity)
        gained = w.give(item_id, out_n * quantity)
        w.milestones.add(f"CRAFT:{item_id}")
        if ITEMS[item_id]["kind"] == "pickaxe" and ITEMS[item_id]["tier"] > ITEMS[w.eq["pickaxe"]]["tier"]:
            w.eq["pickaxe"] = item_id
        w.tick += 30
        w.refresh()
        return {"ok": True, "item_id": item_id, "crafted": gained, "station": station}

    def _tool_place_station_or_furniture(self, item_id: str, dx: int = 0, dy: int = 0) -> dict[str, Any]:
        w = self.world
        dx, dy = min(6, max(-6, int(dx))), min(6, max(-6, int(dy)))
        tx, ty = min(W - 2, max(1, w.pos[0] + dx)), min(H - 3, max(1, w.pos[1] + dy))
        tile_map = {**PLACEABLE, "wood_platform": PLATFORM, "wood": WOOD, "wood_block": WOOD, "stone": STONE}
        inv_key = "wood" if item_id == "wood_block" else item_id
        if item_id not in tile_map or not w.has(inv_key, 1):
            return {"ok": False, "error": f"cannot place '{item_id}' (not in inventory or not placeable)"}
        w.take(inv_key, 1)
        w.set_tile(tx, ty, tile_map[item_id], placed=True)
        if item_id in ("workbench", "furnace", "anvil"):
            w.milestones.add(item_id.upper())
        w.tick += 20
        w.refresh()
        return {"ok": True, "placed": item_id, "at": [tx, ty]}

    def _tool_build_blueprint(self, preset: str = "shelter", matrix: list[list[int]] | None = None,
                              origin_dx: int = -4, origin_dy: int = -4) -> dict[str, Any]:
        w = self.world
        px, py = w.pos
        if matrix is not None:
            # Explicit 2D tile matrix placement (used by blueprint synthesis or partial-enclosure tests)
            placed_count = 0
            for r_idx, row in enumerate(matrix):
                for c_idx, t in enumerate(row):
                    tx, ty = px + int(origin_dx) + c_idx, py + int(origin_dy) + r_idx
                    if not (1 <= tx < W - 1 and 1 <= ty < H - 2) or t < 0:
                        continue
                    if t == WOOD:
                        if not w.has("wood", 1):
                            continue
                        w.take("wood", 1)
                    elif t == PLATFORM:
                        if not w.has("wood_platform", 1):
                            continue
                        w.take("wood_platform", 1)
                    w.set_tile(tx, ty, int(t), placed=(t != AIR))
                    placed_count += 1
            w.tick += 120
            w.refresh()
            return {"ok": True, "mode": "matrix", "tiles_set": placed_count, "housing": w.housing(), "arena": w.arena()}

        if preset == "shelter":
            # Need at least 20 wood (or existing workbench + wood) to build a 9x6 enclosed cabin (7x4=28 interior tiles)
            if w.inv.get("wood", 0) < 16 and not w.has("workbench"):
                return {"ok": False, "error": "need at least 16 wood to build a valid shelter"}
            if not w.has("workbench") and w.has("wood", 10):
                w.take("wood", 10)
                w.give("workbench", 1)
                w.milestones.add("CRAFT:workbench")
            if not w.has("torch") and not w.has("campfire") and w.has("wood", 1) and w.has("gel", 1):
                w.take("wood", 1)
                w.take("gel", 1)
                w.give("torch", 3)
                w.milestones.add("CRAFT:torch")
            w.take("wood", min(w.inv.get("wood", 0), 16))
            x0 = min(W - 12, max(3, px - 4))
            y0 = min(H - 10, max(4, py - 4))
            for y in range(y0, y0 + 6):
                for x in range(x0, x0 + 9):
                    border = x in (x0, x0 + 8) or y in (y0, y0 + 5)
                    if border:
                        t = DOOR if (x == x0 and y == y0 + 4) else WOOD
                        w.set_tile(x, y, t, placed=True)
                    else:
                        w.set_tile(x, y, AIR, placed=False)
            w.pos = (x0 + 4, y0 + 4)
            if w.has("workbench"):
                w.take("workbench", 1)
                w.set_tile(x0 + 2, y0 + 4, WORKBENCH, placed=True)
                w.milestones.add("WORKBENCH")
            elif w.g[y0 + 4][x0 + 2] != WORKBENCH:
                w.set_tile(x0 + 2, y0 + 4, WORKBENCH, placed=True)
                w.milestones.add("WORKBENCH")
            if w.has("campfire"):
                w.take("campfire", 1)
                w.set_tile(x0 + 6, y0 + 4, CAMPFIRE, placed=True)
            elif w.has("torch"):
                w.take("torch", 1)
                w.set_tile(x0 + 4, y0 + 2, TORCH, placed=True)
            else:
                w.set_tile(x0 + 4, y0 + 2, TORCH, placed=True)
            w.tick += 180
            w.refresh()
            return {"ok": True, "preset": "shelter", "housing": w.housing(), "position": list(w.pos)}

        if preset == "arena":
            # Build 3 platform rows x 26 wide above surface near player + campfire
            need_plat = 78
            while w.inv.get("wood_platform", 0) < need_plat and w.has("wood", 1):
                w.take("wood", 1)
                w.give("wood_platform", 2)
                w.milestones.add("CRAFT:wood_platform")
            if w.inv.get("wood_platform", 0) < 36:
                return {"ok": False, "error": "insufficient wood_platform or wood for arena (need >= 36 platforms)"}
            if not w.has("campfire") and w.has("wood", 10) and w.has("torch", 1):
                w.take("wood", 10)
                w.take("torch", 1)
                w.give("campfire", 1)
                w.milestones.add("CRAFT:campfire")
            cx = min(W - 20, max(20, px))
            base_y = max(8, w.surf[cx] - 3)
            ys = (base_y - 6, base_y - 3, base_y)
            for ry in ys:
                for rx in range(cx - 12, cx + 13):
                    if w.has("wood_platform", 1):
                        w.take("wood_platform", 1)
                    w.set_tile(rx, ry, PLATFORM, placed=True)
            if w.has("campfire"):
                w.take("campfire", 1)
                w.set_tile(cx, ys[1] - 1, CAMPFIRE, placed=True)
            elif w.has("torch"):
                w.take("torch", 1)
                w.set_tile(cx, ys[1] - 1, CAMPFIRE, placed=True)
            else:
                w.set_tile(cx, ys[1] - 1, CAMPFIRE, placed=True)
            w.pos = (cx, ys[1])
            w.tick += 200
            w.refresh()
            return {"ok": True, "preset": "arena", "arena": w.arena(), "on_arena": w.on_arena(), "position": list(w.pos)}

        if preset == "hellbridge":
            if py < UNDERWORLD_Y:
                return {"ok": False, "error": "must be in the underworld (y >= 88) to build a hellbridge"}
            while w.inv.get("wood_platform", 0) < 64 and w.has("wood", 1):
                w.take("wood", 1)
                w.give("wood_platform", 2)
            if w.inv.get("wood_platform", 0) < 60:
                return {"ok": False, "error": "need at least 60 wood_platform (or 30 wood) for hellbridge"}
            by = min(95, max(UNDERWORLD_Y, py))
            x0 = min(W - 68, max(4, px - 30))
            for rx in range(x0, x0 + 64):
                if w.has("wood_platform", 1):
                    w.take("wood_platform", 1)
                w.set_tile(rx, by, PLATFORM, placed=True)
            w.pos = (x0 + 32, by)
            w.tick += 220
            w.refresh()
            return {"ok": True, "preset": "hellbridge", "bridge_len": w.bridge_len(here=True), "position": list(w.pos)}

        return {"ok": False, "error": f"unknown blueprint preset '{preset}'"}

    def _tool_equip_item(self, item_id: str) -> dict[str, Any]:
        w = self.world
        if not w.has(item_id) or item_id not in ITEMS:
            return {"ok": False, "error": f"item '{item_id}' not in inventory"}
        kind = ITEMS[item_id]["kind"]
        if kind not in ("weapon", "armor", "pickaxe"):
            return {"ok": False, "error": f"item '{item_id}' ({kind}) cannot be equipped"}
        w.eq[kind] = item_id
        w.refresh()
        return {"ok": True, "equipped": dict(w.eq), "tech_tier": w.tech_tier, "defense": w.defense}

    def _tool_consume_buff_or_heal(self, item_id: str = "lesser_healing_potion") -> dict[str, Any]:
        w = self.world
        if not w.has(item_id):
            return {"ok": False, "error": f"'{item_id}' not in inventory"}
        if item_id == "lesser_healing_potion":
            if w.abs_tick < w.sick_until:
                return {"ok": False, "error": "Potion Sickness active", "sick_ticks_remaining": w.sick_until - w.abs_tick}
            w.take(item_id, 1)
            before = w.hp
            w.hp = min(w.max_hp, w.hp + 70)
            w.sick_until = w.abs_tick + 3600
            return {"ok": True, "item_id": item_id, "healed": w.hp - before, "hp": w.hp, "sick_ticks": 3600}
        if item_id == "ironskin_potion":
            w.take(item_id, 1)
            w.buffs["ironskin"] = w.abs_tick + 18000
            return {"ok": True, "item_id": item_id, "defense": w.defense, "buff_ticks": 18000}
        return {"ok": False, "error": f"'{item_id}' is not a consumable potion"}

    def _tool_manage_inventory_trash(self, item_id: str, quantity: int = 1) -> dict[str, Any]:
        w = self.world
        quantity = max(1, int(quantity))
        if item_id == "copper_pickaxe":
            return {"ok": False, "error": "starter copper_pickaxe is soulbound and cannot be trashed"}
        if not w.has(item_id, 1):
            return {"ok": False, "error": f"'{item_id}' not in inventory"}
        removed = min(quantity, w.inv.get(item_id, 0))
        w.take(item_id, removed)
        w.refresh()
        return {"ok": True, "trashed": item_id, "quantity": removed, "slots_used": w.slots()}

    # -------------------------------------------------------- Family 5: Combat
    def _tool_summon_or_trigger_boss(self, boss: str = "eye") -> dict[str, Any]:
        w = self.world
        if w.boss is not None:
            return {"ok": False, "error": f"boss '{w.boss['kind']}' is already active"}
        if boss in ("eye", "suspicious_looking_eye"):
            if not w.has("suspicious_looking_eye"):
                return {"ok": False, "error": "need 'suspicious_looking_eye' in inventory"}
            w.take("suspicious_looking_eye", 1)
            w.spawn_boss("eye")
            return {"ok": True, "boss": "eye", "hp": w.boss["hp"]}
        if boss in ("wall_of_flesh", "guide_voodoo_doll"):
            if w.pos[1] < UNDERWORLD_Y:
                return {"ok": False, "error": "must be in the underworld (y >= 88) to summon Wall of Flesh"}
            if not w.has("guide_voodoo_doll"):
                return {"ok": False, "error": "need 'guide_voodoo_doll' in inventory"}
            w.take("guide_voodoo_doll", 1)
            w.spawn_boss("wall_of_flesh")
            return {"ok": True, "boss": "wall_of_flesh", "hp": w.boss["hp"]}
        return {"ok": False, "error": f"unknown boss '{boss}'"}

    def _tool_engage_combat_burst(self, target_id: str | None = None, weapon: str | None = None,
                                  kite_distance: float = 12.0, dodge_priority: float = 0.5,
                                  vertical_platform_use: bool = True) -> dict[str, Any]:
        w = self.world
        res = w.burst(target_id=target_id, weapon=weapon, kite=kite_distance, dodge=dodge_priority, vertical=vertical_platform_use)
        w.refresh()
        return res

    def _tool_retreat_and_stabilize(self, use_potion: bool = True) -> dict[str, Any]:
        w = self.world
        w.stance = "funnel"
        if w.boss is None:
            w.walk(*w.home, budget=240)
        healed = 0
        if use_potion and w.hp < w.max_hp * 0.65 and w.has("lesser_healing_potion") and w.abs_tick >= w.sick_until:
            w.take("lesser_healing_potion", 1)
            before = w.hp
            w.hp = min(w.max_hp, w.hp + 70)
            w.sick_until = w.abs_tick + 3600
            healed = w.hp - before
        if w.enclosure(*w.pos) is not None:
            w.hp = min(w.max_hp, w.hp + 10)
        w.tick += 90
        w.refresh()
        return {"ok": True, "position": list(w.pos), "stance": w.stance, "healed": healed, "hp": w.hp,
                "sheltered": w.housing()["valid"]}

    def _tool_deploy_combat_arena_tactics(self, tactic: str = "grapple_tier") -> dict[str, Any]:
        w = self.world
        w.stance = "funnel"
        a = w.arena()
        if a["valid"]:
            cx = (a["x_range"][0] + a["x_range"][1]) // 2
            cy = a["y_range"][0]
            w.pos = (cx, cy)
        elif w.has("campfire"):
            w.take("campfire", 1)
            w.set_tile(w.pos[0], w.pos[1], CAMPFIRE, placed=True)
        w.tick += 60
        w.refresh()
        return {"ok": True, "tactic": tactic, "stance": w.stance, "on_arena": w.on_arena(), "position": list(w.pos)}

    def _tool_query_threat_telemetry(self) -> dict[str, Any]:
        w = self.world
        threat, alerts = w.alerts()
        boss_info = None
        if w.boss:
            spec = BOSSES[w.boss["kind"]]
            boss_info = {
                "kind": w.boss["kind"],
                "title": spec["title"],
                "hp": round(w.boss["hp"], 1),
                "max_hp": spec["hp"],
                "phase_2": w.boss["kind"] == "eye" and w.boss["hp"] <= 0.5 * spec["hp"],
                "advance": w.boss["advance"],
                "bridge_remaining": max(0, w.bridge_len(here=True) - w.boss["advance"]) if w.boss["kind"] == "wall_of_flesh" else None,
            }
        return {
            "ok": True,
            "threat": threat,
            "threat_multiplier": w.threat_multiplier(),
            "boss": boss_info,
            "boss_min_ratio": dict(w.boss_min),
            "targets": w.targets(),
            "alerts": alerts,
        }

    def _tool_record_tactical_checkpoint(self, hypothesis: str, plan: str = "", confidence: float = 0.8) -> dict[str, Any]:
        w = self.world
        entry = {"w": w.window, "tick": w.tick, "hypothesis": str(hypothesis), "plan": str(plan), "confidence": float(confidence)}
        w.notes.append(entry)
        return {"ok": True, "recorded": entry, "total_checkpoints": len(w.notes)}
