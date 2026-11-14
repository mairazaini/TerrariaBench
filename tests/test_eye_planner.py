"""Long-term plan (Strategy + roadmap) and short-term candidates of the Eye planner.

Real Terraria is not available in CI, so the end-to-end tests drive the planner against a small
rule model of the game (``FakeTerraria``): each skill changes the HUD the way the mod reports it.
It checks the thing that matters in theory: under every strategy the plan advances step by step
and reaches the fight without stalling or looping.
"""
from __future__ import annotations

import pytest

from terraria_bench.eye_planner import REFERENCE_STRATEGY, SUMMON, EyePlanner, Strategy

RECIPES = {  # item -> (station, {ingredient: qty}, output qty)
    "Work Bench": ("", {"Wood": 10}, 1),
    "Wooden Bow": ("Work Bench", {"Wood": 10}, 1),
    "Wooden Sword": ("Work Bench", {"Wood": 7}, 1),
    "Wood Helmet": ("Work Bench", {"Wood": 20}, 1),
    "Wood Breastplate": ("Work Bench", {"Wood": 30}, 1),
    "Wood Greaves": ("Work Bench", {"Wood": 25}, 1),
    "Wooden Arrow": ("Work Bench", {"Wood": 1, "Stone Block": 1}, 25),
    "Torch": ("", {"Wood": 1, "Gel": 1}, 3),
    "Furnace": ("Work Bench", {"Stone Block": 20, "Wood": 4, "Torch": 3}, 1),
    "Iron Bar": ("Furnace", {"Iron Ore": 3}, 1),
    "Iron Anvil": ("Work Bench", {"Iron Bar": 5}, 1),
    "Iron Bow": ("Anvil", {"Iron Bar": 7}, 1),
    "Iron Pickaxe": ("Anvil", {"Iron Bar": 12}, 1),
    "Iron Helmet": ("Anvil", {"Iron Bar": 15}, 1),
    "Iron Chainmail": ("Anvil", {"Iron Bar": 25}, 1),
    "Iron Greaves": ("Anvil", {"Iron Bar": 20}, 1),
    SUMMON: ("Demon Altar", {"Lens": 6}, 1),
}
ALTAR = [150, 0]


class FakeTerraria:
    """Just enough game rules to answer every skill the planner can ask for."""

    def __init__(self) -> None:
        self.bag: dict[str, int] = {"Copper Pickaxe": 1, "Copper Axe": 1, "Copper Shortsword": 1}
        self.worn: list[str] = []
        self.pos = [0, 0]
        self.clock = 8.0
        self.hp = self.max_hp = 100
        self.stations: dict[str, list[int]] = {}
        self.boss = None
        self.crystals_left = 3
        self.calls: list[str] = []

    # ---- what the mod would report
    def night(self) -> bool:
        return self.clock >= 19.5 or self.clock < 4.5

    def near(self, kind: str) -> bool:
        at = ALTAR if kind == "Demon Altar" else self.stations.get(kind)
        return bool(at) and max(abs(at[0] - self.pos[0]), abs(at[1] - self.pos[1])) <= 30

    def craftable(self) -> list[dict]:
        out = []
        for item, (station, cost, _) in RECIPES.items():
            if (not station or self.near(station)) and all(self.bag.get(k, 0) >= v for k, v in cost.items()):
                out.append({"name": item})
        return out

    def hud(self) -> dict:
        underground = self.pos[1] > 25
        wv = {
            "stations": [{"kind": k, "dx": v[0] - self.pos[0], "dy": v[1] - self.pos[1]}
                         for k, v in self.stations.items() if self.near(k)],
            "altars": [{"dx": ALTAR[0] - self.pos[0], "dy": ALTAR[1] - self.pos[1]}]
            if abs(self.pos[0] - ALTAR[0]) <= 40 and not underground else [],
            "ores": [{"dx": 3, "dy": 2}] if underground and self.pos[1] >= 40 else [],
            "life_crystals": [{"dx": 4, "dy": 1}] if underground and self.crystals_left and len(self.calls) % 3 == 0 else [],
            "chests": [],
        }
        return {
            "world": "fake", "pos": {"x": self.pos[0], "y": self.pos[1]}, "hp": self.hp, "max_hp": self.max_hp,
            "time_phase": "Night" if self.night() else "Day", "clock": self.clock, "dead": False,
            "backpack": {k: v for k, v in self.bag.items() if v > 0}, "equipment_now": list(self.worn),
            "craftable_now": self.craftable(), "world_view": wv, "boss": self.boss, "buffs": [],
            "evil_biome": abs(self.pos[0] - ALTAR[0]) <= 40,
        }

    # ---- what a skill would do
    def run(self, name: str, a: dict) -> tuple[str, str]:
        self.calls.append(name)
        self.clock = (self.clock + 0.75) % 24
        bag = self.bag
        if name == "chop":
            bag["Wood"] = max(bag.get("Wood", 0), int(a["wood"]))
        elif name == "craft":
            item, count = str(a["item"]), int(a.get("count", 1))
            if item not in {c["name"] for c in self.craftable()}:
                return "failed", f"no recipe for {item} here"
            _, cost, out_qty = RECIPES[item]
            batches = max(1, min(-(-count // out_qty), *(bag[k] // v for k, v in cost.items())))
            for k, v in cost.items():
                bag[k] -= v * batches
            bag[item] = bag.get(item, 0) + out_qty * batches
        elif name == "place":
            item = str(a["item"])
            bag[item] -= 1
            self.stations["Anvil" if "Anvil" in item else item] = list(self.pos)
        elif name == "equip":
            bag[a["item"]] -= 1
            self.worn = [w for w in self.worn if w.split()[-1] != str(a["item"]).split()[-1]] + [str(a["item"])]
        elif name == "mine":
            kind = str(a["kind"])
            if kind == "stone":
                bag["Stone Block"] = bag.get("Stone Block", 0) + int(a["count"])
            elif kind == "ore":
                if self.pos[1] < 40:
                    return "failed", "no ore in reach"
                bag["Iron Ore"] = bag.get("Iron Ore", 0) + min(int(a["count"]), 45)
            else:
                bag["Life Crystal"] = bag.get("Life Crystal", 0) + 1
                self.crystals_left -= 1
        elif name == "dig_down":
            self.pos[1] += int(a.get("depth", 20))
        elif name == "surface":
            self.pos[1] = 0
        elif name == "goto":
            self.pos = [int(a["x"]), int(a["y"])]
        elif name == "explore":
            self.pos[0] += int(a.get("tiles", 40)) * (1 if a.get("dir") == "right" else -1)
        elif name == "hunt":
            if a.get("npc") == "Demon Eye" and not self.night():
                return "failed", "no Demon Eye came"
            bag[str(a["item"])] = bag.get(str(a["item"]), 0) + 3
        elif name == "use":
            item = str(a["item"])
            bag[item] -= 1
            if item == "Life Crystal":
                self.max_hp += 20
            elif item == SUMMON:
                self.boss = {"name": "Eye of Cthulhu", "hp": 2800, "max_hp": 2800}
        elif name == "wait":
            self.clock = 20.0 if a.get("until") == "night" else 6.0
        elif name == "fight":
            self.boss = None
            return "success", "Eye of Cthulhu defeated"
        elif name not in ("build_arena", "bunker", "loot_chest"):
            raise AssertionError(f"planner asked for an unknown skill: {name}")
        return "success", "ok"


def _play(planner: EyePlanner, max_decisions: int = 500) -> FakeTerraria:
    game = FakeTerraria()
    for _ in range(max_decisions):
        hud = game.hud()
        planner.observe(hud)
        cands = planner.candidates(hud)
        if not cands:
            break
        status, reason = game.run(cands[0].name, cands[0].args)
        planner.record_result(cands[0].name, cands[0].args, status, reason, game.hud())
    return game


STRATEGIES = {
    "reference": REFERENCE_STRATEGY,
    "rush": Strategy(bow="wooden", armor="none", max_hp=100, night="work"),
    "fortress": Strategy(bow="metal", armor="metal", max_hp=160, pickaxe=True, arrows=600),
}


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_reaches_and_wins_the_fight(tmp_path, name):
    planner = EyePlanner(memory_path=tmp_path / "mem.json")
    planner.set_strategy(STRATEGIES[name], author="test")
    game = _play(planner)
    assert planner.mem["eye_defeated"], (name, game.calls[-25:], planner.mem["log"][-6:])
    assert {r["status"] for r in planner.roadmap(game.hud())} <= {"done", "skipped"}
    # the plan shaped the run: what was prepared matches what the strategy asked for
    st = STRATEGIES[name]
    assert (game.bag.get("Iron Bow", 0) > 0) == (st.bow == "metal")
    assert any(w.startswith("Iron ") for w in game.worn) == (st.armor == "metal")
    assert game.max_hp >= st.max_hp
    assert ("bunker" in game.calls) <= (st.night == "bunker")


def test_strategies_cost_different_amounts_of_preparation(tmp_path):
    lengths = {}
    for name, st in STRATEGIES.items():
        planner = EyePlanner(memory_path=tmp_path / f"{name}.json")
        planner.set_strategy(st, author="test")
        lengths[name] = len(_play(planner).calls)
    assert lengths["rush"] < lengths["reference"] < lengths["fortress"], lengths


def test_roadmap_has_one_current_step_and_advances(tmp_path):
    planner = EyePlanner(memory_path=tmp_path / "mem.json")
    game = FakeTerraria()
    first = planner.roadmap(game.hud())
    assert [r["status"] for r in first].count("current") == 1 and first[0]["status"] == "current"
    assert planner.candidates(game.hud())[0].name == "chop"
    game.bag.update({"Wood": 200, "Wooden Bow": 1})
    planner.mem["stations"]["Work Bench"] = [0, 0]
    later = planner.roadmap(game.hud())
    assert later[0]["status"] == later[1]["status"] == "done"
    assert [r["status"] for r in later].count("current") == 1
    assert "long_term_plan" in planner.briefing(game.hud())


def test_strategy_from_model_output_is_clamped():
    st = Strategy.from_dict({"bow": "Laser", "armor": "METAL", "max_hp": 9999, "arrows": -5, "night": "sleep",
                             "pickaxe": "false", "arena_width": 3, "delete_world": True, "rationale": "x" * 900})
    assert (st.bow, st.armor, st.max_hp, st.arrows, st.night, st.pickaxe, st.arena_width) == \
        ("metal", "metal", 400, 0, "bunker", False, 40)
    assert len(st.rationale) == 400 and "delete_world" not in st.as_dict()
    assert Strategy.from_dict("not json") == Strategy()


def test_unarmed_night_bunkers_then_waits_for_morning(tmp_path):
    planner = EyePlanner(memory_path=tmp_path / "mem.json")
    game = FakeTerraria()
    game.clock, game.bag["Wood"] = 21.0, 40
    assert planner.candidates(game.hud())[0].name == "bunker"
    planner.record_result("bunker", {}, "success", "closed", game.hud())
    assert planner.candidates(game.hud())[0].label().startswith("wait(until=day)")
    game.clock = 7.0
    planner.observe(game.hud())
    assert planner.mem["bunker"] is None and planner.candidates(game.hud())[0].name != "wait"
    # armed players use the night to hunt for Lens instead
    game.clock = 21.0
    game.bag.update({"Wooden Bow": 1, "Wooden Arrow": 900, "Wooden Sword": 1, "Wood": 300})
    game.worn = ["Wood Helmet", "Wood Breastplate", "Wood Greaves"]
    planner.mem["stations"]["Work Bench"] = [0, 0]
    names = [c.name for c in planner.candidates(game.hud())]
    assert "bunker" not in names and "hunt" in names


def test_low_health_heals_first(tmp_path):
    planner = EyePlanner(memory_path=tmp_path / "mem.json")
    game = FakeTerraria()
    game.hp, game.bag["Lesser Healing Potion"] = 30, 2
    first = planner.candidates(game.hud())[0]
    assert (first.name, first.args["item"]) == ("use", "Lesser Healing Potion")
