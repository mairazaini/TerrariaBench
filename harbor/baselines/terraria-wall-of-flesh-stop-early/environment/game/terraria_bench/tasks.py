"""Task lenses over the shared seeded world (StoreBench §3.3 / Appendix A).

Every lens is a small configuration: horizon (windows), operation budget, event
script, starting kit and the reliability checkpoints (SLA), which the agent is told. The world
itself is keyed only by the seed, so all lenses share one world per seed.

Event channels (StoreBench §3.2):
  announced - explicit alert with lead time (``announce`` = first window shown)
  rumored   - vague text, no numbers (eye_spawn, cave_chill)
  silent    - never surfaced as an alert; visible only in telemetry
              (spawn_spike, corruption spread, traps, boss phase 2)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

REPORTED_SEEDS = (7, 11, 13)
HELD_OUT_SEEDS = (17, 23)

RULES = (
    "You are the player in a Terraria world. You act only through the provided tools. "
    "Time is metered in operations, not wall-clock: the world is frozen while you think. "
    "Each 12-hour window (Day, then Night, alternating) gives you a fixed budget of operations; "
    "every tool call costs one operation except terraria_status, which is free. "
    "Spending the last operation or calling end_window lets the rest of the window play out "
    "with you standing wherever you are, so be somewhere safe first. Unspent operations do not "
    "roll over. Hostiles cannot reach you inside a fully enclosed space; bosses can. "
    "Dying drops half of your raw materials and costs you operations in the next window."
)


HARDCORE_RULE = (
    "HARDCORE: you have exactly one life. The first time you die, the run is over at that moment: you make no "
    "more calls, every later window counts as not attended, and every checkpoint from then on counts as missed. "
    "You keep credit for what you achieved before dying. You die when your HP reaches 0. The usual ways to die "
    "are: a window ends while you are not inside valid housing; a window ends while a boss you summoned is still "
    "alive; or you keep fighting or exploring at low HP. inspect_housing_and_arena tells you whether the place "
    "you are standing in is valid housing right now. Building or digging next to a house can break it."
)

# The scoring rule, told to every agent in full. It restates grader.grade; nothing scored is left out.
SCORING = (
    "How you are scored (one number from 0 to 1, computed from the server's record of the whole run):\n"
    "- 70% progress. Your progress is compared with two fixed runs on this same world: a player who does nothing "
    "(0) and an expert run (1). Progress is permanent achievements: gear tier (weapon and armor both Wood, then "
    "Iron, Silver, Gold: each tier is worth more than the last), valid housing, crafting stations, a boss arena, "
    "reaching the cavern layer and the Underworld, Life Crystals used, events beaten, damage dealt to bosses and "
    "boss kills (worth the most). Items you hold add a little. Each death subtracts.\n"
    "- 20% reliability. Half is, for each window, being alive at its end and having made progress recently; the "
    "other half is the checkpoints listed below.\n"
    "- 10% attendance: taking part in every window. It only counts once you have made real progress.\n"
    "The checkpoints are minimums, not the goal. A run that meets every checkpoint and then stops scores far "
    "below an expert run, because unused operations are simply lost. Repeating an action that no longer adds "
    "anything (re-crafting, placing and breaking, farming weak enemies) earns nothing or almost nothing."
)


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    split: str  # "eval" | "train"
    windows: int
    budget: int
    brief: str
    objectives: tuple = ()
    events: tuple = ()
    sla: tuple = ()  # (deadline_window, kind) - stated to the agent in the prompt, see checkpoint_text
    kit: dict = field(default_factory=dict)
    equip: tuple = ()
    start_max_hp: int = 100
    start_milestones: tuple = ()
    death_fee: float = 25.0
    corruption_rate: int = 2
    hardcore: bool = False  # one life: the first death ends the run

    @property
    def prompt(self) -> str:
        text = f"{RULES}\n\nYour situation: {self.brief}\nYou have {self.windows} windows and {self.budget} operations per window."
        if self.hardcore:
            text = text.replace(" Dying drops half of your raw materials and costs you operations in the next window.",
                                "") + "\n\n" + HARDCORE_RULE
        if self.sla:
            text += ("\n\n" + SCORING + "\n\nCheckpoints. Each one is checked once, at the moment the stated window "
                     "ends.\n" + self.checkpoint_text)
        return text

    @property
    def checkpoint_text(self) -> str:
        """Every deadline the grader checks, in plain words. The agent is scored on nothing it was not told."""
        return "\n".join(f"- End of window {w}: {describe_checkpoint(kind)}" for w, kind in self.sla)


_TIERS = {1: "Wood", 2: "Iron", 3: "Silver", 4: "Gold"}
_MILESTONES = {
    "SIEGE": "the Goblin Army is completely defeated",
    "BLOODMOON": "you have killed 8 Blood Zombies during the Blood Moon",
    "SLIME_RAIN": "you have killed 10 slimes during the Slime Rain",
    "HELLBRIDGE": "a platform bridge at least 60 tiles long exists (the Underworld runway)",
    "FURNACE": "a Furnace has been placed",
}


def describe_checkpoint(kind: str) -> str:
    """Plain-language form of one checkpoint; mirrors grader.sla_met exactly."""
    head, _, rest = kind.partition(":")
    if head == "shelter":
        return "you are inside valid housing when the window ends"
    if head == "arena":
        return "a valid boss arena has been built"
    if head == "tier":
        n = int(rest)
        return f"your equipped weapon AND equipped armor are both {_TIERS[n]} tier or better (tech tier {n})"
    if head == "depth":
        return f"you have been down to depth {rest} or deeper at some point"
    if head == "crystal":
        return f"you have used at least {rest} Life Crystal(s)"
    if head == "def":
        return f"the armor you are wearing gives at least {rest} defense (potions do not count)"
    if head == "milestone":
        return _MILESTONES[rest]
    if head == "boss":
        boss, _, thr = rest.rpartition(":")
        name = {"eye": "the Eye of Cthulhu", "wall_of_flesh": "the Wall of Flesh"}.get(boss, boss)
        return f"{name} has been brought down to {round(float(thr) * 100)}% of its health or lower"
    if head == "kill":
        name = {"eye": "the Eye of Cthulhu", "wall_of_flesh": "the Wall of Flesh"}.get(rest, rest)
        return f"{name} has been killed"
    raise ValueError(f"no description for checkpoint {kind!r}")


def _ev(kind, w, **kw):
    return {"kind": kind, "w": w, **kw}


_GOLD_KIT = {
    "gold_broadsword": 1, "gold_bow": 1, "gold_armor": 1, "gold_pickaxe": 1,
    "wooden_arrow": 400, "lesser_healing_potion": 5, "ironskin_potion": 2,
    "guide_voodoo_doll": 1, "wood": 220, "wood_platform": 160, "torch": 20,
    "campfire": 1, "gel": 6,
}

TASKS: dict[str, Task] = {t.id: t for t in [
    # ------------------------------------------------------------- evaluation
    Task("first-night", "Survive the first night", "eval", 4, 20,
         "You have just arrived with starter copper tools. Night is dangerous for the unsheltered. "
         "Build valid housing before dusk and get your first real gear.",
         objectives=("shelter", "gear"),
         events=(_ev("spawn_spike", 2),),
         sla=((2, "shelter"), (3, "tier:1"), (4, "shelter"))),
    Task("cavern-expedition", "Go deep and come back", "eval", 10, 16,
         "Establish a base, then push underground: reach the cavern layer, forge iron-tier gear "
         "and find a Life Crystal. The depths do not announce their dangers.",
         objectives=("shelter", "gear", "depth", "crystal"),
         events=(_ev("cave_chill", 6, dur=2),),
         sla=((2, "shelter"), (4, "depth:50"), (8, "tier:2"), (10, "crystal:1"))),
    Task("goblin-siege", "Hold the line", "eval", 12, 20,
         "Scouts say goblins have been seen in the region. You start with a bow and an iron pickaxe. "
         "Fortify, arm yourself, and defeat the army if it comes.",
         objectives=("shelter", "gear", "siege"),
         events=(_ev("spawn_spike", 4), _ev("goblin_siege", 9, announce=6)),
         sla=((2, "shelter"), (8, "def:9"), (10, "milestone:SIEGE")),
         kit={"wooden_bow": 1, "wooden_arrow": 150, "iron_pickaxe": 1, "wood": 60, "torch": 8},
         equip=("wooden_bow",), start_milestones=("T1", "T2")),
    Task("blood-moon-survival", "The red night", "eval", 8, 16,
         "Something is wrong with the moon. Get sheltered and armed, and when the worst night comes, "
         "stand your ground rather than only hiding.",
         objectives=("shelter", "gear", "blood_moon"),
         events=(_ev("spawn_spike", 2), _ev("blood_moon", 6, announce=5)),
         sla=((2, "shelter"), (4, "tier:1"), (6, "milestone:BLOODMOON"), (8, "shelter"))),
    Task("arena-architect", "Build the arena", "eval", 8, 12,
         "A powerful creature will find you within a few nights and you cannot choose when. "
         "Operations are scarce: build a multi-tier boss arena with a campfire and be on it, armed.",
         objectives=("shelter", "arena", "boss:eye"),
         events=(_ev("eye_spawn", 8),),
         sla=((2, "shelter"), (6, "arena"), (8, "boss:eye:0.85")),
         kit={"wooden_bow": 1, "wooden_arrow": 250, "wood": 40},
         equip=("wooden_bow",), start_milestones=("T1",)),
    Task("eye-of-cthulhu", "Defeat the Eye", "eval", 14, 20,
         "You carry a Suspicious Looking Eye. Use it at night when you are ready - or the Eye of "
         "Cthulhu will eventually come on its own terms. Prepare gear and an arena, then kill it.",
         objectives=("shelter", "gear", "arena", "boss:eye"),
         events=(_ev("spawn_spike", 4), _ev("eye_spawn", 12)),
         sla=((2, "shelter"), (8, "tier:2"), (10, "arena"), (12, "boss:eye:0.6"), (14, "kill:eye")),
         kit={"suspicious_looking_eye": 1}),
    Task("wall-of-flesh", "Descend and challenge the Wall", "eval", 12, 20,
         "You are fully geared in gold and carry a Guide Voodoo Doll. Descend to the underworld, "
         "build a long bridge to fight on, and do as much damage to the Wall of Flesh as you can.",
         objectives=("shelter", "depth", "boss:wall_of_flesh"),
         events=(_ev("cave_chill", 4, dur=2),),
         sla=((2, "shelter"), (5, "depth:88"), (8, "milestone:HELLBRIDGE"),
              (10, "boss:wall_of_flesh:0.8"), (12, "boss:wall_of_flesh:0.65")),
         kit=_GOLD_KIT, equip=("gold_bow", "gold_armor"), start_max_hp=300,
         start_milestones=("T1", "T2", "T3", "T4")),
    Task("full-progression", "The marathon", "eval", 30, 12,
         "A long stay with few operations per window. Progress from nothing through the ore tiers, "
         "weather whatever the world sends, and take on the Eye of Cthulhu. You carry a Suspicious "
         "Looking Eye.",
         objectives=("shelter", "gear", "arena", "siege", "blood_moon", "boss:eye"),
         events=(_ev("spawn_spike", 4), _ev("blood_moon", 10, announce=9), _ev("cave_chill", 14, dur=2),
                 _ev("goblin_siege", 19, announce=16), _ev("eye_spawn", 26)),
         sla=((2, "shelter"), (8, "tier:1"), (14, "tier:2"), (20, "milestone:SIEGE"),
              (24, "arena"), (26, "boss:eye:0.6"), (30, "tier:3")),
         kit={"suspicious_looking_eye": 1}),
    # --------------------------------------------------------------- training
    Task("woodland-scramble", "The forest is dying", "train", 4, 12,
         "Wood is the first bottleneck, and the nearby forest is not as healthy as it looks. "
         "Secure wood, shelter and wooden gear with few operations.",
         objectives=("shelter", "gear"),
         sla=((2, "shelter"), (4, "tier:1")), corruption_rate=9),
    Task("ore-rush", "From ore to anvil", "train", 6, 16,
         "You start with a workbench and timber. Get a furnace and an anvil running and forge "
         "iron-tier gear as fast as you can.",
         objectives=("shelter", "gear"),
         sla=((2, "shelter"), (4, "milestone:FURNACE"), (6, "tier:2")),
         kit={"wood": 90, "torch": 6, "workbench": 1}),
    Task("slime-rain", "It is raining slime", "train", 4, 16,
         "The sky looks strange. Shelter first, then make the most of what falls.",
         objectives=("shelter", "gear", "slime_rain"),
         events=(_ev("slime_rain", 3, announce=1),),
         sla=((2, "shelter"), (3, "milestone:SLIME_RAIN"))),
    Task("shelter-crunch", "Eleven moves to dusk", "train", 4, 11,
         "Very few operations per window and a hard first night. Every move must count.",
         objectives=("shelter", "gear"),
         events=(_ev("spawn_spike", 2),),
         sla=((2, "shelter"), (4, "shelter"))),
]}

# Hardcore variants of the two headline tasks: same world, same brief, same checkpoints, one life.
for _base in ("first-night", "eye-of-cthulhu"):
    _t = TASKS[_base]
    TASKS[f"{_base}-hardcore"] = replace(_t, id=f"{_base}-hardcore", title=f"{_t.title} (Hardcore)",
                                         split="hardcore", hardcore=True)

HARDCORE_TASKS = tuple(t for t in TASKS if TASKS[t].split == "hardcore")
EVAL_TASKS = tuple(t for t in TASKS if TASKS[t].split == "eval")
TRAIN_TASKS = tuple(t for t in TASKS if TASKS[t].split == "train")
assert len(EVAL_TASKS) == 8 and len(TRAIN_TASKS) == 4
assert all(11 <= t.budget <= 20 for t in TASKS.values())
