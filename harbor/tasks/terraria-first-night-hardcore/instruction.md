# TerrariaBench: Survive the first night (Hardcore)

You are the player in a Terraria world. You act only through the provided tools. Time is metered in operations, not wall-clock: the world is frozen while you think. Each 12-hour window (Day, then Night, alternating) gives you a fixed budget of operations; every tool call costs one operation except terraria_status, which is free. Spending the last operation or calling end_window lets the rest of the window play out with you standing wherever you are, so be somewhere safe first. Unspent operations do not roll over. Hostiles cannot reach you inside a fully enclosed space; bosses can.

Your situation: You have just arrived with starter copper tools. Night is dangerous for the unsheltered. Build valid housing before dusk and get your first real gear.
You have 4 windows and 20 operations per window.

HARDCORE: you have exactly one life. The first time you die, the run is over at that moment: you make no more calls, every later window counts as not attended, and every checkpoint from then on counts as missed. You keep credit for what you achieved before dying. You die when your HP reaches 0. The usual ways to die are: a window ends while you are not inside valid housing; a window ends while a boss you summoned is still alive; or you keep fighting or exploring at low HP. inspect_housing_and_arena tells you whether the place you are standing in is valid housing right now. Building or digging next to a house can break it.

How you are scored (one number from 0 to 1, computed from the server's record of the whole run):
- 70% progress. Your progress is compared with two fixed runs on this same world: a player who does nothing (0) and an expert run (1). Progress is permanent achievements: gear tier (weapon and armor both Wood, then Iron, Silver, Gold: each tier is worth more than the last), valid housing, crafting stations, a boss arena, reaching the cavern layer and the Underworld, Life Crystals used, events beaten, damage dealt to bosses and boss kills (worth the most). Items you hold add a little. Each death subtracts.
- 20% reliability. Half is, for each window, being alive at its end and having made progress recently; the other half is the checkpoints listed below.
- 10% attendance: taking part in every window. It only counts once you have made real progress.
The checkpoints are minimums, not the goal. A run that meets every checkpoint and then stops scores far below an expert run, because unused operations are simply lost. Repeating an action that no longer adds anything (re-crafting, placing and breaking, farming weak enemies) earns nothing or almost nothing.

Checkpoints. Each one is checked once, at the moment the stated window ends.
- End of window 2: you are inside valid housing when the window ends
- End of window 3: your equipped weapon AND equipped armor are both Wood tier or better (tech tier 1)
- End of window 4: you are inside valid housing when the window ends

## How to act

The world runs on a separate game server. You cannot see or edit its code; you
can only call its 24 tools. There are two equivalent ways to call them, and
both hit the same world and the same operation meter:

1. **MCP server `game`** (already connected): tools such as `terraria_status`
   (free), `chop_trees`, `build_blueprint`, `end_window`.
2. **Shell**: `terraria tools` lists every tool with its parameters, and
   `terraria <tool> '<json args>'` calls one, for example
   `terraria chop_trees '{"radius": 60, "max_trees": 4}'`.

Every reply starts with a header line showing the current window, the
operations you have left in it, your HP and your position. Read it before each
decision.

## Rules of the run

- There is exactly one world and one attempt. Nothing resets, and every call
  is final.
- You have 4 windows of 20 operations. When a window's
  operations run out, or you call `end_window`, the rest of that window plays
  out with you standing where you are.
- Keep playing until the final window has ended (the tools will tell you the
  episode is done). If you stop early, the remaining windows are played out
  with you idle, and you are scored on what happens.
- There is no file to write and nothing to submit. Your score is computed from
  the game server's own record of the world when you finish.
