# TerrariaBench: Go deep and come back

You are the player in a Terraria world. You act only through the provided tools. Time is metered in operations, not wall-clock: the world is frozen while you think. Each 12-hour window (Day, then Night, alternating) gives you a fixed budget of operations; every tool call costs one operation except terraria_status, which is free. Spending the last operation or calling end_window lets the rest of the window play out with you standing wherever you are, so be somewhere safe first. Unspent operations do not roll over. Hostiles cannot reach you inside a fully enclosed space; bosses can. Dying drops half of your raw materials and costs you operations in the next window.

Your situation: Establish a base, then push underground: reach the cavern layer, forge iron-tier gear and find a Life Crystal. The depths do not announce their dangers.
You have 10 windows and 16 operations per window.

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

## How you are scored

One score between 0 and 1, with partial credit, computed from the server's
record of the whole run. It rewards real progress on the situation above
(shelter, gear, depth, events survived, bosses fought), doing things by the
time they are needed rather than at the last moment, staying alive, and tool
calls that actually work. Failed, wasted or rule-bending calls count against
you. A run that does nothing scores 0.

## Rules of the run

- There is exactly one world and one attempt. Nothing resets, and every call
  is final.
- You have 10 windows of 16 operations. When a window's
  operations run out, or you call `end_window`, the rest of that window plays
  out with you standing where you are.
- Keep playing until the final window has ended (the tools will tell you the
  episode is done). If you stop early, the remaining windows are played out
  with you idle, and you are scored on what happens.
- There is no file to write and nothing to submit. Your score is computed from
  the game server's own record of the world when you finish.
