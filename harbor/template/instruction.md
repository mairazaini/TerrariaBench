# TerrariaBench: __TITLE__

__PROMPT__

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
- You have __WINDOWS__ windows of __BUDGET__ operations. When a window's
  operations run out, or you call `end_window`, the rest of that window plays
  out with you standing where you are.
- Keep playing until the final window has ended (the tools will tell you the
  episode is done). If you stop early, the remaining windows are played out
  with you idle, and you are scored on what happens.
- There is no file to write and nothing to submit. Your score is computed from
  the game server's own record of the world when you finish.
