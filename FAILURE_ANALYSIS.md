# Why the agents fall short

Every number and quote here comes from the transcripts in `results/runs/` and `results/harbor_agent/`.
`python -m evaluation.judge` recomputes the tables in `RESULTS.md` from those files by replaying their
recorded calls, so nothing below depends on trusting us.

## The short version

On the Hardcore Eye of Cthulhu task (280 actions, one life) the scripted reference scores 0.985 and never
dies. The strongest model we could run, Kimi K3.1 Pro, plays nine runs like this:

| Outcome | Runs | Score |
|:---|---:|:---|
| Kills the Eye, never dies, beats the reference | 6 | 1.00 each |
| Dies in window 4 of 14, asleep in a house it had broken itself | 3 | 0.27, 0.28, 0.28 |

So its mean is 0.76, against 0.985. The model is not short of skill: when it survives it is better than
our script. It is short of one habit: **checking that something it verified earlier is still true after
it changed the world**. The other models fail the same way more often, and add two more failures of
their own.

## Failure 1: trusting a house that is no longer a house

*9 of the 23 first deaths. All 3 of Kimi's Hardcore deaths.*

What happens, every time:

1. Window 1: the agent builds a shelter. The tool reports `"housing": {"valid": true}`.
2. Window 3 or 4: it builds a boss arena near home. The arena is 25 tiles wide. Its edge overwrites
   part of the shelter's wall. The arena call reports the arena as valid and returns its `x_range`.
3. It goes mining, comes home with operations to spare, crafts something, and ends the night.
4. The house leaks. The agent dies in its sleep.

Its last message before dying, in all three Hardcore runs:

> "I am at 85 HP **inside the validated shelter** with Iron armor, a Silver Bow, and a valid arena secured,
> so I will settle the night safely rather than risk another trip with one operation left." (run 3)

> "End the night from **inside the validated shelter** with Iron armor and the Silver Bow equipped,
> preserving all progress and avoiding unnecessary risk at 61 HP." (run 8)

> "Settle the night from **inside the valid shelter** with the arena completed and iron gear equipped,
> avoiding unnecessary nighttime damage at 76 HP." (run 9)

"Validated" refers to a check made before the arena existed.

Run 9 is the sharpest case. The agent did think about the risk. It placed the arena "well left of the
shelter so its footprint cannot break the validated house", and was off by one tile. It then called
`inspect_housing_and_arena` three times from the arena. All three said
`"housing": {"valid": false, "reasons": ["not enclosed: the space around you leaks to the outside"]}`.
Its interpretation:

> "the housing warning likely reflects standing outdoors at the arena rather than d[amage to the shelter]"

That was a reasonable hypothesis. It walked home with four operations left, which was the moment to test
it for one operation. It spent them on a recipe list and smelting, and died.

### What separates the six runs that lived

Same model, same world, same prompt. We replayed all nine runs and recorded the exact call that broke the
house and the call, if any, that repaired it:

| Kimi run | Score | House broken by its arena? | Housing checks afterwards | Repaired? |
|---:|---:|:---|---:|:---|
| 1 | 1.00 | yes, call 58 | 5 | yes, call 64: patched the wall with wood |
| 2 | 1.00 | yes, call 69 | 7 | yes, call 77: rebuilt the shelter |
| 5 | 1.00 | yes, call 81 | 7 | yes, call 124: rebuilt the shelter (after one night outside, survived on armor) |
| 7 | 1.00 | yes, call 46 | 3 | yes, call 60: rebuilt the shelter |
| 4 | 1.00 | no | 1 | not needed |
| 6 | 1.00 | no (arena built 30 tiles away) | 0 | not needed |
| 3 | 0.28 | yes, call 65 | 0 | no: never looked; died that night |
| 8 | 0.28 | yes, call 65 | 0 | no: never looked; died that night |
| 9 | 0.27 | yes, call 62 | 3 | no: saw "invalid" three times and explained it away; died that night |

Seven of the nine runs broke their own house. The four that looked afterwards, believed what they saw and
fixed it all scored 1.00. The three that did not all died the same night. The difference between 1.00 and
0.28 is one operation out of 280.

### Why this is a fair test

* The task text says, in the Hardcore rule: *"inspect_housing_and_arena tells you whether the place you are
  standing in is valid housing right now. Building or digging next to a house can break it."*
* The check costs one operation. Kimi ends the fatal window having spent operations on an inventory
  listing or a recipe list instead.
* The arena call returns the footprint it occupied (`x_range`, `y_range`), and the agent knows where home is.
* The reference builds its arena 32 tiles from home for exactly this reason
  (`terraria_bench/policies.py::_arena`).

What we do not claim: that the collision is hard to cause. It is easy to cause; seven of nine runs
caused it. The task does not punish causing it. It punishes not noticing.

## Failure 2: starting a fight the clock cannot finish

*7 of the 23 first deaths. Mostly DeepSeek V4.1 Flash and Gemma 4 31B.*

A boss that is still alive when a window ends kills the player. The task text says so. The Eye has 2,800
health, and each `engage_combat_burst` call reports the damage it dealt. So before summoning, the agent
can divide.

DeepSeek, Hardcore run 1: summons in window 4 with 17 operations left, wielding an iron broadsword that
deals about 190 a burst. That needs 15 bursts with no room to heal. A few calls from the end:

> "HP 6, boss at 666 HP. Any hit kills me. I have 3 ops left and cannot kill 666 HP in 3 bursts. Fighting
> means near-certain death; the boss being alive at window end also means death. There is no safe path."

It is right, and it was already true when it summoned. Hardcore run 3 does the same in window 6 and ends:

> "Eye at 11% with 297 HP left; one more burst should finish it, and I have 1 op left so I must kill it now
> or die at window end."

Its bursts had been dealing 206 to 218. The Eye survived on 83 health.

Kimi, by contrast, enters the fight with a gold or silver bow in five of its six surviving Hardcore runs
(270 to 370 a burst, 8 to 10 bursts) with 13 to 18 operations left, and kills the Eye inside the window in
all six. The reference summons on the first operation of a night with a silver bow. The information that separates them, damage
per burst and operations remaining, is on screen for both.

## Failure 3: no margin to get home

*7 of the 23 first deaths.*

The agent is underground when the window's operations run out.

> "Out of ops; I'm underground not in housing but window ends, need to be safe. End window to settle."
> (DeepSeek, Hardcore run 2, window 3)

> "I have 1 op left and low HP; I'm underground and unsafe. I'll end the window now since I can't reach
> shelter in one op." (DeepSeek, first-night run 2)

Every `navigate_to` result reports how far it got (`"arrived": false, "steps": 40`), so the cost of the
trip home can be estimated on the way down. The reference keeps a running estimate
(`Pilot.home_cost`) and turns back when its remaining operations equal it.

## The second effect of one life: caution that costs progress

On Hardcore first-night nobody died. The scores still fell: Kimi from 0.96 on the normal task to 0.83
(0.71, 0.81, 0.97), DeepSeek to 0.66. Told that death is final, the models stay near home and stop at
Wood or Iron gear, where the reference (which also never dies) reaches Silver. So the Hardcore rule
does not only remove the careless runs. It makes the careful ones give up reward they could have had
safely.

## What the harness changes

The same model does very differently in different harnesses:

| DeepSeek V4.1 Flash, Hardcore Eye of Cthulhu | Runs | Scores | Mean |
|:---|---:|:---|---:|
| Minimal loop (`evaluation/run_model.py`): one JSON call per turn | 3 | 0.11, 0.33, 0.41 | 0.28 |
| Harbor + OpenCode over MCP (the standard harness) | 3 | 0.52, 1.00, 1.00 | 0.84 |

OpenCode's scaffolding roughly triples the score. Its one failure is Failure 2 again: it summoned the Eye
in window 10 and had not killed it when the window closed. Judges re-running through OpenCode
should expect numbers near the second row, not the first. Kimi K3.1 Pro through OpenCode scored 1.00 on the
one trial that started (two others never began: the agent install timed out on our laptop).

## What does not explain the gap

| Possible excuse | Why it does not apply |
|:---|:---|
| Hidden information | The instruction states the scoring weights, every checkpoint with its window, and the Hardcore rule including the commonest ways to die. A test fails if any graded checkpoint is missing from the text (`tests/test_evaluation.py`). |
| Broken tools | Between 1% and 22% of paid calls are rejected, almost all for a missing ingredient or station, and the rejection says which. A test feeds the reference's calls through the model interface and gets the reference's score. |
| Impossible timing | There is no wall-clock. The world is frozen while the agent thinks. The reference needs 193 of the 280 operations. |
| Too little budget | Kimi's fatal windows end with operations spent on listing inventory or recipes. |
| An unbeatable reference | Kimi beats it in 6 of 9 runs. |

## How we got here

We report this because it changed the benchmark, and because the first three versions did not have a gap.

1. **Deadlines not stated** (the version we inherited). Models scored low. So would anyone: the grader
   checked things the agent was never told. We removed that.
2. **Checkpoints stated, scoring not.** Kimi scored 0.61, 0.71 and 0.91 on first-night. In two of three
   runs it met every checkpoint and then ended the remaining windows without acting: "both earlier
   checkpoints secured, so I'll settle the final night now". A fair reading of an incomplete brief.
   Transcripts: `results/pilot_checkpoints_only/`.
3. **Scoring stated in full.** Kimi scored 0.96 on first-night (reference 1.00) and 0.99 on Eye of Cthulhu
   (reference 0.95). No gap. But it died once in 5 of those 6 runs, and one DeepSeek run died five times and
   still scored 0.76. The reward was not paying for the thing the brief asked for, staying alive.
4. **One life** (Hardcore), with a stronger reference. This is the version reported here.

## Limits of this analysis

* One world (seed 7). The reference is checked on five seeds; the models were run on one.
* Nine runs for Kimi on the headline task, three for the others. A failure rate of 3 in 9 has a wide
  interval: roughly 12% to 65%.
* Two of the three models are not the largest available. We ran what our credits covered.
* The cause labels in `RESULTS.md` are assigned by a rule in `evaluation/judge.py::first_death`
  (boss alive at window end; terrain changed after the last housing check; otherwise away from housing).
  The quotes are verbatim; the labels are our classification.
