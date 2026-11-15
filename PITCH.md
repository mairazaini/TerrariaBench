# Ten-minute demo

One sentence to open with:

> We built a world where the agent is told everything, including how it is most likely to die, and the best
> model we tested still dies in a third of its runs, saying "I am inside the validated shelter".

## Minute by minute

| Time | Show | Say |
|:---|:---|:---|
| 0:00 | The headline table in `README.md` | "One task, one life, 280 actions. Reference 0.985. Runs come out at 1.00 or at 0.3, nothing in between." |
| 1:00 | `harbor/tasks/terraria-eye-of-cthulhu-hardcore/instruction.md` | "This is everything the agent reads. The scoring weights are in it. Every deadline is in it. The warning about breaking your own house is in it. Only the world seed is missing." |
| 2:00 | Run `uv run python -m evaluation.judge` live (about a minute) | "No API key, no Docker. Reference, do-nothing, random, stop-early, twelve exploit attacks, and a call-by-call replay of every model transcript we saved. If any of it fails, the command fails." |
| 3:30 | `RESULTS.md` section 1 | "Do nothing: zero on all ten tasks. Stop early: always below finishing. Random calls: about zero." |
| 4:30 | `FAILURE_ANALYSIS.md`, the table of nine Kimi runs | "Seven of nine runs broke their own house by building next to it. Four checked, saw it was broken, fixed it: all four scored 1.00. Three did not: all three died that night. One operation out of 280." |
| 6:00 | The run 9 quote | "This one checked three times. The tool said invalid three times. It decided the tool was wrong." |
| 7:00 | `RESULTS.md`, the OpenCode table | "Through your harness the numbers are higher: DeepSeek 1.00, 1.00, 0.52. We expect your re-run to look like that, and we would rather tell you now." |
| 8:00 | "How the benchmark changed" in `docs/DETAILS.md` | "Our first three versions had no honest gap. Hidden deadlines: unfair. Deadlines without scoring: the model stopped early, reasonably. Full scoring: the model matched us. Only then did we see it died in five of six runs and our reward did not care." |
| 9:00 | The small-model table | "A 4B Gemma, trained on 652 expert steps from training tasks only: 0.25 to 0.72, deaths 13 to 3." |
| 9:40 | Limits section | "One world, small samples, a simulator. All written down." |

## Why industry cares (beyond gaming)

When a judge asks **"Who cares if AI can play Terraria?"**:

- **The architecture pattern:** Frontier supervisor models manage long- and short-term planning, while fast decision/reflex layers handle sub-second movement, aim, and reactive control — cutting latency and increasing reliability in continuous 60 Hz interaction.
- **The three failure modes we expose aren't about pixel-art nostalgia:** Even multi-layered multimodal agents struggle with (1) **long-term forethought & commitment** (saving return-home margins, pacing prerequisites), (2) **smart short-term survival reactions** (knowing when to disengage/retreat instead of fighting into an unwinnable loss), and (3) **2D spatial blind spots** (breaking their own workspace invariants without re-verifying).
- **Direct real-world transfer:** Those exact three failure axes bottleneck **autonomous driving** (planner vs. evasive reflex handoff & aborting unsafe merges), **embodied robotics** (re-verifying physical safety after manipulating the workspace), and **long-horizon operational agents** acting outside purely digital sandboxes.

## Questions to expect

**"Why a game? Who cares if AI plays Terraria?"**
Digital code and chat benchmarks reset and have no physical geometry or continuous reactions. Terraria gives us a reproducible, zero-cost testbed for **supervisor + reflex control**, **smart retreat vs. fight decisions**, and **2D spatial self-modification** — the exact bottlenecks holding back self-driving, physical robotics, and real-world agentic execution.

**"Kimi beats your reference. Where is the gap?"**
In reliability, not skill. Six runs beat it; three score 0.28. Mean 0.76 against 0.985. A benchmark that
only measured the good runs would say the problem is solved.

**"Is the house collision a trick?"**
The instruction says building next to a house can break it and names the tool that checks. The check
costs one action. Four runs used it and lived. We also say in `docs/DETAILS.md` that the collision is easy to
cause and that the arena's width is not in the tool description.

**"Why should we trust the model numbers?"**
You do not have to. Every transcript holds the model id, endpoint and every call. `evaluation.judge`
replays the calls through the engine and recomputes the score; an edited file fails.

**"What would you do with another week?"**
More seeds and more runs per model; a zero-death reference for the 360-action task so it can be
Hardcore too; the tuned small model on the Hardcore tasks.

## Do not say

* That frontier models "cannot" do this. They can, two times out of three.
* That the results rank models. Three runs each does not rank anything.
* Anything about real Terraria. The judged environment is the simulator.
