# Details

## Tasks

Harbor tasks, in `harbor/tasks/`:

| Task | Actions | What it is |
|:---|---:|:---|
| `terraria-eye-of-cthulhu-hardcore` | 280 | **Headline.** Prepare, build an arena, kill the boss. One life. |
| `terraria-first-night-hardcore` | 80 | Shelter and first gear. One life. |
| `terraria-eye-of-cthulhu`, `terraria-first-night` | 280, 80 | The same two without the one-life rule. Current models match the reference here. |
| `cavern-expedition`, `goblin-siege`, `blood-moon-survival`, `arena-architect`, `wall-of-flesh`, `full-progression` | 96 to 360 | Baselines and exploits verified; no model results reported. |

`harbor/baselines/` holds a stop-early variant of every task: its "solution" is the first quarter of the
reference run, then nothing.

## The environment

**Setting.** A deterministic 2D survival world modelled on Terraria's early game: trees, ore veins,
caves, day and night, enemies that find an unsheltered player, a tech tree (wood, iron, silver, gold)
and a boss. We chose it because the early game is a real planning problem that people learn from a
wiki: every goal has prerequisites, night punishes whoever is unprepared, and every action changes a
world you then have to keep track of.

**Actions.** 24 tools over MCP: status, observation, gathering and movement, crafting and building,
combat. Each call costs one operation, except `terraria_status`, which is free.

**Time.** The world is frozen while the agent thinks. A task is a number of 12-hour windows, each with
a budget of operations. When the budget is spent, the rest of the window plays out with the player
standing where they are. The scarce resource is decisions, not seconds, and a slow model is not
penalised.

**What the agent is told** (`terraria_bench/tasks.py`; the same text is the Harbor instruction):

* the rules of time, shelter and death;
* the scoring rule with its weights (`SCORING`);
* every checkpoint the grader checks, with its window number (`checkpoint_text`);
* in Hardcore tasks, that the first death ends the run, the three usual ways to die, the tool that
  checks whether you are safe, and that building next to a house can break it (`HARDCORE_RULE`).

**The reference.** A scripted policy (`terraria_bench/policies.py`, `smart-triage`) that plays through
the same 24 tools. For Hardcore tasks it is run at nine pacings and the best is kept. It never dies on
any of the five calibrated seeds and uses 193 of the 280 operations on the headline task. It is replayed
by `harbor run ... -a oracle`. It is not a ceiling: Kimi out-scores it when it survives.

## The reward

```
score = 0.7 * clip((progress - absent) / (expert - absent), 0, 1)     progress
      + 0.2 * (0.5 * survival + 0.5 * checkpoints)                     reliability
      + 0.1 * attendance * clip(progress_term / 0.10, 0, 1)            attendance, gated on progress
```

* **Progress** is the growth of a value made of permanent achievements: gear tier, housing, stations,
  arena, depth, Life Crystals, boss damage and kills. Each counts once, as a high-water mark.
* **Calibration.** `absent` and `expert` are measured per task and seed by running scripted anchor
  policies on that exact world. `python -m terraria_bench.calibrate` rebuilds them and checks four gates
  on all 70 task-seed cells.
* **Hardcore.** The first death freezes the run: what was achieved before it is kept, every later
  window is unattended and every later checkpoint is missed.
* **Sealed ledger.** The game runs in a sidecar container the agent cannot read or restart. The score
  is computed from the server's record, sealed with SHA-256 over the world, the source and the
  calibration. A missing or edited ledger grades 0.

Shortcuts, measured (`RESULTS.md` sections 1 and 3):

| Shortcut | Score |
|:---|:---|
| Do nothing | 0.000 on all 10 tasks |
| Turn up to every window and only call `end_window` | 0.000 |
| Random tools with random arguments | 0.00 to 0.09 on 9 tasks; 0.24 on wall-of-flesh |
| Play the reference perfectly, then stop at 25% / 50% / 75% | below the full run on every task and seed |
| Twelve catalogued exploits (place-and-break farming, kill farming, boss re-summoning, sealing yourself in a box, attendance farming, ...) | all closed; each is attacked for real in `terraria_bench/exploits.py` |
| Edit a saved transcript's score | `evaluation.judge` replays its calls and fails on a mismatch |

## Commands we used for the model runs

```bash
# Harbor + OpenCode with an OpenAI-compatible endpoint
harbor run -p harbor/tasks/terraria-eye-of-cthulhu-hardcore -a opencode -m verda/kimi-k3-1-pro -k 3 \
  --api-key-name VERDA_API_KEY --env-file .env \
  --ak 'opencode_config={"provider":{"verda":{"npm":"@ai-sdk/openai-compatible","options":{"baseURL":"https://inference.verda.com/v1","apiKey":"{env:VERDA_API_KEY}"},"models":{"kimi-k3-1-pro":{}}}}}'

# Minimal loop: one JSON tool call per turn; writes one transcript per run to results/runs/<model>/
uv run python -m evaluation.run_model --base-url https://inference.verda.com/v1 --key-env VERDA_API_KEY \
    --model kimi-k3-1-pro --tasks eye-of-cthulhu-hardcore --repeats 9

# Tests: calibration gates, exploit regressions, determinism, Hardcore rules, fairness of instructions
uv run pytest
```

After any change to `terraria_bench/`, run `python -m terraria_bench.calibrate` and
`python harbor/build_tasks.py`.

## A smaller model trained on experience from the environment

`post_training/` turns the scripted policy's play on four *training* tasks (disjoint from the evaluation
tasks) into 652 turn-by-turn examples, keeps an episode only if it grades at least 0.95 with no deaths,
and trains a rank-32 LoRA on `google/gemma-4-E4B-it`.

| | Base | Tuned |
|:---|---:|---:|
| Mean composite over 8 cells | 0.249 | **0.720** |
| Deaths over 8 episodes | 13 | 3 |
| Unseen worlds of training tasks (4 cells) | 0.210 | 0.829 |
| Evaluation tasks never trained on (2 cells) | 0.257 | 0.472 |

One greedy episode per cell on the normal tasks; per-cell numbers in `RESULTS.md` section 4. This was
measured before the scoring text and Hardcore tasks were added, and the tuned model has not been run on
Hardcore. The adapter is on a Modal volume, not in this repository;
`modal run post_training/modal_grpo_gemma.py::eval_bc` re-runs the comparison.

## How the benchmark changed

1. The version we started from did not tell the agent its deadlines. Models scored low for a reason
   that was our fault.
2. With deadlines stated but not the scoring, Kimi met them and stopped playing
   (`results/pilot_checkpoints_only/`).
3. With the scoring stated in full, Kimi scored 0.96 and 0.99. No gap. But it died in 5 of 6 runs and
   barely lost points for it.
4. So the headline tasks became Hardcore, with a stronger reference. That is what the README reports.

## Limits

* **Not real Terraria.** The judged environment is a simulator written for this benchmark. The
  repository also holds an experimental tModLoader bridge to the real game (`mod/`,
  `terraria_bench/real_game.py`). No reported number comes from it.
* **Small samples, one world.** Nine runs for one model, three for the others, all on seed 7. The
  3-in-9 failure rate has a 95% interval of roughly 12% to 65%.
* **Thin OpenCode evidence for the strongest model.** One Kimi trial through OpenCode completed; two
  never started because the agent install timed out on our laptop.
* **A collision that is easy to cause.** The arena preset is 25 tiles wide and overwrites what it
  touches. The tool description does not state the width; the Hardcore rule warns about building near
  a house and names the tool that checks. Seven of nine Kimi runs broke their house this way; the four
  that checked and repaired scored 1.00.
* **wall-of-flesh is the weakest task**: the player starts with a full gold kit and random play
  reaches 0.24.

## Repository map

| Path | What it is |
|:---|:---|
| `terraria_bench/engine.py`, `tools.py`, `tasks.py` | The world, the 24 tools, the tasks and the text agents are given |
| `terraria_bench/grader.py`, `calibrate.py`, `exploits.py`, `provenance.py` | Reward, calibration, exploit attacks, sealing |
| `terraria_bench/policies.py` | The scripted anchors, including the reference |
| `evaluation/judge.py`, `evaluation/run_model.py` | One-command evaluation; model runner that saves full transcripts |
| `harbor/tasks/`, `harbor/baselines/`, `harbor/build_tasks.py` | Harbor tasks (generated) |
| `results/runs/`, `results/harbor/`, `results/harbor_agent/` | Model transcripts and Harbor job outputs behind `RESULTS.md` |
| `post_training/` | Dataset builder, LoRA training and evaluation for the small model |
| `tests/` | Calibration gates, exploit regressions, determinism, Hardcore rules, fairness of instructions |
| `mod/`, `terraria_bench/real_game.py`, `eye_planner.py`, `dashboard/`, `telemetry/`, `agents/`, `mcp_server/`, `tools/` | Experimental real-game bridge, viewers and helpers; not part of the judged path |
