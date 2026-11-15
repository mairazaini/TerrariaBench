# Results

Written by `python -m evaluation.judge`. Sections 1, 3 and 4 are recomputed from scratch on every run (deterministic, no API key). Section 2 is recomputed by replaying the calls recorded in each transcript in `results/runs/` through the engine; a file whose saved ledger differs from its replay fails the run. Each file holds the model id, the endpoint, every reply, every tool call and result, and the sealed ledger.

## 1. Reference run and the two baselines

Composite score, mean over seeds 7, 11, 13. The reference is the scripted `smart-triage` policy, replayed through the same 24 tools the agent uses (`harbor run -p harbor/tasks/<task> -a oracle`). Stop-early replays the first N% of the reference's own calls and then quits, so it is the *best possible* way to stop early.

| Task | Ops | Reference | Do nothing | Random calls | Stop at 25% | Stop at 50% | Stop at 75% |
|:---|---:|---:|---:|---:|---:|---:|---:|
| first-night-hardcore | 80 | 1.000 | 0.000 | 0.056 | 0.297 | 0.480 | 0.733 |
| eye-of-cthulhu-hardcore | 280 | 0.969 | 0.000 | 0.004 | 0.200 | 0.535 | 0.758 |
| first-night | 80 | 1.000 | 0.000 | 0.040 | 0.253 | 0.536 | 0.795 |
| cavern-expedition | 160 | 0.996 | 0.000 | 0.092 | 0.269 | 0.611 | 0.832 |
| goblin-siege | 240 | 0.943 | 0.000 | 0.043 | 0.299 | 0.346 | 0.618 |
| blood-moon-survival | 128 | 1.000 | 0.000 | 0.050 | 0.246 | 0.510 | 0.741 |
| arena-architect | 96 | 1.000 | 0.000 | 0.032 | 0.198 | 0.385 | 0.591 |
| eye-of-cthulhu | 280 | 0.957 | 0.000 | 0.041 | 0.197 | 0.467 | 0.614 |
| wall-of-flesh | 240 | 0.976 | 0.000 | 0.237 | 0.355 | 0.876 | 0.950 |
| full-progression | 360 | 0.981 | 0.000 | 0.008 | 0.374 | 0.651 | 0.830 |

Doing nothing scores 0 everywhere, and on every task and seed the full reference run beats every stop-early point. *Random calls* is the mean of 10 episodes of random tools with random arguments on seed 7. It is highest on wall-of-flesh, where the player starts with a full gold kit and a random agent sometimes summons and damages the boss; that task discriminates least and is not one of the two headline tasks.

The same baselines run through Harbor in Docker on this repository (`results/harbor/`):

| Harbor task | Agent | Reward |
|:---|:---|---:|
| terraria-eye-of-cthulhu-hardcore | reference (oracle) | 0.985 |
| terraria-eye-of-cthulhu-hardcore | do nothing (nop) | 0.000 |
| terraria-eye-of-cthulhu-hardcore-stop-early | stop early (first quarter of the reference) | 0.174 |
| terraria-first-night-hardcore | reference (oracle) | 1.000 |
| terraria-first-night-hardcore | do nothing (nop) | 0.000 |
| terraria-first-night-hardcore-stop-early | stop early (first quarter of the reference) | 0.358 |
| terraria-eye-of-cthulhu | reference (oracle) | 0.950 |
| terraria-eye-of-cthulhu | do nothing (nop) | 0.000 |
| terraria-eye-of-cthulhu-stop-early | stop early (first quarter of the reference) | 0.146 |

## 2. The gap: model runs against the reference

### A minimal harness: one JSON tool call per turn

`evaluation/run_model.py`: the model gets the task text (which states the scoring rule and every checkpoint), the 24 tools and the full text of every tool result, and replies with one call per turn. No strategy hints, no scaffolding. Seed 7, same world for everyone. Mean composite, with the range over repeated runs and the number of runs in brackets.

| Player | first-night-hardcore (runs) | eye-of-cthulhu-hardcore (runs) | first-night (runs) | eye-of-cthulhu (runs) |
|:---|---:|---:|---:|---:|
| **Reference (scripted)** | **1.000** | **0.985** | **1.000** | **0.950** |
| Pass threshold | 0.860 | 0.659 | 0.950 | 0.696 |
| DeepSeek V4.1 Flash | 0.660 (0.52-0.77) (3) | 0.284 (0.11-0.41) (3) | 0.695 (0.55-0.80) (3) | 0.908 (0.76-0.99) (3) |
| Gemma 4 31B | 0.648 (0.60-0.74) (3) | 0.403 (0.32-0.51) (3) | 0.716 (0.53-0.84) (3) | 0.711 (0.15-0.99) (3) |
| Kimi K3.1 Pro | 0.827 (0.71-0.97) (3) | 0.759 (0.27-1.00) (9) | 0.961 (0.94-1.00) (3) | 0.989 (0.98-0.99) (3) |
| Do nothing | 0.000 | 0.000 | 0.000 | 0.000 |

### The standard harness: Harbor + OpenCode over MCP

The same tasks played by the OpenCode agent inside Harbor's container, talking to the game's MCP server. This is the path judges re-run. Each row is one trial; its reward, sealed ledger and the agent's full trajectory are in `results/harbor_agent/`.

| Harbor task | Model | Reward | Deaths | Tool calls |
|:---|:---|---:|---:|---:|
| terraria-first-night-hardcore | DeepSeek V4.1 Flash | 0.810 | 0 | 80 |
| terraria-eye-of-cthulhu-hardcore | DeepSeek V4.1 Flash | 0.523 | 1 | 148 |
| terraria-eye-of-cthulhu-hardcore | DeepSeek V4.1 Flash | 0.996 | 0 | 225 |
| terraria-eye-of-cthulhu-hardcore | DeepSeek V4.1 Flash | 1.000 | 0 | 212 |
| terraria-eye-of-cthulhu-hardcore | Kimi K3.1 Pro | 1.000 | 0 | 264 |
| terraria-eye-of-cthulhu-hardcore | Kimi K3.1 Pro | did not start (agent install timed out on our machine) | - | - |
| terraria-eye-of-cthulhu-hardcore | Kimi K3.1 Pro | did not start (agent install timed out on our machine) | - | - |

Means: DeepSeek V4.1 Flash on first-night-hardcore: 0.810 over 1; DeepSeek V4.1 Flash on eye-of-cthulhu-hardcore: 0.840 over 3; Kimi K3.1 Pro on eye-of-cthulhu-hardcore: 1.000 over 1.

### What each run did with its budget

Read from the transcripts. *Unused* is the share of the operation budget never spent. *Looking* is the share of paid calls that only inspect. *Failed* is the share of paid calls the game rejected. *Missing* lists what the reference achieved on this world that the run did not, most valuable first.

| Player | Task | Run | Score | Deaths (first) | Unused | Looking | Failed | Checkpoints missed | Missing vs reference |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|
| DeepSeek V4.1 Flash | eye-of-cthulhu-hardcore | 1 | 0.330 | 1 (window 4) | 72% | 14% | 15% | 4/5 | KILL:eye, T4, T3, UNDERWORLD, CRYSTAL1 |
| DeepSeek V4.1 Flash | eye-of-cthulhu-hardcore | 2 | 0.109 | 1 (window 3) | 79% | 17% | 22% | 4/5 | KILL:eye, T4, T3, T2, UNDERWORLD |
| DeepSeek V4.1 Flash | eye-of-cthulhu-hardcore | 3 | 0.413 | 1 (window 6) | 58% | 13% | 12% | 4/5 | KILL:eye, T4, T3, UNDERWORLD, CRYSTAL1 |
| DeepSeek V4.1 Flash | eye-of-cthulhu | 1 | 0.980 | 2 (window 6) | 30% | 18% | 7% | 0/5 | nothing |
| DeepSeek V4.1 Flash | eye-of-cthulhu | 2 | 0.758 | 5 (window 4) | 24% | 20% | 10% | 0/5 | CRYSTAL1 |
| DeepSeek V4.1 Flash | eye-of-cthulhu | 3 | 0.986 | 2 (window 6) | 10% | 21% | 10% | 0/5 | nothing |
| DeepSeek V4.1 Flash | first-night-hardcore | 1 | 0.522 | 0 (-) | 2% | 19% | 12% | 0/3 | T3, T2, CRYSTAL1, CAVERN, CRYSTAL2 |
| DeepSeek V4.1 Flash | first-night-hardcore | 2 | 0.687 | 0 (-) | 1% | 19% | 10% | 1/3 | T3, CRYSTAL1, CAVERN, CRYSTAL2 |
| DeepSeek V4.1 Flash | first-night-hardcore | 3 | 0.772 | 0 (-) | 20% | 20% | 5% | 0/3 | T3, CRYSTAL1, CRYSTAL2 |
| DeepSeek V4.1 Flash | first-night | 1 | 0.796 | 1 (window 4) | 0% | 15% | 14% | 1/3 | nothing |
| DeepSeek V4.1 Flash | first-night | 2 | 0.550 | 1 (window 3) | 9% | 15% | 16% | 1/3 | T2, CRYSTAL1 |
| DeepSeek V4.1 Flash | first-night | 3 | 0.739 | 1 (window 2) | 8% | 18% | 15% | 1/3 | CRYSTAL1 |
| Gemma 4 31B | eye-of-cthulhu-hardcore | 1 | 0.380 | 1 (window 8) | 66% | 14% | 9% | 3/5 | KILL:eye, T4, T3, UNDERWORLD |
| Gemma 4 31B | eye-of-cthulhu-hardcore | 2 | 0.316 | 1 (window 4) | 72% | 19% | 13% | 4/5 | KILL:eye, T4, T3, UNDERWORLD, CRYSTAL1 |
| Gemma 4 31B | eye-of-cthulhu-hardcore | 3 | 0.513 | 1 (window 12) | 48% | 23% | 7% | 1/5 | KILL:eye, T4, T3, UNDERWORLD |
| Gemma 4 31B | eye-of-cthulhu | 1 | 0.986 | 2 (window 6) | 33% | 29% | 6% | 0/5 | nothing |
| Gemma 4 31B | eye-of-cthulhu | 2 | 0.153 | 3 (window 4) | 21% | 23% | 17% | 4/5 | KILL:eye, T2, ARENA, T1 |
| Gemma 4 31B | eye-of-cthulhu | 3 | 0.993 | 1 (window 8) | 27% | 24% | 7% | 0/5 | nothing |
| Gemma 4 31B | first-night-hardcore | 1 | 0.605 | 0 (-) | 20% | 16% | 9% | 0/3 | T3, T2, CRYSTAL2 |
| Gemma 4 31B | first-night-hardcore | 2 | 0.602 | 0 (-) | 40% | 10% | 19% | 0/3 | T3, T2, CRYSTAL2 |
| Gemma 4 31B | first-night-hardcore | 3 | 0.737 | 0 (-) | 29% | 16% | 11% | 0/3 | T3, CRYSTAL2 |
| Gemma 4 31B | first-night | 1 | 0.773 | 0 (-) | 21% | 14% | 10% | 0/3 | T2 |
| Gemma 4 31B | first-night | 2 | 0.843 | 0 (-) | 38% | 10% | 10% | 0/3 | T2 |
| Gemma 4 31B | first-night | 3 | 0.532 | 0 (-) | 51% | 15% | 8% | 0/3 | T2, CRYSTAL1, CAVERN, ANVIL |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 1 | 1.000 | 0 (-) | 1% | 22% | 1% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 2 | 1.000 | 0 (-) | 1% | 21% | 3% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 3 | 0.281 | 1 (window 4) | 71% | 19% | 6% | 4/5 | KILL:eye, T4, T3, UNDERWORLD, CRYSTAL2 |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 4 | 1.000 | 0 (-) | 0% | 20% | 2% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 5 | 1.000 | 0 (-) | 0% | 25% | 3% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 6 | 1.000 | 0 (-) | 0% | 19% | 4% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 7 | 1.000 | 0 (-) | 1% | 20% | 5% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 8 | 0.283 | 1 (window 4) | 71% | 21% | 2% | 4/5 | KILL:eye, T4, T3, UNDERWORLD, CRYSTAL2 |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 9 | 0.267 | 1 (window 4) | 71% | 21% | 1% | 4/5 | KILL:eye, T4, T3, UNDERWORLD, CRYSTAL2 |
| Kimi K3.1 Pro | eye-of-cthulhu | 1 | 0.989 | 1 (window 4) | 2% | 23% | 5% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu | 2 | 0.993 | 1 (window 4) | 2% | 20% | 3% | 0/5 | nothing |
| Kimi K3.1 Pro | eye-of-cthulhu | 3 | 0.984 | 1 (window 4) | 2% | 21% | 4% | 0/5 | nothing |
| Kimi K3.1 Pro | first-night-hardcore | 1 | 0.967 | 0 (-) | 0% | 15% | 4% | 1/3 | nothing |
| Kimi K3.1 Pro | first-night-hardcore | 2 | 0.705 | 0 (-) | 0% | 24% | 4% | 0/3 | T3, T2 |
| Kimi K3.1 Pro | first-night-hardcore | 3 | 0.808 | 0 (-) | 0% | 21% | 4% | 0/3 | T3 |
| Kimi K3.1 Pro | first-night | 1 | 1.000 | 0 (-) | 0% | 12% | 4% | 0/3 | nothing |
| Kimi K3.1 Pro | first-night | 2 | 0.942 | 1 (window 4) | 0% | 16% | 5% | 1/3 | nothing |
| Kimi K3.1 Pro | first-night | 3 | 0.942 | 1 (window 2) | 8% | 16% | 9% | 1/3 | nothing |

### How each run first died

Every death in the saved runs, with the cause the game reported and the reason the model gave for the call that killed it (quoted from the transcript). In a Hardcore task the first death ends the run.

| Player | Task | Run | Died in window | Cause | The model's stated reason for its last call |
|:---|:---|---:|:---|:---|:---|
| DeepSeek V4.1 Flash | eye-of-cthulhu-hardcore | 1 | 4 of 14 | boss fight not finished by the end of the window | "HP 6, boss at 666 HP. Any hit kills me. I have 3 ops left and cannot kill 666 HP in 3 bursts. Fighting means near-certain death; the boss being alive at window end also means death. There is no safe path, but the only nonzero chan" |
| DeepSeek V4.1 Flash | eye-of-cthulhu-hardcore | 2 | 3 of 14 | window ended away from valid housing | "Out of ops; I'm underground not in housing but window ends — need to be safe. End window to settle." |
| DeepSeek V4.1 Flash | eye-of-cthulhu-hardcore | 3 | 6 of 14 | boss fight not finished by the end of the window | "Eye at 11% with 297 HP left; one more burst should finish it, and I have 1 op left so I must kill it now or die at window end." |
| DeepSeek V4.1 Flash | eye-of-cthulhu | 1 | 6 of 14 | window ended away from valid housing | "Eye is dead and threat is calm; end the window safely to bank the kill and progress." |
| DeepSeek V4.1 Flash | eye-of-cthulhu | 2 | 4 of 14 | boss fight not finished by the end of the window | "Engage the Eye of Cthulhu with a combat burst to deal damage while kiting." |
| DeepSeek V4.1 Flash | eye-of-cthulhu | 3 | 6 of 14 | boss fight not finished by the end of the window | "Engage the boss without specifying a target id to deal damage this window." |
| DeepSeek V4.1 Flash | first-night | 1 | 4 of 4 | house broken by its own later building, never re-checked | "End the window safely inside base housing with iron gear equipped." |
| DeepSeek V4.1 Flash | first-night | 2 | 3 of 4 | window ended away from valid housing | "I have 1 op left and low HP; I'm underground and unsafe. I'll end the window now since I can't reach shelter in one op, but first check status is free—actually just end to settle." |
| DeepSeek V4.1 Flash | first-night | 3 | 2 of 4 | window ended away from valid housing | "Return to shelter to be safe before the window ends, since I'm out of ops for progress." |
| Gemma 4 31B | eye-of-cthulhu-hardcore | 1 | 8 of 14 | window ended away from valid housing | "I am now back in my shelter. I have a Gold Broadsword and Iron Armor, meeting the Tech Tier 2 requirement for the Window 8 checkpoint. I will end the window now to secure the checkpoint and stay safe." |
| Gemma 4 31B | eye-of-cthulhu-hardcore | 2 | 4 of 14 | boss fight not finished by the end of the window | "My housing is currently not valid (not enclosed), and I am at very low HP with a boss active. I must urgently seal the shelter to ensure I survive the window end, as failing to be in valid housing at the end of a window in Hardcor" |
| Gemma 4 31B | eye-of-cthulhu-hardcore | 3 | 12 of 14 | boss fight not finished by the end of the window | "I have reached my shelter and met the 60% HP checkpoint for window 12. Since I am at critical HP and cannot heal, I will end the window now to ensure I am safe inside valid housing, avoiding death from the boss or the window settl" |
| Gemma 4 31B | eye-of-cthulhu | 1 | 6 of 14 | house broken by its own later building, never re-checked | "The Eye of Cthulhu has been defeated, completing the final checkpoint early. I will now end the window to settle the progress and recover." |
| Gemma 4 31B | eye-of-cthulhu | 2 | 4 of 14 | window ended away from valid housing | "I am inside the shelter at [95, 15]. I will attempt to smelt the iron ore one last time before the window ends." |
| Gemma 4 31B | eye-of-cthulhu | 3 | 8 of 14 | house broken by its own later building, never re-checked | "The Eye of Cthulhu has been defeated. I am at very low health, so I will end the window now to recover and settle the victory." |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 3 | 4 of 14 | house broken by its own later building, never re-checked | "I am at 85 HP inside the validated shelter with Iron armor, a Silver Bow, and a valid arena secured, so I will settle the night safely rather than risk another trip with one operation left." |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 8 | 4 of 14 | house broken by its own later building, never re-checked | "End the night from inside the validated shelter with Iron armor and the Silver Bow equipped, preserving all progress and avoiding unnecessary risk at 61 HP." |
| Kimi K3.1 Pro | eye-of-cthulhu-hardcore | 9 | 4 of 14 | window ended away from valid housing | "Settle the night from inside the valid shelter with the arena completed and iron gear equipped, avoiding unnecessary nighttime damage at 76 HP." |
| Kimi K3.1 Pro | eye-of-cthulhu | 1 | 4 of 14 | house broken by its own later building, never re-checked | "Use the final operation to reach the valid enclosed shelter so the window settles safely after the boss kill." |
| Kimi K3.1 Pro | eye-of-cthulhu | 2 | 4 of 14 | house broken by its own later building, never re-checked | "Use the final operation safely inside the shelter to smelt the remaining gold ore into two more bars, banking progress toward a Gold-tier weapon or armor before the night settles." |
| Kimi K3.1 Pro | eye-of-cthulhu | 3 | 4 of 14 | boss fight not finished by the end of the window | "The Eye is at 466 HP but I cannot safely kill it before settlement, so I will use the final operation to retreat, stabilize, and avoid a death while preserving the substantial boss damage already dealt." |
| Kimi K3.1 Pro | first-night | 2 | 4 of 4 | house broken by its own later building, never re-checked | "Smelt all six remaining iron ore into three bars while standing inside the validated shelter; the final operation will then settle the night safely indoors with the added crafted materials secured." |
| Kimi K3.1 Pro | first-night | 3 | 2 of 4 | house broken by its own later building, never re-checked | "Use the final operation safely inside the shelter to craft a campfire for later arena regeneration, then let the night settle with the housing checkpoint secured." |

23 of 42 runs had at least one death. Causes of the first death: house broken by its own later building, never re-checked (9); boss fight not finished by the end of the window (7); window ended away from valid housing (7). In 14 of those 23 the model's own reason for its last call says it is safe.

The written analysis of these runs is in `FAILURE_ANALYSIS.md`.

## 3. Reward exploits, attempted

Each row is an attack that is actually executed against the engine by `terraria_bench/exploits.py`, then graded. *Closed* means every check held.

| Exploit | What the attacker does | Why it earns nothing | Result |
|:---|:---|:---|:---|
| continuity_for_quitters | Attend a few windows, then quit and keep the continuity ratio. | Continuity's denominator is the task's full window count, never the windows played. | closed |
| attendance_farming | Call end_window every window and do nothing, to collect the attendance term. | Attendance credit is scaled by normalized progress; with none, attending is worth exactly zero. | closed |
| dirt_box_camping | Seal yourself in and idle to farm survival. | Survival credit needs recent progress and decays with idleness; half of reliability is SLA checkpoints. | closed |
| place_break_farming | Place blocks and mine them back for mining value. | Tiles carry placed-by-player provenance and drop nothing; milestones are unique unlocks. | closed |
| slime_grinding | Farm trivial hostiles for combat progression. | Kill value is capped per family; boss credit is locked to the HP high-water mark. | closed |
| boss_resummon_scumming | Despawn and re-summon a boss to re-earn the opening damage. | Boss credit uses the episode-wide minimum HP ratio per boss type. | closed |
| craft_trash_cycling | Craft, trash and re-craft the same item. | Crafting credit is a set indicator over unique items; trashing only loses salvage value. | closed |
| suicide_respawn_teleport | Die on purpose to teleport home for free. | A death forfeits progression and costs more next-window operations than walking home. | closed |
| last_window_dumping | Start a boss fight at the end and let the episode expire. | Settlement resolves any live boss fight as a loss, including in the final window. | closed |
| partial_enclosure_spoofing | Claim housing with a leaky or doorway-only structure. | Housing is verified by flood fill; any gap, platform hole or open side fails. | closed |
| chest_duplicate_looting | Loot the same container or crystal twice. | Looting consumes the tile; a second attempt finds nothing. | closed |
| healing_potion_stalling | Chain potions to out-heal a fight indefinitely. | Potion Sickness blocks a second potion for 60 seconds of simulated time. | closed |

## 4. A smaller model trained on experience from the environment

`google/gemma-4-E4B-it` (a 4B-class model), before and after a rank-32 LoRA trained by behaviour cloning on 652 turn-by-turn steps of the reference policy on the four *training* tasks. Same prompt, greedy decoding, one episode per cell, official composite. Source: `artifacts/post_training/bc_eval.json`; reproduce with `modal run post_training/modal_grpo_gemma.py::eval_bc`. This run predates the scoring text now in the prompt.

| Cell | Seen in training? | Base | Tuned | Deaths (base / tuned) |
|:---|:---|---:|---:|:---:|
| woodland-scramble:7 | task and seed | 0.491 | 0.916 | 1 / 0 |
| ore-rush:7 | task and seed | 0.145 | 0.582 | 3 / 1 |
| woodland-scramble:17 | task only (new world) | 0.307 | 0.976 | 0 / 0 |
| shelter-crunch:17 | task only (new world) | 0.238 | 0.609 | 1 / 0 |
| slime-rain:17 | task only (new world) | 0.150 | 0.800 | 2 / 0 |
| ore-rush:17 | task only (new world) | 0.145 | 0.932 | 2 / 0 |
| first-night:7 | no (evaluation task) | 0.312 | 0.620 | 1 / 0 |
| cavern-expedition:7 | no (evaluation task) | 0.201 | 0.324 | 3 / 2 |
| **Mean** | | **0.249** | **0.720** | |

