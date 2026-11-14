# TerrariaBench

TerrariaBench is a deterministic, long-horizon agent evaluation and reinforcement learning benchmark over the Model Context Protocol (MCP), engineered to mirror the methodology, mathematical grading, anchor calibration, and anti-exploit architecture of AfterQuery's StoreBench (arXiv:2610.10942v2).

Everything runnable — the headless world simulator, 24-tool MCP surface, port 8765 bridge server, human operator console, live leaderboard dashboard, and evaluation/RL harnesses — is implemented in pure Python (>= 3.11) with zero Node.js/npm build steps or required .NET tooling.

## At-a-Glance for Judges

**What TerrariaBench is:** A deterministic, long-horizon survival and boss-progression RL environment built natively for **Harbor (`opencode`)** and **MCP**. Across 8 evaluation scenarios (spanning 4 to 30 twelve-hour day/night windows), an agent uses 24 metered tools to gather resources, climb a multi-stage crafting tech tree, build shelter and boss arenas, and defeat bosses like the Eye of Cthulhu. At episode end, an isolated sidecar verifier settles any unplayed windows and writes a single calibrated score in `[0, 1]` to `/logs/verifier/reward.txt`.

| Judging Criterion | How TerrariaBench Delivers |
| :--- | :--- |
| **1. The Gap (30 pts)** | Our scripted reference policy (`-a oracle` / `smart-triage`) consistently achieves **0.950–1.000** composite (**100%** Eye progress). Frontier agents (`-a opencode`) score significantly lower across repeated seeded runs (`gemini-2.5-pro` and `codex` score **0.588** composite, while un-tuned `gemma-base` achieves **0.172** composite), while `do-nothing` (`0.000`) and `stop-early` (`0.100` / `< 0.050` on boss scenarios) baselines earn near-zero credit. |
| **2. Meaningful Difficulty (30 pts)** | Every recipe, threat timer, and world state is fully observable via `instruction.md`, a **free `0-op` `terraria_status` header**, and telemetry tools—nothing is unfairly withheld. Implicit environmental deadlines (observable via telemetry, not spoon-fed) challenge frontier agents, which fail on three concrete planning bottlenecks: **(a) Observation over-spending** (wasting 35–50% of window op budgets re-scanning instead of acting), **(b) Tech-tree ordering errors** (trying to craft metal gear or summon bosses before placing a furnace, anvil, or arena), and **(c) Poor dusk preparation** (mining until the final daytime op instead of building shelter, leading to night deaths that drop 50% of materials with `ρ = 0.15` salvage discount). |
| **3. Reward Integrity (25 pts)** | Graded in an isolated sidecar container via baseline-subtracted Net Progression Value (`z`), high-water-mark milestone tracking, and a raw-material salvage discount (`ρ = 0.15`). **11 reward-hacking strategies**—including early stopping, idle window attendance, craft/trash loops, starter-kit liquidation, and kill farming—are structurally blocked and regression-tested in `tests/test_reward_exploits.py` (`exploiter` policy scores `0.011 < 0.050`). |
| **4. Craft & Reproducibility (15 pts)** | 100% pure Python (>= 3.11) symbolic engine with purpose-keyed SHA-256 RNG streams (`provenance.stream`) for byte-identical replay. Every Harbor task (`harbor/tasks/*`), baseline (`oracle`, `nop`, `absentee`), and test suite runs in one command. |
| **Tiebreaker (Post-Training)** | Fine-tuning `google/gemma-4-E4B-it` (rank-32 LoRA) on 652 filtered expert turns from 4 disjoint training tasks lifts mean composite performance across the 8 evaluation tasks from **0.249 (base)** to **0.720 (tuned)** (`post_training/`). In our standard evaluation task `first-night`, GRPO post-training elevates Gemma composite performance from **0.172 (17.18 / 100)** to **0.638 (63.78 / 100)**. |

## Reproduce the results

```bash
# 1. Re-run all 6 scripted anchors (do-nothing, absentee/stop-early, exploiter, rule-based, blast-list, smart-triage) & rewrite EVAL_RESULTS.md
uv run python reporting/generate_report.py

# 2. Run calibration gates, exploit regressions, determinism, and Harbor task sync tests (< 15s)
uv run --extra dev --extra agents pytest

# 3. Harbor standard evaluation runs (Reference, Do-Nothing, Stop-Early, and 3 Frontier Agent runs)
harbor run -p harbor/tasks/terraria-eye-of-cthulhu -a oracle   # reference run through Harbor (0.950 composite, 100% Eye progress)
harbor run -p harbor/tasks/terraria-eye-of-cthulhu -a nop      # do-nothing baseline through Harbor (0.000)
harbor run -p harbor/tasks/terraria-eye-of-cthulhu -a absentee # stop-early / absentee baseline (< 0.050)
harbor run -p harbor/tasks/terraria-eye-of-cthulhu -a opencode -m <provider/model> --env-file .env -n 3
```

---

## Harbor Tasks & Sidecar Architecture

The eight evaluation scenarios ship as [Harbor](https://github.com/laude-institute/harbor) tasks in `harbor/tasks/`, each with `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/` and `solution/`.

```bash
harbor run -p harbor/tasks/terraria-first-night -a opencode -m gemini/gemini-4-carbon \
    --api-key-name GEMINI_API_KEY --env-file .env
harbor run -p harbor/tasks/terraria-first-night -a oracle   # reference solution
harbor run -p harbor/tasks -a opencode -m <provider/model>  # all eight scenarios
```

`.env` holds the model provider's API key (for Google models, `GEMINI_API_KEY`). Harbor's provider prefix for Google is `gemini/`, and `--api-key-name` is needed for model ids its registry does not know yet. Docker is required: each task is a two-container environment.

**What the agent has to do.** Play a Terraria-like survival world to the end of a scenario: survive the first night, dig to the caverns, hold off a goblin siege or a blood moon, build a boss arena, fight the Eye of Cthulhu or the Wall of Flesh, or do all of it in one 30-window marathon. The agent acts only through 24 game tools (scan, navigate, chop, mine, craft, build, equip, fight, retreat, and so on), offered both as an MCP server and as a `terraria` shell command. Time is metered in operations rather than wall-clock: each 12-hour day or night window allows a fixed number of tool calls, and when they run out the rest of the window plays out with the player wherever it was left standing.

**Why it is hard.**

- *Long horizon, one attempt.* A scenario is 80 to 360 metered calls against a single world that never resets. A night spent outside, or a death that drops half the player's materials, compounds through every later window.
- *Scarce actions.* Looking costs the same as doing. An agent that inspects everything runs out of operations before it has built anything.
- *Implicit environmental deadlines (observable via telemetry, not spoon-fed).* Every threat timing, recipe, and world state is fully accessible in `instruction.md`, the free `0-op` `terraria_status` header, `list_craftable_recipes`, and `query_threat_telemetry`. Yet frontier models struggle to anticipate deadlines (shelter before first dusk, armour before the siege, an arena before the boss) and must plan proactively from telemetry signals rather than waiting for failure prompts.
- *Dependent progress.* Gear follows a tech tree (wood, then ore, furnace, anvil, armour), bosses need an arena and a summoning item, and the Underworld needs depth and a bridge. Steps done out of order are wasted.
- *No shortcuts.* The simulator, the calibration and the score ledger live in a sidecar container the agent cannot read or restart. Eleven known reward exploits (free tiles, kill farming, sell-and-rebuy loops, quitting early, and others) are closed and regression-tested in `tests/test_reward_exploits.py`.

**How it is scored.** When the agent stops, the verifier asks the game server to settle any windows the agent did not play (with the player idle) and to grade the sealed episode ledger. It writes one number between 0 and 1 to `/logs/verifier/reward.txt`, with the breakdown in `grade.json` and the full ledger in `ledger.json` beside it:

```
reward = 0.7 * clip((growth - z) / s, 0, 1)                       # progress
       + 0.2 * (0.5 * survival_rate + 0.5 * deadlines_met_rate)   # reliability
       + 0.1 * (windows_attended / windows_in_task)               # continuity
```

`growth` is the increase in Net Progression Value (crafted gear, structures, depth, events survived, boss damage). `z` is what an absent player gets on that world and `s` is the gap from there to the best scripted policy, both taken from a sealed calibration file, so 0 means "no better than doing nothing" and 1 means "as good as the best scripted policy". Partial credit is the norm. The reference solution replays that scripted policy and scores between 0.95 and 1.0 depending on the scenario; an agent that does nothing scores 0.

Tasks are generated from `harbor/template/` by `python harbor/build_tasks.py`. Re-run it after changing `terraria_bench/` or `artifacts/`; `tests/test_harbor_tasks.py` fails if the copies drift.

---

## Committed Evaluation Run Logs & Provenance

The repository includes committed evaluation trajectories and sealed ledgers verifying live agent runs across evaluation scenarios:
- **Eye of Cthulhu Live Run Trajectory & Transcript:** [artifacts/recordings/eye-of-cthulhu_seed7_live.json](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/recordings/eye-of-cthulhu_seed7_live.json) (interactive playback: [artifacts/recordings/eye-of-cthulhu_seed7_live.html](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/recordings/eye-of-cthulhu_seed7_live.html))
- **First Night Survival Live Run:** [artifacts/recordings/first-night_seed7_live.json](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/recordings/first-night_seed7_live.json) (interactive playback: [artifacts/recordings/first-night_seed7_live.html](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/recordings/first-night_seed7_live.html))
- **Real Terraria Bridge Session:** [artifacts/recordings/real_terraria_live.json](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/recordings/real_terraria_live.json)
- **Human Reference Baseline:** [artifacts/runs/human/first-night_seed7.json](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/runs/human/first-night_seed7.json)
- **Out-of-Process Aiven HMAC Audit Ledger:** [artifacts/telemetry/audit_ledger.db](file:///Users/krishnasuri/.gemini/antigravity/scratch/TerrariaBench_repo/artifacts/telemetry/audit_ledger.db)

---

## Quickstart (Windows PowerShell & POSIX)

Windows laptops without `make` can invoke every workflow directly via `python -m` (or `uv run`):

```powershell
# 1. Install in editable mode (dependencies: pydantic, numpy, fastapi, uvicorn)
python -m pip install -e ".[dev]"

# 2. Run calibration across all 6 scripted anchors and seeds 7, 11, 13 (+ held-out 17, 23)
python -m terraria_bench.calibrate

# 3. Run unit, calibration-gate, exploit-regression, and replay tests (< 15s)
uv run --with pytest pytest tests/ -v

# 4. Run standard evaluation and print StoreBench Table 1 & Table 3
python -m terraria_bench.harness.runner --eval-standard
```

### Interactive Human & Bridge Commands

```powershell
# Record a demonstration trajectory and sealed SHA-256 ledger
python -m terraria_bench.human_recorder --record-demo --task first-night --seed 7

# Replay & verify byte-identical ledger reproduction for all recorded trajectories
python -m terraria_bench.human_recorder --verify-all

# Launch the pure-Python JSON-RPC bridge server on port 8765
python -m terraria_bench.human_recorder --serve --port 8765

# Launch the StoreBench §5 Human Operator Console (Web UI or --cli)
python -m terraria_bench.web_console --task first-night --seed 7 --port 8080
python -m terraria_bench.web_console --task first-night --seed 7 --cli

# Launch the FastAPI Live Leaderboard & Trajectory Audit Dashboard
python -m uvicorn dashboard.app:app --port 8000
```

POSIX users can equivalently run `make install`, `make calibrate`, `make test`, `make test-exploits`, `make eval`, `make train-gemma`, `make dashboard`, `make console`, or `make bridge`.

---

## StoreBench Architectural Parity

| StoreBench Pillar (`arXiv:2610.10942v2`) | TerrariaBench Implementation |
| :--- | :--- |
| **Windowed Operation Budgets (§3.4)** | Each episode spans $W$ 12-hour windows (`27,000` ticks at 60 Hz, alternating Day/Night) with budget $B \in [11, 20]$ ops/window. World freezes while the agent thinks; all tools cost 1 op except `terraria_status` (`0` ops). |
| **One-Line State Header (§3.4)** | Every tool output prepends `[Seed: ... \| Win: ... \| Ops: ... \| HP: ... \| Mana: ... \| Def: ... \| NetProg: ... \| TechTier: ... \| Threat: ...]`. |
| **3 Perturbation Channels (§3.2)** | **Announced** (Goblin Army, Blood Moon with lead time), **Rumored** (Eye of Cthulhu, Cave Chill), and **Silent** (nocturnal spawn multipliers, hidden traps, corruption spread, Eye Phase 2 enrage). |
| **Task Lenses & Shared Seeds (§3.3)** | 8 evaluation tasks (`EVAL_TASKS`) + 4 disjoint training tasks (`TRAIN_TASKS`) sharing seeded worlds (`7, 11, 13` reported; `17, 23` held out). |
| **Composite Grader & Dense Channel (§4.1, App. C)** | $R = 0.7 \cdot \text{clip}_{[0,1]}((\Delta\text{net} - z)/s) + 0.2 \cdot (0.5 \cdot \text{survival} + 0.5 \cdot \text{sla}) + 0.1 \cdot (\text{attended}/W)$ with $\rho = 0.15$ raw salvage discount and $[-2, 1]$ clipped dense GRPO channel. |
| **6 Anchors & 4 Gates (§4.2)** | `do-nothing`, `absentee`, `blast-list`, `rule-based`, `smart-triage`, and `exploiter`, validated against Ordering, Judgment Gap ($\ge 0.15$), Flap Margin ($\ge 0.02$), and Exploit Suppression ($< 0.05$). |
| **Audit-Close-Pin Exploit Catalogue (§4.3, App. F)** | 11 closed reward exploits registered in `terraria_bench.grader.EXPLOITS`, enforced structurally in `engine.py` / `grader.py` / `tools.py`. |
| **Determinism & Fail-Closed Provenance (App. G)** | Purpose-keyed SHA-256 RNG streams (`provenance.stream`) and SHA-256 sealed ledgers/artifacts; missing or tampered ledgers grade to `0.0`. |

---

## The 24 MCP Tools (5 Families)

1. **Status & Time (2):** `terraria_status` *(free, 0 ops)*, `end_window`
2. **Observation & Spatial Inspection (5):** `observe_viewport` *(Base64 PNG + 61x35 ASCII)*, `scan_biome_and_ores`, `inspect_inventory`, `list_craftable_recipes`, `inspect_housing_and_arena`
3. **Gathering, Mining & Navigation (5):** `navigate_to`, `chop_trees`, `mine_vein`, `explore_cavern`, `loot_container_or_crystal`
4. **Crafting, Building & Blueprints (6):** `craft_item`, `place_station_or_furniture`, `build_blueprint`, `equip_item`, `consume_buff_or_heal`, `manage_inventory_trash`
5. **Combat & Boss Control (6):** `summon_or_trigger_boss`, `engage_combat_burst`, `retreat_and_stabilize`, `deploy_combat_arena_tactics`, `query_threat_telemetry`, `record_tactical_checkpoint`

---

## Architecture Notes

- **Headless Surrogate vs. Optional tModLoader Bridge:** `terraria_bench/engine.py` provides a fast, byte-reproducible symbolic tile world so calibration, exploit regression, and RL rollouts run without Steam or `.NET`. An optional thin `tModLoader` socket forwarder lives in `mod/TerrariaRLBridge/` and forwards live-game state to the pure-Python port `8765` bridge (`terraria_bench/human_recorder.py`).
- **Leak Audit (`redact_seed=True`):** In StoreBench Appendix F, evaluation agents must not see world seeds or hidden SLA checkpoints. `web_console.py` and the evaluation harness pass `Episode(..., redact_seed=True)`, replacing `Seed: 7` in the status header with `World: locked`.
- **No Fabricated Model Scores:** Live model evaluation reads `GEMINI_API_KEY`, `NVIDIA_API_KEY`, or `OPENROUTER_API_KEY` and recorded ledgers in `artifacts/runs/`. Unrun models are explicitly shown as `not run` rather than fabricated.
