# TerrariaBench

**TerrariaBench** is a deterministic, long-horizon agent evaluation and reinforcement learning benchmark over the Model Context Protocol (MCP), engineered to mirror the methodology, mathematical grading, anchor calibration, and anti-exploit architecture of AfterQuery's **StoreBench** (`arXiv:2610.10942v2`).

Everything runnable — the headless world simulator, 24-tool MCP surface, port `8765` bridge server, human operator console, live leaderboard dashboard, and evaluation/RL harnesses — is implemented in **pure Python (>= 3.11)** with zero Node.js/npm build steps or required `.NET` tooling.

---

## Reproduce the results

```bash
uv run python reporting/generate_report.py     # re-runs the six scripted anchors, rewrites EVAL_RESULTS.md
uv run --extra dev --extra agents pytest       # gates, exploit regressions, determinism, planner
harbor run -p harbor/tasks/terraria-first-night -a oracle      # reference run through Harbor
harbor run -p harbor/tasks/terraria-first-night -a opencode -m <provider/model> --env-file .env
```

`EVAL_RESULTS.md` holds only numbers that these commands regenerate or that come from sealed ledgers of live
runs in `artifacts/runs/`. A model with no saved ledger is listed as "not run".

## Two scores

| Score | Where it is the reward | What it measures |
| :--- | :--- | :--- |
| **Composite** (0..1) | Simulator and Harbor tasks (`reward.txt`) | Calibrated progress, reliability and attendance, see below. 0 = an absent player, 1 = the best scripted policy. |
| **Eye progress** (0..100%) | Real-Terraria sessions; also reported in every simulator `grade.json` | How far the run got toward killing the Eye of Cthulhu. |

Eye progress (`terraria_bench.grader.compute_eye_progress`):

| Part | Weight | Earned by |
| :--- | ---: | :--- |
| Health | 10% | Max HP gained over the starting value (full at +80, four Life Crystals) |
| Armor | 10% | Best armor actually worn, over the starting armor (full at 8 defense); potions do not count |
| Tasks on the way | 35% | Wood, Work Bench, shelter, weapons, arrows, stone/ore, furnace, anvil, metal gear, lenses, altar, arena, summoning item |
| Fighting the Eye | 25% | 5% for damaging it, plus 20% scaled by the lowest Eye health reached |
| Killing the Eye | to 100% | A kill is always exactly 100%; without one the score is capped at 80% |

All parts are high-water marks, so dropping, re-crafting or re-summoning cannot add credit. Doing nothing scores 0%
on every task, including the ones where the Eye appears by itself or the player starts with a strong kit
(`tests/test_reward_exploits.py`).

## Real Terraria: who decides what

The tModLoader mod (`mod/TerrariaRLBridge/`) executes *skills*: closed-loop programs (chop, mine, craft, hunt, build
an arena, fight, wall in for the night) that press the same keys a person would, 60 times a second. Python decides
which skill runs next, in two layers (`terraria_bench/eye_planner.py`):

1. **Long-term plan.** Before the first skill, the strategy model writes a `Strategy` for the run: wooden or metal
   bow, armor tier, how many Life Crystals to look for, how many arrows, whether to wall in on unarmed nights. It is
   stored with the run, shown as a numbered roadmap with the current step, and revised after a death. Each choice
   trades preparation time against the odds of winning a fight that must finish before dawn.
2. **Next step.** The planner lists the skills that are possible right now under that plan; the decision model picks
   one. `TERRARIA_AUTOPILOT=1` runs the scripted reference plan with no model calls: that is the reference run.

Every tick reports which tier produced what. A tier reads LIVE only when its model actually answered; the
structured `world_view` the mod sends in skill mode is not counted as vision.

## Smaller model trained on environment experience

`post_training/` builds 652 turn-by-turn examples from the expert policy on the four training tasks (each source
episode is replayed and kept only if it grades at least 0.95 with zero deaths), trains a rank-32 LoRA on
`google/gemma-4-E4B-it`, and evaluates base against tuned with the official composite. Results are in
`EVAL_RESULTS.md` (mean 0.249 base, 0.720 tuned over 8 cells, one episode each). The adapter itself is on a Modal
volume, not in this repository; `modal deploy post_training/modal_grpo_gemma.py` serves it, and setting
`GEMMA_TUNED_BASE_URL` to that endpoint lets `runner --play gemma_tuned --live` use it.

---

## Harbor tasks (OpenCode harness)

The eight evaluation scenarios ship as [Harbor](https://github.com/laude-institute/harbor) tasks in `harbor/tasks/`, each with `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/` and `solution/`.

```bash
harbor run -p harbor/tasks/terraria-first-night -a opencode -m gemini/gemini-4-carbon \
    --api-key-name GEMINI_API_KEY --env-file .env
harbor run -p harbor/tasks/terraria-first-night -a oracle   # reference solution
harbor run -p harbor/tasks -a opencode -m <provider/model>  # all eight scenarios
```

`.env` holds the model provider's API key (for Google models, `GEMINI_API_KEY`). Harbor's provider prefix for Google is `gemini/`, and `--api-key-name` is needed for model ids its registry does not know yet. Docker is required: each task is a two-container environment.

### Verda Inference models

The live Python harness also supports Verda's OpenAI-compatible endpoint. Put `VERDA_INFERENCE_API_KEY` in `.env` or the process environment; the harness defaults to `https://inference.verda.com/v1`. The model IDs are `deepseek-4-1-flash-1017` and `kimi-k3-1-pro`.

```powershell
# Run one live episode with either model
python -m terraria_bench.harness.runner --play verda_deepseek --live --task eye-of-cthulhu
python -m terraria_bench.harness.runner --play verda_kimi --live --task eye-of-cthulhu

# Run both models across the standard evaluation set and add their saved scores to the leaderboard
python -m terraria_bench.harness.runner --eval-standard --live
```

Standard evaluation saves each model under `artifacts/runs/verda_deepseek/` and `artifacts/runs/verda_kimi/`. Override model IDs with `TERRARIA_VERDA_DEEPSEEK_MODEL` or `TERRARIA_VERDA_KIMI_MODEL`, and the endpoint with `VERDA_INFERENCE_BASE_URL`.

For Harbor/OpenCode runs, merge the [Verda OpenCode provider config](docs/opencode-verda.jsonc) into `~/.config/opencode/opencode.jsonc`, then select either `verda/deepseek-4-1-flash-1017` or `verda/kimi-k3-1-pro`:

```powershell
harbor run -p harbor/tasks/terraria-eye-of-cthulhu -a opencode -m verda/deepseek-4-1-flash-1017 --api-key-name VERDA_INFERENCE_API_KEY --env-file .env
harbor run -p harbor/tasks/terraria-eye-of-cthulhu -a opencode -m verda/kimi-k3-1-pro --api-key-name VERDA_INFERENCE_API_KEY --env-file .env
```

**What the agent has to do.** Play a Terraria-like survival world to the end of a scenario: survive the first night, dig to the caverns, hold off a goblin siege or a blood moon, build a boss arena, fight the Eye of Cthulhu or the Wall of Flesh, or do all of it in one 30-window marathon. The agent acts only through 24 game tools (scan, navigate, chop, mine, craft, build, equip, fight, retreat, and so on), offered both as an MCP server and as a `terraria` shell command. Time is metered in operations rather than wall-clock: each 12-hour day or night window allows a fixed number of tool calls, and when they run out the rest of the window plays out with the player wherever it was left standing.

**Why it is hard.**

- *Long horizon, one attempt.* A scenario is 80 to 360 metered calls against a single world that never resets. A night spent outside, or a death that drops half the player's materials, compounds through every later window.
- *Scarce actions.* Looking costs the same as doing. An agent that inspects everything runs out of operations before it has built anything.
- *Deadlines it is not told.* Each scenario has unlisted deadlines (shelter before the first dusk, armour before the siege, an arena before the boss). The agent has to infer what will be needed from the brief and the world, and prepare ahead of time.
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
