# TerrariaBench

**A survival-progression RL environment and live 60 Hz benchmark where frontier models know how to fight, yet die from trusting stale world state and losing the race against operation budgets.**

TerrariaBench implements the OpenAI / Terraria progression domain across **14 tasks x 5 seeds (70 calibrated cells)** in a deterministic 24-tool MCP simulator (`terraria_bench/`), packaged into isolated Docker verifier containers (`harbor/`), and paired with a **real 60 Hz `tModLoader` bridge** (`mod/TerrariaRLBridge/`, `live_play.py`, `real_eval.py`) that plays the same Eye of Cthulhu progression objective through pixels, keyboard, and mouse.

> **Read the full evidence trail:**
> - [`RESULTS.md`](RESULTS.md) — Deterministic baselines, Harbor container runs, **42 replay-verified model transcripts**, death table, and **12 closed exploit checks** (recomputed offline by `python -m evaluation.judge`).
> - [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) — Transcript autopsy of the **23/42 fatal runs** (and why 14/23 died immediately after declaring themselves safe).
> - [`EVAL_RESULTS.md`](EVAL_RESULTS.md) — Full 8-task standard simulator leaderboard, 3 failure areas, `google/gemma-4-E4B-it` BC & wiki-grounded GRPO post-training, and Real Terraria scoring (recomputed offline by `python reporting/generate_report.py`).
> - [`PITCH.md`](PITCH.md) & [`docs/DETAILS.md`](docs/DETAILS.md) — Challenge thesis, environment mechanics, 7-component potential function $\Phi$, and scoring equations.
> - [`paper/main.tex`](paper/main.tex) & [`TerrariaBench.pptx`](TerrariaBench.pptx) — Complete research paper and pitch deck.

---

## Why Terraria? Research & Industry Relevance Beyond Gaming

Most existing agent benchmarks test purely digital workflows (chat, single-file code edits, or resettable browser forms). Real-world physical and operational autonomy — **autonomous driving, embodied robotics, and long-running industrial workflows** — requires an agent to coordinate across multiple timescales inside a persistent spatial environment where mistakes are irreversible.

TerrariaBench serves as a deterministic, reproducible testbed for that transition, and our experiments establish two core takeaways:

1. **Hierarchical supervisor + reflex architectures work:** Pairing a high-level **frontier supervisor** (managing long- and short-term strategy and prerequisite plans) with low-latency **decision and reflex layers** (handling sub-second reactions, aiming, movement, and immediate hazard avoidance in `live_play.py` and `eye_planner.py`) dramatically improves responsiveness and efficiency at 60 Hz.
2. **Three failure modes persist even in multimodal, multi-layered agents:**
   - **Long-term forethought & plan commitment:** Models fail to hold safety margins (like cave return budgets in 7/23 deaths) or commit cleanly along multi-step preparation trees before deadlines hit.
   - **Short-term "smart-ish" tactical reactions:** Decision layers default to greedy engagement instead of smart disengagement — fighting an unwinnable boss (7/23 deaths) or pressing into hazards rather than retreating to recover.
   - **2D spatial blind spots in multimodal models:** Even with visual images, ASCII grids, and explicit warnings, agents fail to recognize how their own physical world modifications invalidate safety invariants (9/23 deaths from building an arena that clipped their own shelter wall without re-checking).

By isolating **planner-to-reflex handoffs**, **context-aware abort/retreat decisions**, and **post-manipulation spatial re-verification** in a fast seeded simulator + live 60 Hz bridge, TerrariaBench directly guides how we architect and post-train models that act in the physical world rather than only digital text interfaces.

---

## One-Command Judge Verification (No API Keys Needed)

Every claim in this repository can be reproduced offline on any laptop from sealed ledgers and engine replay:

```powershell
# 1. Install dependencies (Python >= 3.11, uv)
uv sync --extra dev --extra agents

# 2. Run the full test suite (engine, calibration gates, 12 exploits, Harbor, UI, post-training)
uv run --extra dev --extra agents pytest -q

# 3. Replay all 42 transcripts, run 12 exploits + deterministic baselines, and verify RESULTS.md
uv run python -m evaluation.judge

# 4. Run the AfterQuery baseline suite (Reference vs. Do-Nothing vs. Stop-Early) on any task
uv run python scripts/run_judging_baselines.py --task eye-of-cthulhu --seed 7

# 5. Recompute the standard simulator leaderboard, Gemma BC/GRPO tables, and Real Terraria summary
uv run python reporting/generate_report.py

# 6. Re-score the committed 60 Hz real-Terraria recordings against referee telemetry
uv run python -m terraria_bench.real_eval --all --save
```

### Standard Harbor Container Evaluation

The headline Hardcore tasks (`terraria-eye-of-cthulhu-hardcore`, `terraria-first-night-hardcore`) and all 8 standard eval tasks are packaged in `harbor/tasks/` along with matching `-stop-early` baselines in `harbor/baselines/`:

```bash
# Oracle reference (0.985 on Hardcore Eye of Cthulhu, 1.000 on Hardcore First Night)
harbor run -p harbor/tasks/terraria-eye-of-cthulhu-hardcore -a oracle

# Do-nothing baseline (0.000) and stop-early baseline (0.174 on Hardcore Eye quarter-run)
harbor run -p harbor/tasks/terraria-eye-of-cthulhu-hardcore -a nop
harbor run -p harbor/baselines/terraria-eye-of-cthulhu-hardcore-stop-early -a oracle

# Any frontier model via OpenCode over MCP
harbor run -p harbor/tasks/terraria-eye-of-cthulhu-hardcore -a opencode -m <provider/model>
```

---

## Summary of Findings (Break-the-Agent Rubric)

### 1. The Gap

**Hardcore & minimal-harness headline (`evaluation/run_model.py`, seed 7, `RESULTS.md`):** Every prompt discloses the full scoring formula, every checkpoint deadline, and the one-life Hardcore rule.

| Player | `first-night-hardcore` | `eye-of-cthulhu-hardcore` | `first-night` | `eye-of-cthulhu` |
|:---|---:|---:|---:|---:|
| **Reference (`smart-triage`)** | **1.000** | **0.985** | **1.000** | **0.950** |
| Pass threshold | 0.860 | 0.659 | 0.950 | 0.696 |
| **Kimi K3.1 Pro** | 0.827 (3 runs) | 0.759 (0.27–1.00, 9 runs) | 0.961 (3 runs) | 0.989 (3 runs) |
| **Gemma 4 31B** | 0.648 (3 runs) | 0.403 (0.32–0.51, 3 runs) | 0.716 (3 runs) | 0.711 (3 runs) |
| **DeepSeek V4.1 Flash** | 0.660 (3 runs) | 0.284 (0.11–0.41, 3 runs) | 0.695 (3 runs) | 0.908 (3 runs) |
| Do nothing | 0.000 | 0.000 | 0.000 | 0.000 |

On **Harbor + OpenCode over MCP** (`results/harbor_agent/`), OpenCode scaffolding raises DeepSeek V4.1 Flash on `terraria-eye-of-cthulhu-hardcore` to **0.840** mean over 3 trials (`0.523`, `0.996`, `1.000`), while Kimi K3.1 Pro scored **1.000** on the 1 trial that started (2 trials timed out installing OpenCode locally).

**8-task simulator leaderboard (`EVAL_RESULTS.md`):** Across all sealed runs mapped to the standard 8-task × 3-seed evaluation grid:

| Policy / model | Cells | Runs | Composite | Pass rate | Eye progress |
|:---|---:|---:|---:|---:|---:|
| **Reference (`smart-triage`)** | 24 | 24 | **0.982** | **100%** | 57% |
| `rule-based` anchor | 24 | 24 | 0.505 | 0% | 32% |
| **Verda Kimi K3.1 Pro** *(normal tasks)* | 2 | 6 | 0.975 | 67% | 70% |
| **Verda DeepSeek V4.1 Flash** *(normal tasks)* | 2 | 6 | 0.801 | 50% | 66% |
| **Gemini 4 / 2.5** *(multi-task, two-tier)* | 10 | 12 | 0.652 | 47% | 45% |
| **Human expert pilot** | 1 | 1 | 0.489 | 0% | 19% |
| **Nemotron 3.5 VL** | 10 | 12 | 0.144 | 0% | 15% |
| **Gemma Base** | 4 | 4 | 0.046 | 0% | 12% |
| `do-nothing` / `stop-early` (`absentee`) / `exploiter` | 24 | 24 | 0.000 | 0% | 0% |

### 2. Meaningful Difficulty

Why does Kimi K3.1 Pro score **1.000** on 6/9 Hardcore Eye of Cthulhu runs (matching or beating our scripted reference) and **0.27–0.28** on the other 3/9? Not missing information — `tests/test_evaluation.py` asserts every graded checkpoint and scoring term is stated in the prompt, and only 1%–22% of tool calls are rejected (almost all for missing crafting ingredients/stations). Instead, `FAILURE_ANALYSIS.md` traces **23 fatal runs** out of 42 transcripts to three spatial/planning decisions:

1. **Trusting stale world state (9 / 23 first deaths; all 3 of Kimi's Hardcore deaths):** In Window 1 the agent builds a shelter (`housing.valid = true`). Later it builds a 25-tile boss arena beside home whose footprint clips the shelter wall (`arena` call returns the modified `x_range`). Without spending 1 operation on `inspect_housing_and_arena`, the model ends the night `"inside the validated shelter"` and dies in its sleep. Seven of Kimi's 9 runs broke their own house; all 4 that re-checked and repaired scored **1.000**, while all 3 that didn't died in Window 4.
2. **Starting a boss fight the operation budget cannot finish (7 / 23 deaths):** Summoning the 2,800-HP Eye late in a window when the weapon's visible per-burst damage guarantees the boss cannot die before settlement (where an active boss at window end kills the player).
3. **No return-home margin underground (7 / 23 deaths):** Spending the final cave operations while `navigate_to` distance exceeds remaining ops. Even when Hardcore induces caution without death (`first-night-hardcore`), scores drop (Kimi `0.96 -> 0.83`, DeepSeek `0.70 -> 0.66`) as models stop at basic gear instead of upgrading to Silver.

In **14 of the 23 fatal runs**, the agent's own reasoning on its final call declared that it was safe.

### 3. Reward Integrity

The composite score maps each run to $[0, 1]$ via **70% calibrated potential $\Phi$** + **20% reliability** (uptime $\times$ SLA checkpoint compliance) + **10% continuity** (fraction of the task's full window count played), anchored at `absentee = 0.000` and calibrated against `smart-triage` across **70 cells (70/70 calibration gates pass)**.
- **Zero reward for inaction or quitting:** `do-nothing` and `absentee` score `0.000` on every task; partial stop-early slices (25%, 50%, 75%, plus Harbor `-stop-early` tasks) strictly underperform the full reference on every cell.
- **12 exploit classes tested and closed** (`terraria_bench/exploits.py` -> `RESULTS.md`): attendance farming, continuity for quitters, dirt-box camping, place/break mining loops, slime grinding, boss resummon HP scumming, craft/trash cycling, suicide respawn teleports, last-window boss dumping, partial enclosure spoofing, duplicate chest/crystal looting, and healing potion stalling.

### 4. Stretch Goal: Post-Training a 4B Model (`post_training/`)

We evaluated **`google/gemma-4-E4B-it`** with rank-32 LoRA adapters on Modal across 8 cells (4 training tasks seen/unseen seeds + 2 held-out evaluation tasks `first-night:7` and `cavern-expedition:7`):

| Method | Base mean | Tuned mean | Base -> Tuned deaths | Notes |
|:---|---:|---:|:---:|:---|
| **Closed-loop Behavior Cloning** (652 expert steps, `bc_eval.json`) | 0.249 | **0.720** | 13 -> 3 | Passes `2/8` cells (`woodland-scramble:17` `0.976`, `ore-rush:17` `0.932`); transfers to unseen `first-night:7` (`0.312 -> 0.620`) |
| **Open-loop GRPO from base** (no BC init) | 0.187 | 0.078 | 14 -> 22 | Degrades closed-loop survival without behavioral grounding |
| **Wiki-grounded BC-init KL GRPO** ($\beta = 0.05$, 12 steps, `grpo_eval.json`) | 0.187 | **0.721** | 14 -> 2 | Improves **8/8** cells; beats BC on `shelter-crunch:17` (**0.843** vs `0.609`) & held-out `cavern-expedition:7` (**0.688** vs `0.324`); `0/8` pass calibrated gates (preserving headroom) |

---

## Demo Track B: Live 60 Hz Real Terraria (`mod/TerrariaRLBridge/`)

Our second live demo bridges the same 13-stage Eye of Cthulhu progression rubric directly into unscripted **real Terraria (`tModLoader`)** via `mod/TerrariaRLBridge/` (`TerrariaRLBridge.tmod`), keyboard/mouse OS control (`real_game.py`), a 3-tier vision/action architecture (`live_play.py`), and referee-checked offline trajectory scoring (`terraria_bench/real_eval.py`):

| Track / Run | Agent Architecture | Frames | Eye Progress | Stages | First Unmet Stage | Deaths | Referee Check |
|:---|:---|---:|---:|---:|:---|---:|:---|
| **Simulator** (MCP symbolic) | Gemini two-tier (4 runs) | - | **90%** (3/4 kills) | - | - | 4 | sealed |
| **`real_terraria_run1.json`** | Gemini three-tier (pixels + KB/mouse) | 300 | **20%** | **4 / 13** | `ammunition_ready` | 2 | consistent |
| **`real_terraria_run2.json`** | Gemini three-tier (pixels + KB/mouse) | 300 | **22%** | **6 / 13** | `ammunition_ready` | 0 | consistent |

Across `run1` -> `run2`, adding house building and crafting milestones took the real-game agent from 4/13 stages (`wood_gathered`, `workbench`, `shelter`, `weapons`) to 6/13 stages (adding `FURNACE_SMELTING` and `ARMOR_EQUIPPED`, 0 deaths), while still stalling at `ammunition_ready` — demonstrating a stark **90% -> 21% sim-to-real gap** even with identical progression goals.

> Note: A live recording against Terraria requires running `tModLoader` with `mod/TerrariaRLBridge/` locally (`uv run python live_play.py --goal ... --agent three-tier`). Pre-recorded referee-verified traces are committed in `artifacts/recordings/` and scored offline by `real_eval.py`.

---

## Repository Layout & Additional Commands

```text
terraria_bench/
  engine.py            # Deterministic 2D tile world, housing flood-fill, day/night windows, bosses
  tasks.py             # 14 tasks (2 hardcore + 8 eval + 4 train), full disclosed checkpoints & scoring
  grader.py            # 7-component potential Φ, SLA/reliability/continuity, Eye progress %
  tools.py             # 24 MCP tools + sealed ledger recording
  policies.py          # Scripted anchors: smart-triage (_paced), rule-based, blast-list, absentee, do-nothing
  exploits.py          # 12 exploit harnesses verifying zero score loopholes
  calibrate.py         # 70-cell (14x5) calibration and 4 sanity gates -> artifacts/calibration.json
  harness/             # Model runner, minimal/two-tier Gemini/Nemotron/Verda drivers
  web_console.py       # Interactive browser & CLI human play console
  human_recorder.py    # Replay-verified human trajectory recorder (--record-demo, --verify-all)
  real_eval.py         # Offline referee scorer for live 60 Hz tModLoader recordings
evaluation/
  judge.py             # Offline transcript replayer, Harbor reader, exploit runner -> RESULTS.md
  run_model.py         # Minimal single-call JSON tool loop for model evaluation
harbor/                # Containerized Harbor task definitions (10 tasks + 10 stop-early baselines)
post_training/         # Modal A100 BC + wiki-grounded GRPO training for google/gemma-4-E4B-it
mod/TerrariaRLBridge/  # C# tModLoader mod exporting 60 Hz JSON state + Skills.cs macros
reporting/             # Report generator -> EVAL_RESULTS.md + artifacts/runs/eval_standard.json
results/runs/          # 42 sealed model transcripts (DeepSeek V4.1 Flash, Kimi K3.1 Pro, Gemma 4 31B)
artifacts/             # Calibration, post-training JSONs, uncapped model ledgers, real_terraria recordings
paper/                 # Research paper (LaTeX source in paper/main.tex)
TerrariaBench.pptx     # Presentation pitch deck
```

### Interactive & Live Run Commands

```powershell
# Start MCP server over stdio
uv run python -m terraria_bench.mcp_server

# Play as a human in the browser UI or CLI (seals ledger into artifacts/runs/human/)
uv run python -m terraria_bench.web_console --task eye-of-cthulhu --seed 7

# Evaluate a model on the minimal loop (writes a sealed replayable transcript to results/runs/)
uv run python -m evaluation.run_model --task eye-of-cthulhu-hardcore --seed 7 --model kimi-k3.1-pro

# Re-run standard harness live evaluations (requires GEMINI_API_KEY / VERDA_INFERENCE_KEY / HF_TOKEN)
uv run python -m terraria_bench.harness.runner --eval-standard --live
```
