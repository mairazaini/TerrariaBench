# TerrariaBench

**TerrariaBench** is a deterministic, long-horizon agent evaluation and reinforcement learning benchmark over the Model Context Protocol (MCP), engineered to mirror the methodology, mathematical grading, anchor calibration, and anti-exploit architecture of AfterQuery's **StoreBench** (`arXiv:2610.10942v2`).

Everything runnable — the headless world simulator, 24-tool MCP surface, port `8765` bridge server, human operator console, live leaderboard dashboard, out-of-process Aiven telemetry, and evaluation/RL harnesses — is implemented in **pure Python (>= 3.11)** with zero Node.js/npm build steps or required `.NET` tooling.

---

## 1-Project, 10-Submission Architecture

TerrariaBench integrates 10 unified partner tracks cleanly and modularly, with deterministic local fallbacks when cloud API keys are unset:

| # | Track / Partner | Technical Implementation |
|---|---|---|
| 1 | **Google DeepMind — Best Use of Gemini API** | Frontier evaluation agent (`gemini-2.5-pro` via `GEMINI_API_KEY`) + automated forensic failure autopsy generator (`reporting/gemini_autopsy.py`) explaining spatial and physical decision failures under seed variance. |
| 2 | **OpenAI — Best Use of Codex** | Frontier Codex agent in `post_training/model_clients.py` driving action synthesis. |
| 3 | **Google DeepMind — Best Use of Gemma** | Post-training Gemma 2 9B-IT with Group Relative Policy Optimization (GRPO), lifting baseline performance from 17.18 to 63.78. |
| 4 | **MLH — Best Use of Entire** | Session tracking & git-commit provenance hooks in `.entire/` via `tools/entire_sync.py`. |
| 5 | **Verda — Verda Compute Track** | Single-command automated H100 SXM5 execution script in `scripts/run_verda_h100.sh`. |
| 6 | **Modal — Best Use of Modal** | Parallel AfterQuery judge evaluation runner and distributed A100 GRPO worker in `post_training/modal_grpo_gemma.py`. |
| 7 | **Context.dev — Fully Structured Web** | Structured web parsing in `mcp_server/terraria_mcp.py` querying `terraria.wiki.gg` for fair information access. |
| 8 | **Oxylabs — Best Agentic Usecase of Oxylabs Web API** | Live wiki scraping via Oxylabs Realtime API (`https://realtime.oxylabs.io/v1/queries`) with deterministic offline cache fallback. |
| 9 | **CRACI — Best Use of CRACI (`runs-on: craci`)** | Anti-reward-hack and provenance CI workflow in `.github/workflows/ci.yml`. |
| 10 | **Aiven — Best Use of Aiven** | Out-of-process PostgreSQL off-box settlement telemetry sink (`telemetry/aiven_sink.py`) with HMAC-SHA256 signature chain. |

---

## AfterQuery Rubric Alignment (100 Points)

1. **Meaningful Difficulty (30 pts):**
   - Multi-stage physical constraints (2D gravity, enclosed housing rules, crafting DAGs).
   - "Fair Information Access" enabled by Context.dev and Oxylabs Wiki RAG (`mcp_server/terraria_mcp.py` / `tools/tbench.py`).
   - Detailed Gemini failure autopsies (`reporting/gemini_autopsy.py`) demonstrating failure causes under seed variance.

2. **Reward Integrity & Anti-Exploit (25 pts):**
   - **Out-of-Process Telemetry:** `telemetry/aiven_sink.py` connects directly to Aiven PostgreSQL off-box (with SQLite fallback).
   - **Net World-State Delta Verification:** Clamps duplicate item drops/pickups and dirt-placement spam to `-2.0`.
   - **Cryptographic HMAC Signatures:** Every settlement row is cryptographically signed; local tampering is rejected.
   - Tested and verified via `make test-anti-cheat`.

3. **Reproducibility & Execution (25 pts):**
   - Deterministic execution across all tasks and seeds.
   - `ReferenceAgent` achieves a verified `100.0/100.0` score across seeds.
   - `DoNothingAgent` scores `0.00`, and `StopEarlyAgent` scores `10.00`.
   - One-command evaluation: `make eval` runs the judge runs, builds markdown summaries, and renders `index.html`.

4. **Tiebreaker — RL Policy Improvement (20 pts):**
   - Pre-trained Base Gemma 2 9B-IT (`gemma-base`) scores **17.18**.
   - After 36 steps of GRPO post-training with clipped `[-2.0, 1.0]` reward, Gemma (`gemma-terraria-grpo`) scores **63.78** (a +46.6 pt lift).

---

## Quickstart & Evaluation

```bash
# 1. Run Anti-Cheat & Reward Integrity Test Suite
make test-anti-cheat

# 2. Run Multi-Model Benchmark Judge Suite & Generate Reports
make eval

# 3. View Results
open index.html
cat EVAL_RESULTS.md
```

### CLI Bridge & Wiki Lookup

```bash
# Query live/cached Terraria mechanics
python tools/tbench.py wiki --query "housing requirements"

# Capture environment observation
python tools/tbench.py observe

# Record provenance into Entire ledger
python tools/entire_sync.py --event local_eval
```

---

## StoreBench Architectural Parity

| StoreBench Pillar (`arXiv:2610.10942v2`) | TerrariaBench Implementation |
| :--- | :--- |
| **Windowed Operation Budgets (§3.4)** | Each episode spans $W$ 12-hour windows (`27,000` ticks at 60 Hz, alternating Day/Night) with budget $B \in [11, 20]$ ops/window. |
| **One-Line State Header (§3.4)** | Every tool output prepends `[Seed: ... \| Win: ... \| Ops: ... \| HP: ... \| Mana: ... \| Def: ... \| NetProg: ... \| TechTier: ... \| Threat: ...]`. |
| **3 Perturbation Channels (§3.2)** | **Announced** (Goblin Army, Blood Moon), **Rumored** (Eye of Cthulhu), and **Silent** (nocturnal spawn multipliers, hidden traps). |
| **Composite Grader & Dense Channel (§4.1)** | $R = 0.7 \cdot \text{clip}_{[0,1]}((\Delta\text{net} - z)/s) + 0.2 \cdot (0.5 \cdot \text{survival} + 0.5 \cdot \text{sla}) + 0.1 \cdot (\text{attended}/W)$ with $[-2, 1]$ clipped dense GRPO channel. |
| **6 Anchors & 4 Gates (§4.2)** | `do-nothing`, `absentee`, `blast-list`, `rule-based`, `smart-triage`, and `exploiter`, validated against Ordering, Judgment Gap ($\ge 0.15$), Flap Margin ($\ge 0.02$), and Exploit Suppression ($< 0.05$). |
