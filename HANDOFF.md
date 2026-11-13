# HANDOFF.md — TerrariaBench Foundation & 4-Teammate Split

**Updated:** 2026-11-14
**Branch:** main
**Repo:** TerrariaBench (`C:/Users/natha/dev/TerrariaBench`)

---

## 1. Completed Foundation (Frozen Shared Contract)

The core deterministic simulation environment, task suite, cryptographic provenance layer, composite grader, and 24-tool MCP surface are **100% implemented and verified**:

- [pyproject.toml](file:///C:/Users/natha/dev/TerrariaBench/pyproject.toml) — Package metadata, dependencies, and pytest configuration.
- [terraria_bench/__init__.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/__init__.py) — Unified public exports for the benchmark package.
- [terraria_bench/provenance.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/provenance.py) — Purpose-keyed SHA-256 RNG streams (`stream`), canonical JSON hashing, sealed artifact loaders (`seal`, `verify_seal`, `write_sealed`, `load_calibration`), and run manifests.
- [terraria_bench/tasks.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/tasks.py) — 8 evaluation task lenses (`EVAL_TASKS`) + 4 disjoint training task lenses (`TRAIN_TASKS`) across reported seeds `(7, 11, 13)` and held-out seeds `(17, 23)`, with Announced, Rumored, and Silent event channels and hidden SLA checkpoints.
- [terraria_bench/engine.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/engine.py) — Deterministic headless world (`World`), terrain/cave/ore/tree/trap generation, flood-fill housing verification, 3-tier boss arena & hellbridge checks, A* digging pathfinder, reactive combat bursts, window settlement, and Net Progression Value $\Phi(s)$ with $\rho = 0.15$ salvage discount.
- [terraria_bench/grader.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/grader.py) — StoreBench Eq. 1 composite scorer ($0.7$ Business / $0.2$ Reliability / $0.1$ Continuity), SLA evaluator (`sla_met`), 11-entry `EXPLOITS` catalogue, and $[-2, 1]$ clipped `dense_rewards` / `training_reward` channels for GRPO.
- [terraria_bench/tools.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/tools.py) — `Episode` session manager implementing all **24 MCP tools** across 5 families, the deterministic 1-line status header (`[Seed: ... | Win: ... | Ops: ...]`), stdlib Base64 PNG + $61 \times 35$ ASCII viewport rendering, windowed operation metering, and SHA-256 episode ledger sealing.

---

## 2. Parallel 4-Teammate Assignment Documents

Assign one document in `docs/` to each teammate (zero overlapping files):

1. **[Teammate 1 — Scripted Anchors, Reward-Exploit Catalogue, Calibration & Verification Suite](file:///C:/Users/natha/dev/TerrariaBench/docs/teammate_1_anchors_exploits_calibration.md)**
   - **Files:** `terraria_bench/policies.py`, `terraria_bench/exploits.py`, `terraria_bench/calibrate.py`, `tests/test_calibration_gates.py`, `tests/test_reward_exploits.py`, `tests/test_determinism_replay.py`
2. **[Teammate 2 — MCP Server & Frontier Agent Evaluation Harness (Runner, Gemini 4 + Image Blueprint + Decisions API, Nemotron 3.5 VL)](file:///C:/Users/natha/dev/TerrariaBench/docs/teammate_2_mcp_and_frontier_harness.md)**
   - **Files:** `terraria_bench/mcp_server.py`, `terraria_bench/harness/__init__.py`, `terraria_bench/harness/runner.py`, `terraria_bench/harness/gemini_agent.py`, `terraria_bench/harness/nemotron_agent.py`
3. **[Teammate 3 — Open-Weight Gemma 4 Agent & GRPO + LoRA Post-Training Pipeline](file:///C:/Users/natha/dev/TerrariaBench/docs/teammate_3_gemma_and_grpo_training.md)**
   - **Files:** `terraria_bench/harness/gemma_agent.py`, `terraria_bench/harness/train_grpo.py`, `tests/test_grpo_pipeline.py`
4. **[Teammate 4 — Real Game Bridge (`tModLoader` C# Mod), Human Telemetry Recorder, Operator Web Console, Leaderboard Dashboard & Packaging](file:///C:/Users/natha/dev/TerrariaBench/docs/teammate_4_bridge_recorder_ui_and_ops.md)**
   - **Files:** `mod/TerrariaRLBridge/{build.txt,description.txt,TerrariaRLBridge.cs}`, `terraria_bench/human_recorder.py`, `terraria_bench/web_console.py`, `dashboard/app.py`, `Makefile`, `Dockerfile`, `README.md`

---

## 3. Final Integration & Acceptance Commands (Once All 4 Tracks Merge)

```powershell
# 1. Calibrate all 6 anchors across seeds 7, 11, 13 (+ held-out 17, 23) and enforce Gates (i)-(iv)
python -m terraria_bench.calibrate

# 2. Run the full unit, determinism, calibration-gate, and reward-exploit test suite (< 15s)
uv run --with pytest pytest tests/ -v

# 3. Run the standard benchmark evaluation & print StoreBench Table 1 and Table 3
python -m terraria_bench.harness.runner --eval-standard
```
