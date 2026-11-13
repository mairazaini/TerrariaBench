# Teammate 3 — Open-Weight Gemma 4 Agent & GRPO + LoRA Post-Training Pipeline

## 1. Mission & Scope

You own **Track 3 (Open-Weight Gemma 4 Policy & GRPO + LoRA Post-Training Pipeline)** of **TerrariaBench**, mirroring StoreBench §7 and Appendices C & K.
The deterministic environment ([engine.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/engine.py)), task splits ([tasks.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/tasks.py)), composite and dense reward channels ([grader.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/grader.py)), provenance ([provenance.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/provenance.py)), and 24-tool `Episode` surface ([tools.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/tools.py)) are already implemented and frozen.

Your responsibility is to build the **Gemma 4 baseline & LoRA checkpoint evaluator (`gemma_agent.py`)** and the **GRPO + LoRA (rank 32, $\alpha=32$) post-training pipeline (`train_grpo.py`)** over the 4 disjoint training tasks (`woodland-scramble`, `ore-rush`, `slime-rain`, `shelter-crunch`) using the $[-2, 1]$ clipped dense/terminal training reward channel and behavior-cloning warm-start from human/expert trajectories.

### Files You Own (No Overlap With Teammates 1, 2, 4)
| File Path | Purpose |
| :--- | :--- |
| `terraria_bench/harness/gemma_agent.py` | Google Gemma 4 (`google/gemma-3-27b-it` / configurable `TERRARIA_GEMMA_MODEL`) baseline and LoRA adapter evaluator |
| `terraria_bench/harness/train_grpo.py` | StoreBench §7 / Appendix K GRPO + LoRA (rank 32, $\alpha=32$) training pipeline + BC warm-start + Modal / Verda GPU entrypoint + local CPU/surrogate smoke-test mode |
| `tests/test_grpo_pipeline.py` *(optional)* | Fast unit test verifying group advantage normalization, $[-2, 1]$ reward clipping, and telescoping dense rewards |

---

## 2. Mathematical & Architectural Foundations (StoreBench §7 & Appendices C, K)

```mermaid
flowchart TD
    TR["TRAIN_TASKS (4 Disjoint Lenses)\nwoodland-scramble, ore-rush, slime-rain, shelter-crunch"] --> RO["Group Rollout Generator (G = 6 rollouts / prompt)"]
    BC["Human / Expert Trajectories (.jsonl)"] --> WS["Behavior Cloning Warm-Start"]
    WS --> PI["Gemma Policy + LoRA (r=32, alpha=32)"]
    PI --> RO
    RO -->|Episode.finalize()| LED["Sealed Episode Ledgers"]
    LED --> DR["grader.training_reward / dense_rewards\nClipped to [-2, 1] with rho = 0.15 salvage"]
    DR --> ADV["Group-Normalized Advantages\nA_i = (R_i - mean(R)) / (std(R) + 1e-6)"]
    ADV --> LOSS["Clipped Token-Mean GRPO Loss (no KL penalty)"]
    LOSS -->|36 Steps (1,080 episodes)| CKPT["artifacts/checkpoints/gemma_grpo_lora/"]
    CKPT --> EVAL["Evaluate on 8 Held-Out EVAL_TASKS (Table 3)"]
```

### Why the $[-2, 1]$ Training Channel Matters (StoreBench §7)
In `grader.py`, the official composite score clips business growth to $[0, 1]$:
$$R = 0.7 \cdot \text{clip}_{[0,1]}\left(\frac{\Delta\text{net} - z}{s}\right) + 0.2 \cdot \text{reliability} + 0.1 \cdot \text{continuity}$$
For a base open-weight model that initially dies or fails to out-progress `absentee` ($\Delta\text{net} < z$), $\text{clip}_{[0,1]}$ maps every rollout in a group to `0.0`, producing zero variance within the group and **zero gradient signal**.

Therefore, [grader.py](file:///C:/Users/natha/dev/TerrariaBench/terraria_bench/grader.py#L133-L146) already provides two training-specific functions that you must use during GRPO:
- `training_reward(ledger, cell) -> float`:
  $$\tilde{R} = \text{clip}_{[-2, 1]}\left(\frac{\Phi(s_W) - \Phi(s_0) - z}{s}\right)$$
- `dense_rewards(ledger, cell) -> list[float]`:
  $$r_t = \text{clip}_{[-2, 1]}\left(\frac{\Phi(s_t) - \Phi(s_{t-1})}{s}\right), \quad t = 1, \dots, W$$
  Because raw inventory in $\Phi(s_t)$ is valued at the salvage discount $\rho = 0.15$ (`engine.RHO`), the unclipped per-window terms telescope **exactly** to $(\Phi(s_W) - \Phi(s_0)) / s$:
  $$\sum_{t=1}^{W} \frac{\Phi(s_t) - \Phi(s_{t-1})}{s} = \frac{\Phi(s_W) - \Phi(s_0)}{s}$$

---

## 3. Detailed Module Specifications

### 3.1 `terraria_bench/harness/gemma_agent.py` — Gemma 4 Baseline & Tuned Agent

1. **Model & Checkpoint Loading**:
   - Configurable model ID via `os.environ.get("TERRARIA_GEMMA_MODEL", "google/gemma-3-27b-it")` (supports `gemma-4` aliases, local HuggingFace/vLLM endpoints via `GEMMA_BASE_URL`, or local `transformers` + `peft` LoRA adapter path `checkpoint_path`).
   - Supports two inference backends:
     - **OpenAI-compatible / vLLM / Ollama / HuggingFace TGI client** when `GEMMA_BASE_URL` or `OPENROUTER_API_KEY` is set.
     - **Local HuggingFace `transformers` + `peft.PeftModel`** when running on a GPU node (Modal / Verda) with a local LoRA checkpoint directory.
     - **Deterministic Policy Adapter Mode** (for fast local CPU testing when GPU/API weights are not mounted): loads learned action-preference weights from `artifacts/checkpoints/gemma_grpo_lora/policy_weights.json` so `gemma_agent.py` can be unit-tested end-to-end on any laptop.
2. **Action Formatting & Turn Discipline**:
   - Formats the 24 MCP tools in `TOOL_SCHEMAS` into a compact system prompt.
   - Parses `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` or native tool-call JSON from model completions and dispatches to `Episode.step(name, **arguments)`.
   - Respects the **150-turn cap** and **~65k-token context compaction** (keeping the last 3 turns verbatim) from `runner.py`.

---

### 3.2 `terraria_bench/harness/train_grpo.py` — GRPO + LoRA Post-Training Pipeline

Must implement the exact StoreBench §7 / Appendix K training recipe:
1. **Training Split Isolation (Strict Anti-Leakage Check)**:
   - Assert at startup that training uses **only** `TRAIN_TASKS = ("woodland-scramble", "ore-rush", "slime-rain", "shelter-crunch")` and never touches `EVAL_TASKS`.
   - Assert that evaluation uses `EVAL_TASKS` and grades every episode with the **official** `grader.grade(ledger, cell)` composite score (never reporting the $[-2, 1]$ training reward as the benchmark score).
2. **Hyperparameters (StoreBench Appendix K)**:
   - `GROUP_SIZE = 6` rollouts per task-seed prompt
   - `PROMPTS_PER_STEP = 5` task-seed pairs sampled from `TRAIN_TASKS × REPORTED_SEEDS` (`4 × 3 = 12` training cells)
   - `TOTAL_STEPS = 36` (`36 × 5 × 6 = 1,080` training episodes)
   - `LORA_RANK = 32`, `LORA_ALPHA = 32`, target modules `("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")`
   - Learning rate schedule: `5e-5` default, raised to `1e-4` for steps `16..26`
   - Sampling temperature `1.0`, `MAX_TURNS = 150`, context compaction at `65_000` tokens keeping last 3 turns verbatim
   - **No KL penalty** (`beta_kl = 0.0`), clipped token-mean surrogate policy objective (`eps_clip = 0.2`)
3. **Group Advantage Computation (`compute_group_advantages`)**:
   ```python
   def compute_group_advantages(rewards: list[float]) -> list[float]:
       """Normalize rollout rewards within a group of G=6 rollouts."""
       mean_r = sum(rewards) / len(rewards)
       var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
       std_r = var_r ** 0.5
       if std_r < 1e-6:
           return [0.0 for _ in rewards]
       return [(r - mean_r) / std_r for r in rewards]
   ```
   Also support combining terminal `training_reward(ledger, cell)` with discounted per-window `dense_rewards(ledger, cell)` (`gamma = 0.98`).
4. **Behavior Cloning (BC) Warm-Start (`bc_warmstart`)**:
   - Reads recorded expert/human `.jsonl` trajectories from `artifacts/human_trajectories/` (or generates warm-start demonstrations using `smart-triage` on `TRAIN_TASKS` only when no human trajectories are present) and runs supervised cross-entropy warm-start steps before GRPO step 1.
5. **Modal Dojo & Verda GPU Integration + Local Dry-Run CLI**:
   - Provide `python -m terraria_bench.harness.train_grpo --dry-run` (runs 2 fast GRPO steps over the surrogate environment on CPU using the tabular/linear action-head fallback, verifies telescoping dense rewards and non-zero group advantages, and writes a valid checkpoint manifest to `artifacts/checkpoints/gemma_grpo_lora/`).
   - Provide `--modal` / `--verda` flags (or Modal `@app.function(gpu="A100-80GB:2")` decorator guarded when `modal` is installed) for full 27B LoRA training on cloud GPUs.

---

## 4. Verification Checklist Before Merging

Run:
```powershell
python -m terraria_bench.harness.train_grpo --dry-run
```
- [ ] Confirm `TRAIN_TASKS` and `EVAL_TASKS` disjointness check passes.
- [ ] Confirm `sum(unclipped_dense_rewards)` equals `(phi_final - phi0) / s` to within `1e-6` on all 4 training tasks.
- [ ] Confirm `--dry-run` completes in < 10 seconds, verifies group advantage calculation, and saves a checkpoint summary to `artifacts/checkpoints/gemma_grpo_lora/`.
