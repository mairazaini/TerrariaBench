"""Tests for StoreBench §7 Gemma 4 Agent and GRPO + LoRA Post-Training Pipeline.

Verifies:
1. StoreBench §7 Config & Task Invariants (LoRA r=32, alpha=32, G=6, steps=36, kl=0.0, reward clip [-2, 1], disjoint tasks)
2. Salvage-discounted reward clipping ([-2, 1]) and partial salvage math
3. Group advantage math (G=6 normalized, zero-std safe fallback)
4. Behavior-Cloning warm-start masking (-100 on prompt tokens) and LoRA weight updates with frozen base weights
5. GRPO loss computation (no reference model, finite LoRA gradients, zero loss on zero advantages)
   and end-to-end 36-step optimization loop with evaluation harness.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GemmaConfig, GemmaForCausalLM

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terraria_bench.harness.gemma_agent import (
    AgentConfig,
    GemmaAgent,
    StepRecord,
    Trajectory,
    evaluate_agent,
)
from terraria_bench.harness.train_grpo import (
    EVAL_TASKS,
    TRAIN_TASKS,
    BehaviorCloningTrainer,
    GRPOConfig,
    GRPOTrainer,
    build_lora_config,
    compute_group_advantages,
    compute_salvage_discounted_reward,
)


class MockTokenizer:
    """Fast, deterministic mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Map characters deterministically into token IDs in range [3, vocab_size - 1]
        tokens = [((ord(c) * 7 + 13) % (self.vocab_size - 3)) + 3 for c in text[:12]]
        if not tokens:
            tokens = [3]
        if add_special_tokens:
            return [self.bos_token_id] + tokens
        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        # Return a valid JSON action tool call string for agent testing
        return '{"action": "craft", "item": "wooden_sword"}'


class MockBenchmarkEnv:
    """Mock environment simulating Terraria benchmark episodes."""

    def __init__(self, task_id: str = "craft_wooden_sword", max_steps: int = 2):
        self.task_id = task_id
        self.max_steps = max_steps
        self.step_count = 0
        self.seed = 0

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.step_count = 0
        self.seed = seed if seed is not None else 0
        obs = {
            "task": self.task_id,
            "nearby_tiles": ["wood", "crafting_table"],
            "inventory": {"wood": 10},
        }
        info = {"seed": self.seed}
        return obs, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # Reward simulation with exploration variation across seeds
        is_success = action.get("action") == "craft"
        seed_bonus = 0.1 * ((self.seed % 6) - 2.5)
        reward = (0.5 if is_success else -0.1) + seed_bonus

        obs = {
            "task": self.task_id,
            "status": "in_progress" if not done else "finished",
        }
        info = {
            "success": is_success and done,
            "terminal_score": 0.2 if is_success and done else 0.0,
            "salvage_credit": 0.2 + seed_bonus,
            "salvage_reward": 0.2 + seed_bonus,
        }
        return obs, reward, done, False, info


@pytest.fixture
def tiny_causal_lm() -> GemmaForCausalLM:
    """Creates a tiny 1-layer Gemma causal LM for ultra-fast, deterministic CPU testing."""
    cfg = GemmaConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        head_dim=16,
    )
    model = GemmaForCausalLM(cfg)
    return model


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer(vocab_size=64)


# ==============================================================================
# 1. StoreBench §7 Config & Task Invariants
# ==============================================================================


def test_grpo_config_storebench_invariants():
    """Verify that GRPOConfig defaults strictly enforce all StoreBench §7 invariants."""
    cfg = GRPOConfig()

    # Exact hyperparameter values per StoreBench §7
    assert cfg.lora_rank == 32, "LoRA rank must be 32 (r=32)"
    assert cfg.lora_alpha == 32, "LoRA alpha must be 32 (lora_alpha=32, scaling factor 1.0)"
    assert cfg.group_size == 6, "Group size must be 6 (G=6 rollouts per step)"
    assert cfg.total_steps == 36, "Total optimization steps must be 36"
    assert cfg.kl_coef == 0.0, "KL penalty coefficient must be strictly 0.0 (beta=0)"
    assert cfg.reward_clip_min == -2.0, "Reward clipping minimum must be -2.0"
    assert cfg.reward_clip_max == 1.0, "Reward clipping maximum must be 1.0"
    assert cfg.clip_eps == 0.2, "PPO surrogate clip epsilon must be 0.2"


def test_task_disjointness_invariants():
    """Verify TRAIN_TASKS has 4 disjoint tasks and is strictly disjoint from EVAL_TASKS."""
    assert len(TRAIN_TASKS) == 4, "TRAIN_TASKS must contain exactly 4 tasks"
    assert len(set(TRAIN_TASKS)) == 4, "TRAIN_TASKS must not contain duplicate tasks"

    assert len(EVAL_TASKS) >= 4, "EVAL_TASKS must contain at least 4 tasks"
    assert len(set(EVAL_TASKS)) == len(EVAL_TASKS), "EVAL_TASKS must not contain duplicate tasks"

    # Strict disjointness
    disjoint = set(TRAIN_TASKS).isdisjoint(set(EVAL_TASKS))
    assert disjoint, f"TRAIN_TASKS and EVAL_TASKS must be completely disjoint: {set(TRAIN_TASKS) & set(EVAL_TASKS)}"

    # Configuration validation rejects invalid task list
    with pytest.raises(ValueError, match="must contain exactly 4 disjoint tasks"):
        GRPOConfig(train_tasks=("task1", "task2", "task3"))  # only 3 tasks

    with pytest.raises(ValueError, match="must contain exactly 4 disjoint tasks"):
        GRPOConfig(train_tasks=("task1", "task2", "task3", "task1"))  # duplicate


def test_build_lora_config_invariants(tiny_causal_lm):
    """Verify build_lora_config creates a valid LoraConfig with r=32 and lora_alpha=32."""
    cfg = GRPOConfig()
    lora_cfg = build_lora_config(cfg, tiny_causal_lm)

    assert isinstance(lora_cfg, LoraConfig)
    assert lora_cfg.r == 32
    assert lora_cfg.lora_alpha == 32
    # Verify target modules exist in the model
    assert len(lora_cfg.target_modules) > 0


# ==============================================================================
# 2. Salvage-Discounted Reward Clipping ([-2.0, 1.0])
# ==============================================================================


def test_salvage_discounted_reward_clipping():
    """Verify strict reward clipping to [-2.0, 1.0] and discounting calculations."""
    # 1. Extreme negative penalty clips to exactly -2.0
    r_extreme_neg = compute_salvage_discounted_reward(
        terminal_score=-5.0,
        invalid_actions=100,
        step_penalty=1.0,
    )
    assert r_extreme_neg == -2.0, f"Expected -2.0 clip, got {r_extreme_neg}"

    # 2. High bonus clips to exactly 1.0
    r_extreme_pos = compute_salvage_discounted_reward(
        terminal_score=10.0,
        salvage_credit=5.0,
    )
    assert r_extreme_pos == 1.0, f"Expected 1.0 clip, got {r_extreme_pos}"

    # 3. Intermediate partial-salvage calculation remains unclipped
    steps = 5
    discount = 0.99
    step_penalty = 0.01
    invalid_actions = 1
    invalid_penalty = 0.05
    terminal_score = 0.0
    salvage_credit = 0.6
    salvage_weight = 0.5

    r_mid = compute_salvage_discounted_reward(
        terminal_score=terminal_score,
        salvage_credit=salvage_credit,
        steps=steps,
        invalid_actions=invalid_actions,
        discount=discount,
        step_penalty=step_penalty,
        invalid_action_penalty=invalid_penalty,
        salvage_weight=salvage_weight,
    )

    expected = (
        (terminal_score + salvage_weight * salvage_credit) * (discount ** steps)
        - steps * step_penalty
        - invalid_actions * invalid_penalty
    )
    assert pytest.approx(r_mid, rel=1e-5) == expected
    assert -2.0 <= r_mid <= 1.0


def test_telescoping_dense_rewards():
    """Confirm sum(unclipped_dense_rewards) equals (phi_final - phi0) / s to within 1e-6 on all 4 training tasks."""
    try:
        from terraria_bench.tools import Episode
    except ImportError:
        pytest.skip("Episode not available")

    cell = {"z": 0.0, "s": 10.0}
    for task_id in TRAIN_TASKS:
        ep = Episode(task_id, 7)
        ep.step("terraria_status")
        ep.step("end_window")
        ledger = ep.finalize()

        prev = ledger["phi0"]
        unclipped = []
        for r in ledger["windows"]:
            unclipped.append((r["phi"] - prev) / cell["s"])
            prev = r["phi"]

        expected = (ledger["phi_final"] - ledger["phi0"]) / cell["s"]
        assert abs(sum(unclipped) - expected) < 1e-6


# ==============================================================================
# 3. Group Advantage Math (G = 6)
# ==============================================================================


def test_compute_group_advantages():
    """Verify group-normalized advantages for G=6 varied and identical rewards."""
    # 1. Varied rewards for G=6 group
    rewards = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float32)
    advantages = compute_group_advantages(rewards)

    assert advantages.shape == (6,)
    # Population mean ~ 0.0, population std ~ 1.0
    assert pytest.approx(advantages.mean().item(), abs=1e-6) == 0.0
    assert pytest.approx(advantages.std(unbiased=False).item(), rel=1e-4) == 1.0

    # 2. Zero-variance group (e.g. all -2.0) safely returns zeros without NaNs or Infs
    identical_rewards = torch.full((6,), -2.0, dtype=torch.float32)
    zero_adv = compute_group_advantages(identical_rewards)

    assert torch.equal(zero_adv, torch.zeros_like(identical_rewards))
    assert not torch.isnan(zero_adv).any()
    assert not torch.isinf(zero_adv).any()


# ==============================================================================
# 4. Behavior-Cloning Warm-Start Masking & Weight Updates
# ==============================================================================


def test_bc_warm_start_masking_and_lora_updates(tiny_causal_lm, mock_tokenizer, tmp_path):
    """Verify BC prompt masking with -100 and LoRA weight updates with frozen base weights."""
    agent_cfg = AgentConfig(torch_dtype="float32", device="cpu", max_episode_steps=5)
    agent = GemmaAgent(config=agent_cfg, model=tiny_causal_lm, tokenizer=mock_tokenizer)

    grpo_cfg = GRPOConfig(output_dir=str(tmp_path), bc_epochs=1, bc_lr=1e-3)
    lora_cfg = build_lora_config(grpo_cfg, agent.model)
    agent.attach_fresh_lora(lora_cfg)

    # Record initial weights
    base_weights_init = {
        name: param.detach().clone()
        for name, param in agent.model.named_parameters()
        if "lora_" not in name
    }
    lora_weights_init = {
        name: param.detach().clone()
        for name, param in agent.model.named_parameters()
        if "lora_" in name and param.requires_grad
    }

    assert len(lora_weights_init) > 0, "LoRA adapter parameters must exist"

    # Human demonstration
    demonstrations = [
        {
            "prompt": "Observation: You see wood on the ground.",
            "action": '{"action": "craft", "item": "wooden_sword"}',
        },
        {
            "prompt": "Observation: You see iron ore in the cave.",
            "action": '{"action": "mine", "target": "iron_ore"}',
        },
    ]

    bc_trainer = BehaviorCloningTrainer(agent=agent, config=grpo_cfg, demonstrations=demonstrations)
    save_path = bc_trainer.train()

    assert save_path.exists()

    # 1. Verify base weights remained strictly frozen
    for name, param in agent.model.named_parameters():
        if "lora_" not in name:
            assert torch.equal(param, base_weights_init[name]), f"Base parameter {name} was unexpectedly modified!"

    # 2. Verify LoRA adapter weights were updated
    lora_changed = any(
        not torch.equal(param, lora_weights_init[name])
        for name, param in agent.model.named_parameters()
        if name in lora_weights_init
    )
    assert lora_changed, "LoRA adapter parameters should have updated during BC training"


# ==============================================================================
# 5. GRPO Loss & End-to-End 36-Step Optimization Loop
# ==============================================================================


def test_grpo_loss_properties(tiny_causal_lm, mock_tokenizer):
    """Verify compute_grpo_loss properties: beta=0 (no ref model), finite LoRA grads, zero loss on zero adv."""
    agent_cfg = AgentConfig(torch_dtype="float32", device="cpu")
    agent = GemmaAgent(config=agent_cfg, model=tiny_causal_lm, tokenizer=mock_tokenizer)

    grpo_cfg = GRPOConfig()
    lora_cfg = build_lora_config(grpo_cfg, agent.model)
    agent.attach_fresh_lora(lora_cfg)

    trainer = GRPOTrainer(agent=agent, config=grpo_cfg)

    # Create dummy trajectories
    prompt_ids = [2, 10, 15, 20]
    action_ids = [30, 35, 40]
    # Compute initial old log-probs
    with torch.no_grad():
        step_rec = StepRecord(
            prompt_text="test prompt",
            action_text="test action",
            parsed_tool_call={"action": "test"},
            prompt_token_ids=prompt_ids,
            action_token_ids=action_ids,
            action_log_probs=[-0.5, -0.6, -0.7],
        )
        trajectories = [
            Trajectory(
                steps=[step_rec],
                task_id="craft_wooden_sword",
                total_clipped_reward=1.0,
            )
            for _ in range(6)
        ]

    # 1. Zero advantages yield zero loss
    zero_adv = torch.zeros(6)
    loss_zero = trainer.compute_grpo_loss(trajectories, zero_adv)
    assert loss_zero.item() == 0.0

    # 2. Non-zero advantages compute valid loss and finite gradients on LoRA params
    non_zero_adv = torch.tensor([1.0, 0.5, -0.5, -1.0, 0.8, -0.8])
    loss = trainer.compute_grpo_loss(trajectories, non_zero_adv)
    assert torch.isfinite(loss)

    trainer.optimizer.zero_grad()
    loss.backward()

    for name, param in agent.model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for trainable param {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_grpo_36_steps_end_to_end_loop(tiny_causal_lm, mock_tokenizer, tmp_path):
    """Executes full 36-step GRPO loop verifying G=6 rollouts, clipped rewards, weight updates, and eval."""
    agent_cfg = AgentConfig(
        torch_dtype="float32",
        device="cpu",
        max_episode_steps=2,
        max_new_tokens=4,
    )
    agent = GemmaAgent(config=agent_cfg, model=tiny_causal_lm, tokenizer=mock_tokenizer)

    grpo_cfg = GRPOConfig(
        output_dir=str(tmp_path),
        total_steps=36,
        group_size=6,
        learning_rate=1e-3,
    )
    lora_cfg = build_lora_config(grpo_cfg, agent.model)
    agent.attach_fresh_lora(lora_cfg)

    # Initial baseline evaluation
    baseline_eval = evaluate_agent(
        agent=agent,
        env_factory=lambda task_id: MockBenchmarkEnv(task_id=task_id, max_steps=2),
        task_ids=list(TRAIN_TASKS),
        num_seeds=1,
    )
    assert "mean_clipped_reward" in baseline_eval
    assert "success_rate" in baseline_eval
    assert len(baseline_eval["tasks"]) == 4

    # Record weights before GRPO
    base_init = {
        name: p.detach().clone()
        for name, p in agent.model.named_parameters()
        if "lora_" not in name
    }
    lora_init = {
        name: p.detach().clone()
        for name, p in agent.model.named_parameters()
        if "lora_" in name and p.requires_grad
    }

    # Run full 36-step GRPO trainer
    trainer = GRPOTrainer(
        agent=agent,
        config=grpo_cfg,
        env_factory=lambda task_id: MockBenchmarkEnv(task_id=task_id, max_steps=2),
    )

    final_adapter_dir = trainer.train()

    assert final_adapter_dir.exists()
    history_file = tmp_path / "training_history.json"
    assert history_file.exists()

    with open(history_file, encoding="utf-8") as f:
        history = json.load(f)

    # Invariant: exactly 36 steps executed
    assert len(history) == 36, f"Expected 36 training steps, got {len(history)}"

    # Verify task distribution across 36 steps (9 passes per task for 4 tasks)
    task_counts = {}
    for entry in history:
        t = entry["task_id"]
        task_counts[t] = task_counts.get(t, 0) + 1
        # All rewards strictly in [-2.0, 1.0]
        assert -2.0 <= entry["min_reward"] <= 1.0
        assert -2.0 <= entry["max_reward"] <= 1.0
        assert -2.0 <= entry["mean_reward"] <= 1.0

    assert len(task_counts) == 4
    for task_name, count in task_counts.items():
        assert count == 9, f"Task {task_name} should have exactly 9 steps, got {count}"

    # Verify base weights frozen and LoRA weights updated
    for name, p in agent.model.named_parameters():
        if "lora_" not in name:
            assert torch.equal(p, base_init[name]), f"Base parameter {name} was modified!"

    lora_updated = any(
        not torch.equal(p, lora_init[name])
        for name, p in agent.model.named_parameters()
        if name in lora_init
    )
    assert lora_updated, "LoRA parameters should have updated during 36-step GRPO loop"

    # Post-LoRA evaluation
    post_eval = evaluate_agent(
        agent=agent,
        env_factory=lambda task_id: MockBenchmarkEnv(task_id=task_id, max_steps=2),
        task_ids=list(TRAIN_TASKS),
        num_seeds=1,
        output_dir=str(tmp_path / "post_eval"),
    )
    assert "mean_clipped_reward" in post_eval
    assert -2.0 <= post_eval["mean_clipped_reward"] <= 1.0
    assert (tmp_path / "post_eval" / "eval_metrics.json").exists()
    assert (tmp_path / "post_eval" / "trajectories.json").exists()
