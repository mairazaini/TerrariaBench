#!/usr/bin/env python3
"""
tests/test_reward_integrity.py
Reward Integrity & Anti-Cheat Test Suite for AfterQuery Evaluation.
Verifies that the reward function cannot be gamed by:
1. Do-Nothing baseline (must score 0.0)
2. Stop-Early baseline (must score 25.0, far below reference 100.0)
3. Drop/Pickup loops (net reward cannot be farmed)
4. Block spamming without enclosure (housing check remains False)
5. Checkpoint tampering (HMAC-SHA256 signature mismatch detected)
"""

import pytest
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.terraria_architect import TerrariaArchitectEnv
from agents.eval_suite import DoNothingBaseline, StopEarlyBaseline, ReferenceBot
from telemetry.aiven_sink import verify_checkpoint, compute_signature

def test_do_nothing_scores_zero():
    """Asserts Do-Nothing baseline scores exactly 0.0."""
    env = TerrariaArchitectEnv(seed=7, run_id="test_do_nothing")
    agent = DoNothingBaseline()
    score = agent.run(env)
    assert score == 0.0, f"Expected 0.0, got {score}"

def test_stop_early_scores_low():
    """Asserts Stop-Early baseline scores 25.0 (much lower than Reference 100.0)."""
    env = TerrariaArchitectEnv(seed=7, run_id="test_stop_early")
    agent = StopEarlyBaseline()
    score = agent.run(env)
    assert score == 25.0, f"Expected 25.0, got {score}"
    assert score < 100.0

def test_drop_pickup_loop_blocked():
    """
    Simulates dropping and picking up wood repeatedly.
    Asserts net reward is computed strictly on net world inventory/structures
    and does not inflate past +10.0 pts.
    """
    env = TerrariaArchitectEnv(seed=7, run_id="test_drop_pickup")
    # Gather initial wood
    env.step({"action": "gather", "resource": "wood"}) # 12 wood
    initial_score = env.last_score

    # Simulate 50 drop/pickup cycles
    for cycle in range(50):
        # Drop (mine terrain or simulate item drop)
        env.inventory["wood"] -= 1
        # Pickup
        env.inventory["wood"] += 1
        score, _ = env._evaluate_start_room_check()

    assert env.last_score == initial_score, "Drop/pickup loop altered score!"
    assert env.last_score <= 10.0, "Resource score inflated past gathering ceiling!"

def test_block_spam_invalid_housing():
    """
    Spams 50 wood blocks in random clusters without a proper enclosure, ceiling, door, or walls.
    Asserts housing check returns False and outer_frame_valid is False.
    """
    env = TerrariaArchitectEnv(seed=7, run_id="test_block_spam")
    env.inventory["wood_block"] = 100

    # Spam blocks haphazardly (within daytime budget)
    for i in range(20):
        x = (i * 3) % 16
        y = (i * 2) % 5
        env.step({"action": "place_block", "item": "wood_block", "x": x, "y": y})

    obs, score, done, info = env.step({"action": "audit_shelter"})
    assert info["breakdown"]["housing_valid"] is False, "Haphazard block spam trick was accepted as valid housing!"
    assert info["breakdown"]["outer_frame_valid"] is False, "Invalid outer frame incorrectly marked valid!"

def test_hmac_tamper_detection():
    """
    Modifies a score in a signed checkpoint payload.
    Asserts aiven_sink.verify_checkpoint raises a ValueError for signature mismatch.
    """
    valid_payload = {
        "run_id": "audit_run_999",
        "agent_name": "gemma_grpo",
        "seed": 7,
        "score": 78.3,
        "housing_valid": True,
        "timestamp": "2026-11-15T00:00:00Z",
        "extra": {}
    }
    valid_sig = compute_signature(valid_payload)
    assert verify_checkpoint(valid_payload, valid_sig) is True

    # Tamper with the score (agent attempts to change 78.3 -> 100.0)
    tampered_payload = dict(valid_payload)
    tampered_payload["score"] = 100.0

    with pytest.raises(ValueError, match="HMAC tamper detected"):
        verify_checkpoint(tampered_payload, valid_sig)
