"""
tests/test_reward_integrity.py - Anti-Cheat and Reward Integrity Test Suite

Verifies:
1. Baselines meet calibrated requirements: do_nothing == 0.0, stop_early <= 15.0.
2. Reference agent achieves 100.0 reproducible score.
3. Exploits (item drop/re-pickup, cycling, quitting) are penalized.
4. Local file tampering is caught and rejected by Aiven off-box HMAC signatures.
"""

import os
import sys
import pytest
import hmac
import hashlib

from agents.reference_and_baselines import DoNothingAgent, StopEarlyAgent, ReferenceAgent
from telemetry.aiven_sink import TamperProofAivenSink, SECRET_KEY


def test_do_nothing_baseline_zero():
    """Verify that an agent taking no actions scores exactly 0.0."""
    agent = DoNothingAgent()
    res = agent.evaluate("first-night", 11)
    assert res["score"] == 0.0, f"Expected 0.0 for DoNothing, got {res['score']}"


def test_stop_early_baseline_capped():
    """Verify that an early halting baseline scores at most 15.0."""
    agent = StopEarlyAgent()
    res = agent.evaluate("first-night", 11)
    assert 0.0 <= res["score"] <= 15.0, f"Expected <= 15.0 for StopEarly, got {res['score']}"


def test_reference_agent_perfect_score():
    """Verify that the ReferenceAgent achieves 100.0 on calibrated task/seed."""
    agent = ReferenceAgent()
    res = agent.evaluate("first-night", 11)
    assert res["score"] == 100.0, f"Expected 100.0 for ReferenceAgent, got {res['score']}"


def test_anti_cheat_delta_verification():
    """Verify that item loop cycling or exploit flag is rejected and penalized."""
    sink = TamperProofAivenSink()
    
    prev_state = {"inventory": {"wood": 10}, "placed": set()}
    curr_state = {"inventory": {"wood": 10}, "placed": set(), "is_exploit_cycle": True}
    
    # Verification should return False for exploit
    is_valid = sink.verify_net_world_delta(prev_state, curr_state)
    assert not is_valid, "Anti-cheat net delta check should fail on exploit cycle"

    # Recording settlement should penalize reward
    rec = sink.record_settlement(
        run_id="exploit_test_run",
        agent_model="exploiter_bot",
        seed=11,
        window_idx=1,
        net_progression=50.0,
        shelter_valid=False,
        reward_score=50.0,
        prev_state=prev_state,
        curr_state=curr_state
    )
    assert rec["reward_score"] == -2.0, "Exploit runs must be clamped to -2.0 penalty"


def test_hmac_tamper_proofing():
    """Verify that modifying local score without secret key invalidates HMAC signature."""
    sink = TamperProofAivenSink()
    rec = sink.record_settlement(
        run_id="run_integrity_check",
        agent_model="gemma-base",
        seed=11,
        window_idx=1,
        net_progression=17.18,
        shelter_valid=False,
        reward_score=17.18
    )

    # Legitimate signature passes
    assert sink.verify_settlement_signature(rec) is True

    # Tampered signature or tampered score must fail
    tampered_rec = dict(rec)
    tampered_rec["reward_score"] = 99.0  # Adversarial agent hacked local score
    assert sink.verify_settlement_signature(tampered_rec) is False, "Tampered score must fail HMAC check"


def test_offline_wiki_mechanics_fallback():
    """Verify that wiki mechanics lookup works instantly offline without cloud credentials."""
    from mcp_server.terraria_mcp import lookup_terraria_mechanics
    result = lookup_terraria_mechanics("housing")
    assert "Valid Terraria NPC Housing Requirements" in result
    assert "Enclosure" in result
