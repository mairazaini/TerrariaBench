"""Byte-identical reproduction, replay, RNG isolation and fail-closed grading."""
from __future__ import annotations

import copy
import json

import pytest

from terraria_bench import engine
from terraria_bench.grader import grade
from terraria_bench.policies import run_policy, run_policy_episode
from terraria_bench.provenance import (
    SEED_DATA_PATH,
    canonical,
    load_sealed,
    seal,
    source_hash,
    stream,
    verify_seal,
)
from terraria_bench.tasks import HELD_OUT_SEEDS, REPORTED_SEEDS
from terraria_bench.tools import Episode

CELL = {"z": 0.0, "s": 100.0, "threshold": 0.5}


@pytest.fixture(scope="module")
def smart_episode():
    return run_policy_episode("smart-triage", "cavern-expedition", 7)


def test_same_cell_same_ledger(smart_episode):
    ledger1 = smart_episode.finalize()
    ledger2 = run_policy("smart-triage", "cavern-expedition", 7)
    assert ledger1["sha256"] == ledger2["sha256"]
    assert ledger1["state_hash"] == ledger2["state_hash"]
    assert canonical(ledger1) == canonical(ledger2)


def test_actions_log_replays_byte_identically(smart_episode):
    original = smart_episode.finalize()
    replay = Episode("cavern-expedition", 7)
    for action in smart_episode.actions_log:
        replay.step(action["tool"], **action["args"])
    assert replay.actions_log == smart_episode.actions_log
    assert canonical(replay.finalize()) == canonical(original)


def test_no_state_leaks_between_episodes():
    """Running other episodes in between (which share cached world templates)
    must not change a ledger: episodes copy the template, never mutate it."""
    before = run_policy("rule-based", "ore-rush", 11)
    grid_before = b"".join(bytes(r) for r in engine._template(11)["grid"])
    run_policy("blast-list", "ore-rush", 11)
    run_policy("smart-triage", "first-night", 11)
    after = run_policy("rule-based", "ore-rush", 11)
    assert canonical(before) == canonical(after)
    assert b"".join(bytes(r) for r in engine._template(11)["grid"]) == grid_before


def test_purpose_keyed_streams_are_isolated():
    a = [stream(7, "mobs", 3, "surface").random() for _ in range(5)]
    # drawing heavily from an unrelated purpose must not shift this one
    noise = stream(7, "terrain")
    for _ in range(1000):
        noise.random()
    b = [stream(7, "mobs", 3, "surface").random() for _ in range(5)]
    assert a == b
    assert stream(7, "terrain").random() != stream(7, "ores").random()
    assert stream(7, "terrain").random() != stream(11, "terrain").random()
    # stable across runs, platforms and Python versions: pinned first draws
    assert stream(7, "terrain").getrandbits(32) == 821038982
    assert stream(23, "mobs", 1, "surface").getrandbits(32) == 1982779275


def test_world_hashes_match_pinned_seed_data():
    pins = load_sealed(SEED_DATA_PATH)
    if pins.get("source_hash") != source_hash():
        pytest.skip("seed pins predate a source change; run python -m terraria_bench.calibrate")
    for seed in REPORTED_SEEDS + HELD_OUT_SEEDS:
        assert pins["seeds"][str(seed)] == engine.world_hash(seed)


# ---- fail closed ------------------------------------------------------------
@pytest.fixture(scope="module")
def sealed_ledger():
    return run_policy("rule-based", "first-night", 7)


def test_untampered_ledger_grades(sealed_ledger):
    assert verify_seal(sealed_ledger)
    assert grade(sealed_ledger, CELL)["valid"] is True


def test_one_byte_mutation_fails_closed(sealed_ledger):
    raw = bytearray(json.dumps(sealed_ledger, sort_keys=True).encode())
    i = raw.index(b'"state_hash": "') + len(b'"state_hash": "')
    raw[i] = ord("0") if raw[i] != ord("0") else ord("1")   # flip one byte
    corrupted = json.loads(raw)
    g = grade(corrupted, CELL)
    assert g["valid"] is False and g["composite"] == 0.0


@pytest.mark.parametrize("mutate", [
    lambda L: L["windows"][0].__setitem__("attended", not L["windows"][0]["attended"]),
    lambda L: L.__setitem__("phi_final", L["phi_final"] + 1000),
    lambda L: L.__setitem__("milestones", L["milestones"] + ["KILL:eye"]),
    lambda L: L.pop("sha256"),
], ids=["attended", "phi_final", "milestones", "no_seal"])
def test_field_tampering_fails_closed(sealed_ledger, mutate):
    corrupted = copy.deepcopy(sealed_ledger)
    mutate(corrupted)
    g = grade(corrupted, CELL)
    assert g["valid"] is False and g["composite"] == 0.0


def test_resealed_partial_episode_fails_closed(sealed_ledger):
    partial = seal({**{k: v for k, v in sealed_ledger.items() if k != "sha256"},
                    "windows": sealed_ledger["windows"][:2]})
    g = grade(partial, CELL)
    assert g["valid"] is False and "finalized" in g["reason"]


@pytest.mark.parametrize("bad", [None, "ledger", {}, []])
def test_garbage_fails_closed(bad):
    assert grade(bad, CELL)["composite"] == 0.0


def test_missing_calibration_fails_closed(sealed_ledger):
    for cell in (None, {}, {"z": 0.0, "s": 0.0}):
        g = grade(sealed_ledger, cell)
        assert g["valid"] is False and g["composite"] == 0.0
