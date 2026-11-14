"""Tests for gemini-decision-1 controls engine, failure clip detection, and live tModLoader bridge."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from terraria_bench.live_play import (
    CONTROL_QUESTIONS,
    DecisionController,
    build_failure_clips,
    create_live_app,
    detect_step_failures,
    run_recorded_episode,
)


def test_offline_recording_and_failure_clip_detection(tmp_path: Path):
    out = run_recorded_episode(
        task_id="eye-of-cthulhu",
        seed=7,
        live=False,
        max_turns=25,
        out_dir=tmp_path,
        run_id="eye_test",
        verbose=False,
    )
    json_path = Path(out["json_path"])
    html_path = Path(out["html_path"])
    assert json_path.exists()
    assert html_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["supervisor_model"] == "gemini-4-hydrogen"
    assert data["decision_model"] == "gemini-decision-1"
    assert len(data["frames"]) >= 20
    assert len(data["failure_clips"]) >= 1
    assert "TerrariaBench Playthrough &amp; Failure Clip Studio" in html_path.read_text(encoding="utf-8")


def test_parallel_decision_controller_with_mock_api(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    recorded_reqs: list[dict] = []

    class _MockClient:
        class decisions:
            @staticmethod
            def create(*, model, input, questions):
                recorded_reqs.append({"model": model, "input": input, "questions": questions})

                class _Choice:
                    def __init__(self, val, conf=0.9):
                        self.choice = val
                        self.confidence = conf
                        self.probabilities = {val: conf}

                class _Noul:
                    def __init__(self, val):
                        self.noul = val
                        self.confidence = max(val, 1.0 - val)

                class _Score:
                    def __init__(self, val):
                        self.score = val
                        self.confidence = 0.85
                        self.probabilities = {"0": 0.1, "1": 0.7, "2": 0.2, "3": 0.0}

                class _Resp:
                    answers = {
                        "next_action": _Choice("chop"),
                        "move_x": _Choice("right"),
                        "jump": _Noul(0.82),
                        "crouch_down": _Noul(0.10),
                        "use_item": _Noul(0.91),
                        "interact": _Noul(0.12),
                        "grapple": _Noul(0.05),
                        "quick_heal": _Noul(0.04),
                        "smart_cursor": _Noul(0.88),
                        "hotbar_slot": _Choice("slot_3_axe"),
                        "aim_target": _Choice("nearest_tree"),
                        "threat_level": _Score(0.8),
                    }

                return _Resp()

    ctrl = DecisionController(live=True)
    ctrl.agent._client = _MockClient()
    ctrl.agent.decisions.client = _MockClient()
    ctrl.agent.supervisor_fn = lambda text, png: json.dumps({
        "strategy": "Chop trees first.",
        "candidates": [{"id": "chop", "tool": "chop_trees", "args": {"radius": 30, "max_trees": 3}, "when": "Need wood"}],
    })

    from terraria_bench.tools import Episode

    ep = Episode("first-night", 7, redact_seed=True)
    dec = ctrl.decide_turn({"ok": True}, [], ep)
    assert dec["tool"] == "chop_trees"
    assert dec["controls"]["right"] is True
    assert dec["controls"]["jump"] is True
    assert dec["controls"]["use_item"] is True
    assert dec["controls"]["smart_cursor"] is True
    assert dec["controls"]["hotbar_slot"] == "slot_3_axe"
    assert dec["controls"]["aim_target"] == "nearest_tree"
    assert set(CONTROL_QUESTIONS).issubset(recorded_reqs[0]["questions"])
    assert "next_action" in recorded_reqs[0]["questions"]


def test_real_game_tick_endpoint_and_failure_clip_index(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    client = TestClient(app)

    r1 = client.post("/real_game_tick", json={
        "x": 100, "y": 50, "hp": 100, "max_hp": 100, "defense": {"Flat": 4, "Base": 2}, "dayTime": True, "deaths": 0,
        "ascii_grid": ["." * 61 for _ in range(35)],
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["ok"] is True
    assert "controls" in d1 and "active_keys" in d1["controls"]
    assert "macro" in d1 and "stage" in d1["macro"]
    assert client.get("/api/real/latest").json()["latest"]["state"]["defense"] == 4

    # Trigger heavy damage & death on next tick -> failure clip created
    r2 = client.post("/real_game_tick", json={
        "x": 102, "y": 50, "hp": 0, "max_hp": 100, "defense": 2, "dayTime": False, "deaths": 1,
        "ascii_grid": ["." * 61 for _ in range(35)],
    })
    assert r2.status_code == 200
    assert (tmp_path / "real_terraria_live.json").exists()
    rec = json.loads((tmp_path / "real_terraria_live.json").read_text(encoding="utf-8"))
    assert len(rec["failure_clips"]) >= 1

    assert client.get("/theater").status_code == 200
    assert client.get("/play/real_terraria_live").status_code == 200
    assert client.get("/api/recordings").json()["ok"] is True
    assert client.get("/api/real/latest").json()["frames"] == 2

    # Direct failure detection unit check
    fails = detect_step_failures(
        {"hp": 100, "deaths": 0},
        {"hp": 80, "deaths": 0, "last_death": "", "phase": "Night", "sheltered": False, "pos": [10, 20], "window": 2},
        "mine_vein",
        {"ok": False, "error": "too far"},
        "replan",
        0.30,
        True,
    )
    tags = {f["tag"] for f in fails}
    assert {"HEAVY_DAMAGE", "NIGHT_UNSHELTERED", "ACTION_FAILED", "LOW_CONFIDENCE_REPLAN"}.issubset(tags)
    clips = build_failure_clips([{"turn": 0, "failure_events": []}, {"turn": 1, "failure_events": fails}])
    assert len(clips) == 1


def test_trapped_in_hole_escape_picks_pickaxe_and_smart_cursor(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    client = TestClient(app)

    grid = ["." * 61 for _ in range(35)]
    grid[15] = "TsssTTssdddsdddddddsssTTT....ddddddddddddddddddddddssTTsssTT."
    grid[16] = "d" * 29 + "...." + "d" * 28
    grid[17] = "d" * 29 + ".@" + "d" * 30
    grid[18] = "d" * 29 + ".." + "d" * 30

    resp = client.post("/real_game_tick", json={
        "x": 2000, "y": 400, "vx": 0.0, "vy": 0.0, "hp": 100, "max_hp": 100,
        "defense": 0, "dayTime": True, "deaths": 0,
        "nearby_enemies": [{"name": "Green Slime", "hp": 14, "dist": 6.0, "dx": 3.0, "dy": -4.0}],
        "ascii_grid": grid,
    })
    assert resp.status_code == 200
    ctrl = resp.json()["controls"]
    assert ctrl["hotbar_slot"] == "slot_2_pickaxe"
    assert ctrl["smart_cursor"] is True
    assert ctrl["aim_target"] == "build_site"
    assert ctrl["left"] is True
    assert ctrl["use_item"] is True
    assert ctrl["quick_heal"] is False
