"""Tests for gemini-decision-1 controls engine, failure clip detection, and live tModLoader bridge."""
from __future__ import annotations

import json
import threading
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
from terraria_bench.real_game import build_questions, builtin_routines


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


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _payload(hp: int = 100, deaths: int = 0, **hud_over) -> dict:
    hud = {
        "hp": hp, "max_hp": 100, "mana": 20, "max_mana": 20, "dead": hp <= 0, "selected_slot": 1,
        "held_item": "Copper Shortsword",
        "hotbar": [
            {"slot": 1, "name": "Copper Shortsword", "stack": 1},
            {"slot": 2, "name": "Copper Pickaxe", "stack": 1},
            {"slot": 3, "name": "Copper Axe", "stack": 1},
        ],
        "buffs": [], "inventory_open": False, "inventory": {}, "equipment": [], "craftable": [],
        "chest_open": False, "chest": None, "cursor": {"dx": 0, "dy": 0}, "events": [],
        "view": {"px_per_tile": 5.3, "shot_width": 640, "shot_height": 360},
    }
    hud.update(hud_over)
    return {
        "protocol": 2, "tick": 1, "screenshot_b64": PNG_1X1, "hud": hud,
        # Ground truth with unmistakable sentinel values: must never appear in any model prompt.
        "referee": {
            "x": 987654.25, "y": 123456.75, "depth_below_surface": 31337, "deaths": deaths, "dayTime": True,
            "defense": 4, "boss_active": False, "boss_name": "SENTINEL_BOSS", "inventory": {"SENTINEL_ORE": 77},
        },
    }


def _wire_mock_models(app, answers: dict, supervisor_fn=None):
    """Replace all three tiers with recording fakes; returns the call log."""
    real = app.state.controller.real
    calls = {"vision": [], "supervisor": [], "decision": []}

    def vision_fn(prompt, png):
        calls["vision"].append((prompt, png))
        return json.dumps({"lighting": "pitch_black", "surroundings": "Nothing visible, the screen is black.", "enemies": []})

    def default_supervisor(prompt, png):
        calls["supervisor"].append((prompt, png))
        return json.dumps({"strategy": "Make light before moving.", "objective": "Craft torches", "steps": ["open inventory"], "avoid": [], "done_when": "torches in hotbar"})

    def decide(questions, context_text, viewport_png_b64=None):
        calls["decision"].append((questions, context_text, viewport_png_b64))
        return {"model": "gemini-decision-1", "source": "decisions_api",
                "answers": {k: {"type": "x", "value": v, "confidence": 0.9} for k, v in answers.items()}}

    real.live = True
    for tier in real.tiers.values():
        tier.status = "WAITING"
    real.vision_fn = vision_fn
    real.supervisor_fn = supervisor_fn or default_supervisor
    real.async_supervisor = False
    real.async_vision = False
    real.agent.decisions.decide = decide
    return calls


def test_real_game_tick_endpoint_and_failure_clip_index(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    client = TestClient(app)

    r1 = client.post("/real_game_tick", json=_payload())
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["ok"] is True
    assert "stage" in d1["macro"]
    # No models connected: the player must stand still, and say so. No scripted stand-in policy.
    assert d1["source"] == "offline_no_models"
    assert d1["controls"]["active_keys"] == []
    assert not any(d1["controls"][k] for k in ("left", "right", "jump", "use_item", "use_tile", "quick_heal"))
    assert {t["status"] for t in d1["tiers"].values()} == {"OFFLINE"}
    latest = client.get("/api/real/latest").json()["latest"]
    assert latest["state"]["defense"] == 4
    assert latest["state"]["frame_url"] == "real_terraria_live_frames/1.png"
    assert client.get("/play/real_terraria_live_frames/1.png").status_code == 200
    assert client.get("/play/real_terraria_live_frames/999.png").status_code == 404

    # Death on next tick -> failure clip created
    r2 = client.post("/real_game_tick", json=_payload(hp=0, deaths=1))
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


def test_real_game_old_protocol_payload_is_rejected_idle(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    client = TestClient(create_live_app(live_models=False))
    d = client.post("/real_game_tick", json={
        "x": 100, "y": 50, "hp": 100, "ascii_grid": ["." * 61] * 35,
        "nearby_enemies": [{"name": "Green Slime", "dx": 3.0, "dy": -4.0}],
    }).json()
    assert d["source"] == "outdated_mod"
    assert d["controls"]["active_keys"] == []
    assert "OUTDATED MOD" in d["note"]
    assert client.get("/api/real/latest").json()["frames"] == 0


def test_real_game_three_tiers_run_and_no_ground_truth_reaches_models(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    calls = _wire_mock_models(app, {
        "move_x": "right", "jump": 0.9, "down": 0.1, "use_item": 0.95, "interact": 0.0, "grapple": 0.0,
        "quick_heal": 0.0, "smart_cursor": 0.0, "hotbar": "3", "cursor_x": "2", "cursor_y": "-1",
        "hold": "long", "inventory": "keep", "replan": 0.0,
    })
    client = TestClient(app)
    for _ in range(3):
        d = client.post("/real_game_tick", json=_payload()).json()

    # Every tick: vision reads the frame, decision produces the inputs. Supervisor wrote the plan first.
    assert len(calls["vision"]) == 3 and len(calls["decision"]) == 3 and len(calls["supervisor"]) == 1
    assert all(png == PNG_1X1 for _, png in calls["vision"])
    assert d["source"] == "decisions_api"
    assert {t["status"] for t in d["tiers"].values()} == {"LIVE"}
    assert "Craft torches" in d["hud"]["supervisor"]
    assert "pitch_black" in d["hud"]["vision"]

    # Decision answers pass through untouched: no aim assist, no forced heal, no tool override.
    c = d["controls"]
    assert (c["right"], c["left"], c["jump"], c["use_item"], c["quick_heal"], c["smart_cursor"]) == (True, False, True, True, False, False)
    assert (c["hotbar"], c["cursor_dx"], c["cursor_dy"], c["hold_ticks"]) == (3, 2.0, -1.0, 90)

    # No prompt to any tier contains referee data.
    prompts = [p for p, _ in calls["vision"]] + [p for p, _ in calls["supervisor"]] + [ctx for _, ctx, _ in calls["decision"]]
    blob = "\n".join(prompts) + json.dumps([q for q, _, _ in calls["decision"]])
    for secret in ("987654", "123456", "31337", "SENTINEL_BOSS", "SENTINEL_ORE", "referee", "depth_below_surface"):
        assert secret not in blob
    # The decision model is told what vision saw.
    assert "pitch_black" in calls["decision"][0][1]


def test_real_game_crafting_only_through_open_inventory(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    calls = _wire_mock_models(app, {"move_x": "idle", "inventory": "open", "craft": "c1", "hold": "tap"})
    client = TestClient(app)

    closed = client.post("/real_game_tick", json=_payload()).json()
    assert "craft" not in calls["decision"][0][0]
    assert closed["controls"]["inventory"] == "open" and closed["controls"]["craft"] == ""

    opened = client.post("/real_game_tick", json=_payload(
        inventory_open=True, inventory={"Wood": 12, "Gel": 3},
        craftable=[{"name": "Work Bench", "makes": 1}, {"name": "Torch", "makes": 3}],
    )).json()
    assert "craft" in calls["decision"][1][0]
    assert opened["controls"]["craft"] == "Torch"
    assert opened["controls"]["hold_ticks"] == 6


def test_real_game_supervisor_failure_is_surfaced_and_player_idles(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)

    def broken_supervisor(prompt, png):
        attempts.append((prompt, png))
        raise RuntimeError("quota exhausted")

    attempts = []
    calls = _wire_mock_models(app, {"move_x": "right", "use_item": 1.0}, supervisor_fn=broken_supervisor)
    d = TestClient(app).post("/real_game_tick", json=_payload()).json()
    assert d["source"] == "no_supervisor_plan"
    assert d["tiers"]["supervisor"]["status"] == "FAILED"
    assert "quota exhausted" in d["hud"]["supervisor"]
    assert d["controls"]["active_keys"] == []
    assert calls["decision"] == []  # no plan, so the decision tier is never asked to act
    # A failed first plan backs off instead of spending another supervisor call on every poll.
    again = TestClient(app).post("/real_game_tick", json=_payload(tick=2)).json()
    assert again["source"] == "no_supervisor_plan"
    assert len(attempts) == 1


def test_real_game_async_first_plan_does_not_block_decision_or_leak_referee(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    calls = _wire_mock_models(app, {"move_x": "right", "use_item": 1.0, "routine": "none"})
    real = app.state.controller.real
    real.async_supervisor = True
    entered, release = threading.Event(), threading.Event()

    def waiting_supervisor(prompt, png):
        calls["supervisor"].append((prompt, png))
        entered.set()
        release.wait(timeout=2)
        return json.dumps({"strategy": "Wait for a safe moment", "objective": "Stay alive"})

    real.supervisor_fn = waiting_supervisor
    try:
        d = TestClient(app).post("/real_game_tick", json=_payload()).json()
        assert entered.wait(timeout=1)
        assert d["source"] == "decisions_api"
        assert d["plan"] is None
        assert d["controls"]["right"] is True
        assert calls["decision"][0][2] == PNG_1X1  # current frame goes straight to the fast tier
        assert "first plan is still being written" in calls["decision"][0][1]
        prompts = [calls["supervisor"][0][0], calls["decision"][0][1]]
        blob = "\n".join(prompts)
        for secret in ("987654", "123456", "31337", "SENTINEL_BOSS", "SENTINEL_ORE", "referee"):
            assert secret not in blob
    finally:
        release.set()


def test_real_game_async_vision_does_not_block_fast_decision(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    calls = _wire_mock_models(app, {"move_x": "left", "routine": "none"})
    real = app.state.controller.real
    real.async_vision = True
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()

    def waiting_vision(prompt, png):
        calls["vision"].append((prompt, png))
        entered.set()
        try:
            release.wait(timeout=2)
            return json.dumps({"lighting": "bright", "surroundings": "Open surface", "enemies": []})
        finally:
            completed.set()

    real.vision_fn = waiting_vision
    try:
        client = TestClient(app)
        d = client.post("/real_game_tick", json=_payload()).json()
        assert entered.wait(timeout=1)
        assert d["source"] == "decisions_api"
        assert d["controls"]["left"] is True
        assert calls["decision"][0][2] == PNG_1X1
        assert d["vision_report"]["age_s"] is None
        second = client.post("/real_game_tick", json=_payload(tick=2)).json()
        assert second["source"] == "decisions_api"
    finally:
        release.set()
    assert completed.wait(timeout=1)
    assert real.tiers["vision"].as_dict(real.ticks)["last_ok_tick"] == 1


def test_real_game_aim_assist_and_routines_are_offered_only_when_usable():
    hud = _payload()["hud"]
    hud["hotbar"] = [{"slot": 1, "name": "Copper Shortsword", "stack": 1}]
    questions, _, _ = build_questions(hud, builtin_routines(hud))
    assert "aim_assist" not in questions
    assert "routine" not in questions

    hud["visible_enemies"] = [{"name": "Green Slime"}]
    hud["hotbar"].append({"slot": 4, "name": "Torch", "stack": 2})
    questions, _, _ = build_questions(hud, builtin_routines(hud))
    assert "aim_assist" in questions
    assert "routine" in questions
    assert "place_torch" in questions["routine"]["criteria"]


def test_real_game_routine_queues_steps_without_more_decision_calls(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    calls = _wire_mock_models(app, {"move_x": "idle", "routine": "place_torch"})
    client = TestClient(app)
    payload = _payload()
    payload["hud"]["hotbar"].append({"slot": 4, "name": "Torch", "stack": 2})

    responses = [client.post("/real_game_tick", json=payload).json() for _ in range(3)]
    assert all(r["source"] == "decisions_api_routine" for r in responses)
    assert len(calls["decision"]) == 1
    assert "routine place_torch step 1/3" in responses[0]["note"]
    assert "routine place_torch step 3/3" in responses[2]["note"]


def test_real_game_routine_aborts_after_damage(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    answers = {"move_x": "idle", "routine": "pillar_up"}
    calls = _wire_mock_models(app, answers)
    client = TestClient(app)
    payload = _payload()
    payload["hud"]["hotbar"].append({"slot": 4, "name": "Dirt Block", "stack": 20})

    started = client.post("/real_game_tick", json=payload).json()
    assert started["source"] == "decisions_api_routine"
    answers["routine"] = "none"
    hurt = client.post("/real_game_tick", json=_payload(hp=89, hotbar=payload["hud"]["hotbar"])).json()
    assert hurt["source"] == "decisions_api"
    assert len(calls["decision"]) == 2
    assert any("routine pillar_up ABORTED" in item for item in app.state.controller.real._history)
    assert not app.state.controller.real._queue


def test_real_game_repeated_ineffective_input_requests_replan(tmp_path: Path, monkeypatch):
    import terraria_bench.live_play as lp

    monkeypatch.setattr(lp, "RECORDINGS_DIR", tmp_path)
    app = create_live_app(live_models=False)
    calls = _wire_mock_models(app, {"move_x": "right", "routine": "none"})
    client = TestClient(app)

    for _ in range(6):
        response = client.post("/real_game_tick", json=_payload()).json()
        assert response["source"] == "decisions_api"

    assert len(calls["decision"]) == 6
    assert len(calls["supervisor"]) == 2
    assert "repeated the same input 5 times" in calls["supervisor"][1][0]
