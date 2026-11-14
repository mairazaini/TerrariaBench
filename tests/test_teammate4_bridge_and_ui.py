"""Tests for Teammate 4 deliverables: pure-Python bridge, human recorder/replayer,
StoreBench §5 operator console, and FastAPI leaderboard dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.app import app as dashboard_app
from terraria_bench import (
    EXPLOITS,
    TASKS,
    TOOL_NAMES,
    ProvenanceError,
    create_bridge_app,
    create_console_app,
    record_from_bridge,
    record_trajectory,
    replay_trajectory,
)
from terraria_bench.provenance import verify_seal


def test_record_and_byte_identical_replay_and_tamper_rejection(tmp_path: Path):
    out_jsonl = tmp_path / "first_night_seed7.jsonl"
    recorded = record_from_bridge("first-night", 7, out_path=out_jsonl)
    ledger = recorded["ledger"]
    assert out_jsonl.exists()
    assert out_jsonl.with_suffix(".json").exists()
    assert verify_seal(ledger)
    assert recorded["steps"] >= 6

    replayed = replay_trajectory(out_jsonl)
    assert replayed["verified"] is True
    assert replayed["state_hash"] == ledger["state_hash"]
    assert replayed["ledger_sha256"] == ledger["sha256"]

    # Tamper with companion ledger and inline ledger -> must fail closed
    tampered = dict(ledger)
    tampered["phi_final"] = float(ledger["phi_final"]) + 999.0
    tampered_jsonl = tmp_path / "tampered.jsonl"
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    lines[-1] = json.dumps({"type": "ledger", "task": "first-night", "seed": 7, "ledger": tampered})
    tampered_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ProvenanceError):
        replay_trajectory(tampered_jsonl)


def test_bridge_jsonrpc_server_end_to_end(tmp_path: Path):
    client = TestClient(create_bridge_app("first-night", 7))

    init_resp = client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"task": "first-night", "seed": 7}})
    assert init_resp.status_code == 200
    assert init_resp.json()["result"]["task"] == "first-night"

    list_resp = client.post("/", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = list_resp.json()["result"]["tools"]
    assert len(tools) == 24
    assert [t["name"] for t in tools] == list(TOOL_NAMES)

    # terraria_status is free (0 ops)
    st_before = client.get("/status").json()
    ops_start = st_before["ops_left"]
    r_free = client.post("/", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "terraria_status", "arguments": {}}})
    assert r_free.json()["result"]["ok"] is True
    assert client.get("/status").json()["ops_left"] == ops_start

    # chop_trees costs 1 op
    r_chop = client.post("/", json={"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "chop_trees", "arguments": {"radius": 35, "max_trees": 3}}})
    assert r_chop.json()["result"]["ok"] is True
    assert client.get("/status").json()["ops_left"] == ops_start - 1

    # finalize writes replayable sealed trajectory
    out_jsonl = tmp_path / "bridge_run.jsonl"
    fin = client.post("/", json={"jsonrpc": "2.0", "id": 5, "method": "episode/finalize", "params": {"out_path": str(out_jsonl)}})
    fin_res = fin.json()["result"]
    assert fin_res["steps"] == 2
    assert verify_seal(fin_res["ledger"])
    assert replay_trajectory(out_jsonl)["verified"] is True


def test_web_console_no_sla_leak_and_saves_on_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import terraria_bench.web_console as wc

    monkeypatch.setattr(wc, "HUMAN_RUNS_DIR", tmp_path)
    app = create_console_app("first-night", 7)
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert TASKS["first-night"].brief in home.text
    assert "World: locked" in home.text  # seed redacted
    assert "tier:1" not in home.text      # hidden SLA must never leak

    api_state = client.get("/api/state").json()
    assert api_state["task_id"] == "first-night"
    assert len(api_state["tools"]) == 24
    assert "sla" not in api_state
    assert "events" not in api_state

    # Play through all windows via form posts until done
    for _ in range(TASKS["first-night"].windows):
        resp = client.post("/step", data={"tool": "end_window", "args_json": "{}"})
        assert resp.status_code == 200

    final_state = client.get("/api/state").json()
    assert final_state["done"] is True
    assert final_state["saved"] is not None
    saved_jsonl = Path(final_state["saved"]["jsonl_path"])
    assert saved_jsonl.exists()
    assert replay_trajectory(saved_jsonl)["verified"] is True


def test_dashboard_endpoints_and_exploits_catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import dashboard.app as dash_mod

    human_dir = tmp_path / "runs" / "human"
    record_trajectory(
        "first-night",
        7,
        [("chop_trees", {"radius": 35, "max_trees": 3}), ("end_window", {}), ("end_window", {}), ("end_window", {}), ("end_window", {})],
        out_path=human_dir / "first-night_seed7.jsonl",
    )
    monkeypatch.setattr(dash_mod, "RUNS_DIR", tmp_path / "runs")

    client = TestClient(dashboard_app)
    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "TerrariaBench Leaderboard" in html_resp.text
    assert "<svg" in html_resp.text

    lb = client.get("/api/leaderboard").json()
    assert lb["ok"] is True
    assert any(row["name"] == "do-nothing" and row["composite"] == 0.0 for row in lb["table1"])
    assert any(row["name"] == "absentee" and row["composite"] == 0.1 for row in lb["table1"])
    assert len(lb["table3"]) == 2
    assert len(lb["trajectories"]) == 1

    cal = client.get("/api/calibration").json()
    assert cal["ok"] is True

    exp = client.get("/api/exploits").json()
    assert exp["ok"] is True
    assert exp["count"] == len(EXPLOITS) == 11
