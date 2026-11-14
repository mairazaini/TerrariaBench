"""Pure-Python game bridge server (port 8765), human playthrough recorder, and replayer.

Replaces external C#/.NET runtime dependencies for evaluation and testing:
all tick metering, action logging, and byte-identical trajectory replay execute
in Python over ``Episode``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

from .grader import components, grade
from .provenance import (
    ARTIFACTS_DIR,
    CALIBRATION_PATH,
    ProvenanceError,
    load_sealed,
    seal,
    verify_seal,
    write_sealed,
)
from .tasks import TASKS
from .tools import TOOL_NAMES, TOOL_SCHEMAS, Episode

HUMAN_RUNS_DIR = ARTIFACTS_DIR / "runs" / "human"


def _write_trajectory_files(
    jsonl_path: Path,
    steps: list[dict[str, Any]],
    ledger: dict[str, Any],
) -> tuple[Path, Path]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in steps]
    lines.append(json.dumps({"type": "ledger", "task": ledger["task"], "seed": ledger["seed"], "ledger": ledger}, sort_keys=True))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger_path = jsonl_path.with_suffix(".json")
    write_sealed(ledger_path, ledger)
    return jsonl_path, ledger_path


def record_trajectory(
    task_id: str,
    seed: int,
    actions: list[tuple[str, dict[str, Any]]],
    out_path: str | Path | None = None,
    calibration: dict | None = None,
) -> dict[str, Any]:
    """Execute ``actions`` on a fresh Episode and write a sealed ``.jsonl`` + ``.json`` trajectory."""
    ep = Episode(task_id, seed)
    steps: list[dict[str, Any]] = []
    for idx, (tool, args) in enumerate(actions, 1):
        if ep.done:
            break
        step_args = dict(args or {})
        win_before = ep.world.window
        res = ep.step(tool, **step_args)
        steps.append({
            "type": "step",
            "step": idx,
            "window": win_before,
            "tool": tool,
            "args": step_args,
            "header": res["header"],
            "phi": ep.world.phi(),
        })
    ledger = ep.finalize(calibration)
    jsonl_path = Path(out_path) if out_path else (HUMAN_RUNS_DIR / f"{ep.task.id}_seed{ep.seed}.jsonl")
    j_path, l_path = _write_trajectory_files(jsonl_path, steps, ledger)
    return {
        "jsonl_path": str(j_path),
        "ledger_path": str(l_path),
        "steps": len(steps),
        "ledger": ledger,
    }


def _default_demo_actions(task_id: str) -> list[tuple[str, dict[str, Any]]]:
    task = TASKS[task_id]
    actions: list[tuple[str, dict[str, Any]]] = [
        ("chop_trees", {"radius": 35, "max_trees": 4}),
        ("build_blueprint", {"preset": "shelter"}),
        ("craft_item", {"item_id": "wooden_sword"}),
        ("craft_item", {"item_id": "wood_armor"}),
        ("equip_item", {"item_id": "wooden_sword"}),
        ("equip_item", {"item_id": "wood_armor"}),
    ]
    for _ in range(task.windows):
        actions.append(("end_window", {}))
    return actions


def record_from_bridge(
    task_id: str = "first-night",
    seed: int = 7,
    host: str = "127.0.0.1",
    port: int = 8765,
    out_path: str | Path | None = None,
    actions: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Record a bridge or demonstration action sequence onto a graded Episode ledger."""
    seq = actions if actions is not None else _default_demo_actions(task_id)
    res = record_trajectory(task_id, seed, seq, out_path=out_path)
    res["bridge"] = f"{host}:{port}"
    return res


def _load_calibration_for_replay(calibration: dict | None) -> dict | None:
    if calibration is not None:
        return calibration
    if CALIBRATION_PATH.exists():
        try:
            return load_sealed(CALIBRATION_PATH)
        except ProvenanceError:
            return None
    return None


def replay_trajectory(jsonl_path: str | Path, calibration: dict | None = None) -> dict[str, Any]:
    """Replay a recorded ``.jsonl`` trajectory and verify byte-identical ledger reproduction."""
    path = Path(jsonl_path)
    if not path.exists():
        raise ProvenanceError(f"missing trajectory: {path}")

    steps: list[dict[str, Any]] = []
    recorded_ledger: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "ledger" and isinstance(obj.get("ledger"), dict):
            recorded_ledger = obj["ledger"]
        elif obj.get("type") == "step" or ("tool" in obj and obj.get("type") != "ledger"):
            steps.append(obj)

    if recorded_ledger is None:
        companion = path.with_suffix(".json")
        if companion.exists():
            recorded_ledger = load_sealed(companion)
    if not isinstance(recorded_ledger, dict) or not verify_seal(recorded_ledger):
        raise ProvenanceError(f"recorded ledger seal is invalid for {path}")

    ep = Episode(recorded_ledger["task"], recorded_ledger["seed"])
    for step in steps:
        if ep.done:
            break
        ep.step(step["tool"], **(step.get("args") or {}))

    replayed = ep.finalize(calibration)
    if (
        replayed["state_hash"] != recorded_ledger["state_hash"]
        or replayed["phi_final"] != recorded_ledger["phi_final"]
        or replayed["windows"] != recorded_ledger["windows"]
    ):
        raise ProvenanceError(f"replay mismatch against recorded ledger in {path}")
    if seal({**replayed, "manifest": recorded_ledger["manifest"]})["sha256"] != recorded_ledger["sha256"]:
        raise ProvenanceError(f"ledger sha256 mismatch on replay for {path}")

    cal = _load_calibration_for_replay(calibration)
    cell = (cal or {}).get("cells", {}).get(f"{ep.task.id}:{ep.seed}")
    metrics = grade(replayed, cell) if cell else components(replayed)
    return {
        "verified": True,
        "task": ep.task.id,
        "seed": ep.seed,
        "state_hash": replayed["state_hash"],
        "ledger_sha256": recorded_ledger["sha256"],
        "metrics": metrics,
        "ledger": replayed,
    }


class BridgeSession:
    """Stateful JSON-RPC 2.0 bridge session backed by a deterministic Episode."""

    def __init__(self, task_id: str = "first-night", seed: int = 7, redact_seed: bool = False):
        self.redact_seed = redact_seed
        self.reset(task_id, seed, redact_seed=redact_seed)

    def reset(self, task_id: str, seed: int, redact_seed: bool | None = None) -> None:
        if redact_seed is not None:
            self.redact_seed = redact_seed
        self.ep = Episode(task_id, int(seed), redact_seed=self.redact_seed)
        self.steps: list[dict[str, Any]] = []
        self.saved: dict[str, Any] | None = None

    def handle_rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        req_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params") or {}

        if method == "initialize":
            task_id = params.get("task", self.ep.task.id)
            seed = int(params.get("seed", self.ep.seed))
            redact = bool(params.get("redact_seed", self.redact_seed))
            self.reset(task_id, seed, redact_seed=redact)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "terraria-bridge", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                    "task": self.ep.task.id,
                    "seed": self.ep.seed,
                    "header": self.ep.header(),
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [{"name": name, **TOOL_SCHEMAS[name]} for name in TOOL_NAMES],
                },
            }

        if method == "tools/call":
            name = params.get("name", "")
            args = dict(params.get("arguments") or {})
            was_done = self.ep.done
            win_before = self.ep.world.window
            res = self.ep.step(name, **args)
            canonical_name = "engage_combat_burst" if name == "engage_combat_ burst" else name
            if not was_done and canonical_name in TOOL_SCHEMAS:
                self.steps.append({
                    "type": "step",
                    "step": len(self.steps) + 1,
                    "window": win_before,
                    "tool": canonical_name,
                    "args": args,
                    "header": res["header"],
                    "phi": self.ep.world.phi(),
                })
            return {"jsonrpc": "2.0", "id": req_id, "result": res}

        if method == "episode/finalize":
            out_path = params.get("out_path")
            calibration = params.get("calibration")
            ledger = self.ep.finalize(calibration)
            jsonl_path = Path(out_path) if out_path else (HUMAN_RUNS_DIR / f"{self.ep.task.id}_seed{self.ep.seed}.jsonl")
            j_path, l_path = _write_trajectory_files(jsonl_path, self.steps, ledger)
            self.saved = {
                "jsonl_path": str(j_path),
                "ledger_path": str(l_path),
                "steps": len(self.steps),
                "ledger": ledger,
            }
            return {"jsonrpc": "2.0", "id": req_id, "result": self.saved}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def create_bridge_app(task_id: str = "first-night", seed: int = 7) -> FastAPI:
    """Create the FastAPI JSON-RPC bridge server for port 8765."""
    bridge = BridgeSession(task_id=task_id, seed=seed)
    app = FastAPI(title="TerrariaBench Python Bridge (Port 8765)")
    app.state.bridge = bridge

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return {
            "ok": True,
            "task": bridge.ep.task.id,
            "seed": bridge.ep.seed,
            "window": bridge.ep.world.window,
            "total_windows": bridge.ep.task.windows,
            "ops_left": bridge.ep.ops_left,
            "done": bridge.ep.done,
            "header": bridge.ep.header(),
            "steps": len(bridge.steps),
        }

    @app.post("/")
    async def rpc_endpoint(request: Request) -> Any:
        payload = await request.json()
        if isinstance(payload, list):
            return [bridge.handle_rpc(item) for item in payload]
        return bridge.handle_rpc(payload)

    return app


def verify_all_trajectories(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Replay and verify all ``.jsonl`` trajectories under ``directory`` (default ``HUMAN_RUNS_DIR``)."""
    root = Path(directory) if directory else HUMAN_RUNS_DIR
    if not root.exists():
        return []
    return [replay_trajectory(p) for p in sorted(root.glob("*.jsonl"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerrariaBench pure-Python bridge, human recorder, and replayer")
    parser.add_argument("--task", default="first-night", choices=sorted(TASKS), help="Task ID")
    parser.add_argument("--seed", type=int, default=7, help="World seed")
    parser.add_argument("--serve", action="store_true", help="Serve the JSON-RPC bridge on --host:--port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for --serve")
    parser.add_argument("--port", type=int, default=8765, help="Bind port for --serve")
    parser.add_argument("--record-demo", action="store_true", help="Record a demonstration trajectory and sealed ledger")
    parser.add_argument("--out", default=None, help="Output .jsonl path for --record-demo")
    parser.add_argument("--replay", default=None, help="Replay and verify a .jsonl trajectory file")
    parser.add_argument("--verify-all", action="store_true", help="Verify all recorded human trajectories")
    args = parser.parse_args(argv)

    if args.serve:
        import uvicorn

        uvicorn.run(create_bridge_app(args.task, args.seed), host=args.host, port=args.port)
        return 0

    if args.record_demo:
        res = record_from_bridge(args.task, args.seed, host=args.host, port=args.port, out_path=args.out)
        print(json.dumps({
            "ok": True,
            "jsonl_path": res["jsonl_path"],
            "ledger_path": res["ledger_path"],
            "steps": res["steps"],
            "state_hash": res["ledger"]["state_hash"],
            "sha256": res["ledger"]["sha256"],
            "phi_final": res["ledger"]["phi_final"],
        }, indent=2))
        return 0

    if args.replay:
        res = replay_trajectory(args.replay)
        print(json.dumps({
            "verified": res["verified"],
            "task": res["task"],
            "seed": res["seed"],
            "state_hash": res["state_hash"],
            "ledger_sha256": res["ledger_sha256"],
            "metrics": res["metrics"],
        }, indent=2))
        return 0

    if args.verify_all:
        results = verify_all_trajectories()
        summary = [
            {
                "task": r["task"],
                "seed": r["seed"],
                "verified": r["verified"],
                "ledger_sha256": r["ledger_sha256"],
                "delta_net": r["metrics"].get("delta_net"),
            }
            for r in results
        ]
        print(json.dumps({"verified_count": len(summary), "trajectories": summary}, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
