"""Pure-Python FastAPI Live Leaderboard & Trajectory Audit Console.

Renders StoreBench Table 1, Table 3, per-(task, seed) calibration gates,
recorded human/agent trajectories with SVG Phi(s_t) sparklines, and the
11 closed reward exploits without any external frontend build toolchain.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from terraria_bench.grader import EXPLOITS, components, grade
from terraria_bench.provenance import (
    ARTIFACTS_DIR,
    CALIBRATION_PATH,
    ProvenanceError,
    load_sealed,
    verify_seal,
)
from terraria_bench.tasks import EVAL_TASKS, REPORTED_SEEDS
from terraria_bench.tools import Episode

ANCHOR_ORDER = ("smart-triage", "rule-based", "blast-list", "absentee", "do-nothing", "exploiter")
RUNS_DIR = ARTIFACTS_DIR / "runs"

app = FastAPI(title="TerrariaBench Leaderboard & Audit Console")


def _load_calibration_state() -> dict[str, Any]:
    if not CALIBRATION_PATH.exists():
        return {"exists": False, "valid": False, "error": "calibration.json not generated yet", "calibration": None}
    try:
        cal = load_sealed(CALIBRATION_PATH)
        return {"exists": True, "valid": True, "error": "", "calibration": cal}
    except ProvenanceError as exc:
        return {"exists": True, "valid": False, "error": str(exc), "calibration": None}


def _extract_ledger(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("ledger"), dict):
        obj = obj["ledger"]
    if isinstance(obj, dict) and verify_seal(obj) and "task" in obj and "windows" in obj:
        return obj
    return None


def _scan_agent_runs(calibration: dict[str, Any] | None) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_agent: dict[str, list[dict[str, Any]]] = {}
    trajectories: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return by_agent, trajectories

    cells = (calibration or {}).get("cells", {})
    for agent_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        agent_name = agent_dir.name
        rows: list[dict[str, Any]] = []
        for fpath in sorted(agent_dir.glob("*.json")):
            ledger = _extract_ledger(fpath)
            if ledger is None:
                continue
            cell_key = f"{ledger['task']}:{ledger['seed']}"
            cell = cells.get(cell_key)
            if cell and cell.get("s", 0) > 0:
                scored = grade(ledger, cell)
            else:
                comp = components(ledger)
                scored = {
                    "composite": None,
                    "business": None,
                    "reliability": round(0.5 * comp["survival"] + 0.5 * comp["sla"], 4),
                    "continuity": round(comp["continuity"], 4),
                    "survival": round(comp["survival"], 4),
                    "sla": round(comp["sla"], 4),
                    "windows": comp["windows"],
                    "delta_net": comp["delta_net"],
                    "deaths": comp["deaths"],
                    "valid": True,
                    "passed": None,
                }
            rows.append(scored)
            trajectories.append({
                "agent": agent_name,
                "file": fpath.name,
                "task": ledger["task"],
                "seed": ledger["seed"],
                "sha256": ledger["sha256"],
                "phi0": ledger["phi0"],
                "phi_final": ledger["phi_final"],
                "delta_net": scored["delta_net"],
                "composite": scored["composite"],
                "windows": ledger["windows"],
            })
        if rows:
            by_agent[agent_name] = rows
    return by_agent, trajectories


def _summarize_runs(name: str, kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    composites = [r["composite"] for r in rows if r.get("composite") is not None]
    businesses = [r["business"] for r in rows if r.get("business") is not None]
    reliabilities = [r["reliability"] for r in rows if r.get("reliability") is not None]
    continuities = [r["continuity"] for r in rows if r.get("continuity") is not None]
    survivals = [r["survival"] for r in rows if r.get("survival") is not None]
    slas = [r["sla"] for r in rows if r.get("sla") is not None]
    passed = [r["passed"] for r in rows if r.get("passed") is not None]
    wins = [r["windows"] for r in rows if r.get("windows") is not None]

    def avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "name": name,
        "kind": kind,
        "status": "evaluated" if rows else "not run",
        "episodes": len(rows),
        "composite": avg(composites),
        "pass_rate": round(sum(1 for p in passed if p) / len(passed), 4) if passed else None,
        "business": avg(businesses),
        "reliability": avg(reliabilities),
        "continuity": avg(continuities),
        "survival": avg(survivals),
        "sla": avg(slas),
        "windows": avg(wins),
    }


def _baseline_reference_rows() -> dict[str, dict[str, Any]]:
    """Deterministic uncalibrated reference rows for do-nothing and absentee."""
    dn_ep = Episode(EVAL_TASKS[0], REPORTED_SEEDS[0])
    dn_ledger = dn_ep.finalize()
    ab_ep = Episode(EVAL_TASKS[0], REPORTED_SEEDS[0])
    for _ in range(ab_ep.task.windows):
        ab_ep.step("terraria_status")
        ab_ep.step("end_window")
    ab_ledger = ab_ep.finalize()
    dn_comp = components(dn_ledger)
    ab_comp = components(ab_ledger)
    return {
        "do-nothing": {
            "name": "do-nothing",
            "kind": "anchor",
            "status": "reference (uncalibrated)",
            "episodes": 1,
            "composite": 0.0,
            "pass_rate": 0.0,
            "business": 0.0,
            "reliability": 0.0,
            "continuity": round(dn_comp["continuity"], 4),
        },
        "absentee": {
            "name": "absentee",
            "kind": "anchor",
            "status": "reference (uncalibrated)",
            "episodes": 1,
            "composite": 0.1,
            "pass_rate": 0.0,
            "business": 0.0,
            "reliability": 0.0,
            "continuity": round(ab_comp["continuity"], 4),
        },
    }


def _build_leaderboard() -> dict[str, Any]:
    cal_state = _load_calibration_state()
    cal = cal_state["calibration"] if cal_state["valid"] else None
    by_agent, trajectories = _scan_agent_runs(cal)

    table1: list[dict[str, Any]] = []
    if cal and isinstance(cal.get("summary"), dict):
        summary = cal["summary"]
        for anchor in ANCHOR_ORDER:
            info = summary.get(anchor)
            if info:
                table1.append({
                    "name": anchor,
                    "kind": "anchor",
                    "status": "calibrated",
                    "episodes": info.get("episodes", len(cal.get("cells", {}))),
                    "composite": info.get("mean_composite"),
                    "pass_rate": info.get("pass_rate"),
                    "business": info.get("business"),
                    "reliability": info.get("reliability"),
                    "continuity": info.get("continuity"),
                })
    else:
        refs = _baseline_reference_rows()
        for anchor in ANCHOR_ORDER:
            if anchor in refs:
                table1.append(refs[anchor])
            else:
                table1.append({
                    "name": anchor,
                    "kind": "anchor",
                    "status": "pending calibration",
                    "episodes": 0,
                    "composite": None,
                    "pass_rate": None,
                    "business": None,
                    "reliability": None,
                    "continuity": None,
                })

    expected_models = ("human", "gemini", "nemotron", "gemma_base", "gemma_tuned")
    seen_agents = set()
    for agent in expected_models:
        seen_agents.add(agent)
        table1.append(_summarize_runs(agent, "human" if agent == "human" else "agent", by_agent.get(agent, [])))
    for extra in sorted(by_agent):
        if extra not in seen_agents:
            table1.append(_summarize_runs(extra, "agent", by_agent[extra]))

    table3 = [
        _summarize_runs("gemma_base", "post-train", by_agent.get("gemma_base", [])),
        _summarize_runs("gemma_tuned", "post-train", by_agent.get("gemma_tuned", [])),
    ]

    return {
        "ok": True,
        "calibrated": bool(cal),
        "calibration_error": cal_state["error"],
        "table1": table1,
        "table3": table3,
        "trajectories": trajectories,
    }


def _phi_sparkline_svg(windows: list[dict[str, Any]], phi0: float) -> str:
    vals = [float(phi0)] + [float(w.get("phi", phi0)) for w in windows]
    lo, hi = min(vals), max(vals)
    span = max(1e-6, hi - lo)
    w_px, h_px = 180, 34
    pts = []
    denom = max(1, len(vals) - 1)
    for i, v in enumerate(vals):
        x = round(4 + (w_px - 8) * i / denom, 1)
        y = round((h_px - 4) - (h_px - 8) * (v - lo) / span, 1)
        pts.append(f"{x},{y}")
    return (
        f'<svg width="{w_px}" height="{h_px}" viewBox="0 0 {w_px} {h_px}" '
        f'style="background:#020617;border:1px solid #334155;border-radius:4px;">'
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{" ".join(pts)}" /></svg>'
    )


def _fmt(val: Any, pct: bool = False) -> str:
    if val is None:
        return "—"
    if pct:
        return f"{val * 100:.1f}%"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


@app.get("/", response_class=HTMLResponse)
async def dashboard_home() -> HTMLResponse:
    lb = _build_leaderboard()
    cal_state = _load_calibration_state()
    cal = cal_state["calibration"]

    t1_rows = "".join(
        f"<tr><td><strong>{html.escape(r['name'])}</strong></td><td>{html.escape(r['kind'])}</td>"
        f"<td>{html.escape(r['status'])}</td><td>{r['episodes']}</td>"
        f"<td>{_fmt(r['composite'])}</td><td>{_fmt(r['pass_rate'], pct=True)}</td>"
        f"<td>{_fmt(r['business'])}</td><td>{_fmt(r['reliability'])}</td><td>{_fmt(r['continuity'])}</td></tr>"
        for r in lb["table1"]
    )
    t3_rows = "".join(
        f"<tr><td><strong>{html.escape(r['name'])}</strong></td><td>{html.escape(r['status'])}</td>"
        f"<td>{r['episodes']}</td><td>{_fmt(r['composite'])}</td><td>{_fmt(r['pass_rate'], pct=True)}</td>"
        f"<td>{_fmt(r['business'])}</td><td>{_fmt(r['reliability'])}</td><td>{_fmt(r['continuity'])}</td>"
        f"<td>{_fmt(r.get('survival'))}</td><td>{_fmt(r.get('sla'))}</td><td>{_fmt(r.get('windows'))}</td></tr>"
        for r in lb["table3"]
    )

    cells_html = "<p><em>Run <code>python -m terraria_bench.calibrate</code> to populate sealed calibration cells.</em></p>"
    if cal and isinstance(cal.get("cells"), dict):
        cell_rows = "".join(
            f"<tr><td>{html.escape(str(c.get('task')))}</td><td>{c.get('seed')}</td>"
            f"<td>{_fmt(c.get('z'))}</td><td>{_fmt(c.get('s'))}</td><td>{_fmt(c.get('threshold'))}</td>"
            f"<td>{'PASS' if c.get('gates_passed') else 'FAIL'}</td></tr>"
            for _, c in sorted(cal["cells"].items())
        )
        cells_html = f"<table><tr><th>Task</th><th>Seed</th><th>z</th><th>s</th><th>Threshold</th><th>Gates</th></tr>{cell_rows}</table>"

    traj_rows = ""
    if lb["trajectories"]:
        traj_rows = "".join(
            f"<tr><td>{html.escape(t['agent'])}</td><td>{html.escape(t['task'])}</td><td>{t['seed']}</td>"
            f"<td>{t['delta_net']:.2f}</td><td>{_fmt(t['composite'])}</td>"
            f"<td>{_phi_sparkline_svg(t['windows'], t['phi0'])}</td>"
            f"<td><code>{html.escape(t['sha256'][:14])}...</code></td></tr>"
            for t in lb["trajectories"]
        )
    else:
        traj_rows = '<tr><td colspan="7"><em>No recorded trajectories found under artifacts/runs/ yet.</em></td></tr>'

    exploit_rows = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td><td>{html.escape(attack)}</td><td>{html.escape(fix)}</td></tr>"
        for name, attack, fix in EXPLOITS
    )

    cal_badge = "SEALED &amp; VALID" if lb["calibrated"] else f"UNAUTHENTICATED / PENDING ({html.escape(lb['calibration_error'] or 'run calibrate')})"
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>TerrariaBench Leaderboard &amp; Audit Console</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; background:#0f172a; color:#e2e8f0; margin:24px; line-height:1.45; }}
.card {{ background:#1e293b; border:1px solid #334155; border-radius:8px; padding:16px; margin-bottom:20px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #334155; }}
th {{ color:#94a3b8; font-weight:600; }}
code {{ background:#020617; padding:2px 6px; border-radius:4px; color:#38bdf8; }}
.badge {{ display:inline-block; padding:4px 10px; border-radius:999px; background:#0369a1; font-size:12px; font-weight:700; }}
</style>
</head>
<body>
<h1>TerrariaBench Leaderboard &amp; Audit Console</h1>
<p><span class="badge">Calibration: {cal_badge}</span> &nbsp; <a href="/api/leaderboard" style="color:#38bdf8;">/api/leaderboard</a> &middot; <a href="/api/calibration" style="color:#38bdf8;">/api/calibration</a> &middot; <a href="/api/exploits" style="color:#38bdf8;">/api/exploits</a></p>

<div class="card">
  <h2>StoreBench Table 1 &mdash; Main Evaluation Leaderboard</h2>
  <table>
    <tr><th>Policy / Model</th><th>Kind</th><th>Status</th><th>Episodes</th><th>Composite</th><th>Pass Rate</th><th>Business (0.7)</th><th>Reliab. (0.2)</th><th>Contin. (0.1)</th></tr>
    {t1_rows}
  </table>
</div>

<div class="card">
  <h2>StoreBench Table 3 &mdash; Gemma Post-Training Transfer</h2>
  <table>
    <tr><th>Policy</th><th>Status</th><th>Episodes</th><th>Composite</th><th>Pass Rate</th><th>Business</th><th>Reliab.</th><th>Contin.</th><th>Survival</th><th>SLA</th><th>Windows</th></tr>
    {t3_rows}
  </table>
</div>

<div class="card">
  <h2>Recorded Human &amp; Agent Trajectories (&Phi;(s<sub>t</sub>) Progression)</h2>
  <table>
    <tr><th>Operator</th><th>Task</th><th>Seed</th><th>&Delta;Net</th><th>Composite</th><th>&Phi;(s<sub>t</sub>) Sparkline</th><th>Ledger Seal</th></tr>
    {traj_rows}
  </table>
</div>

<div class="card">
  <h2>Calibration Matrix &amp; Gates (i)&ndash;(iv)</h2>
  {cells_html}
</div>

<div class="card">
  <h2>Closed Reward Exploit Catalogue ({len(EXPLOITS)} Hardened Invariants)</h2>
  <table>
    <tr><th>Exploit ID</th><th>Attack Vector</th><th>Closed Structural Defense</th></tr>
    {exploit_rows}
  </table>
</div>
</body>
</html>"""
    return HTMLResponse(body)


@app.get("/api/leaderboard")
async def api_leaderboard() -> dict[str, Any]:
    return _build_leaderboard()


@app.get("/api/calibration")
async def api_calibration() -> dict[str, Any]:
    state = _load_calibration_state()
    return {"ok": True, **state}


@app.get("/api/exploits")
async def api_exploits() -> dict[str, Any]:
    return {
        "ok": True,
        "count": len(EXPLOITS),
        "exploits": [{"id": name, "attack": attack, "mitigation": fix} for name, attack, fix in EXPLOITS],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
