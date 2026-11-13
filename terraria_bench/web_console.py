"""Pure-Python Human Operator Console (StoreBench §5).

Humans operate the environment through the exact same 24-tool surface,
windowed operation budgets, and task briefings as LLM agents, with zero
privileged information (hidden SLAs and scripted event parameters never leak).
Provides both a FastAPI server-rendered web UI and an interactive CLI.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .engine import H, TILE_CHARS, W
from .human_recorder import HUMAN_RUNS_DIR, _write_trajectory_files
from .tasks import TASKS
from .tools import TOOL_NAMES, TOOL_SCHEMAS, Episode, _encode_png


def _readonly_viewport(ep: Episode, width: int = 61, height: int = 35) -> dict[str, Any]:
    """Render the current 61x35 viewport without spending ops or mutating world.known."""
    w = ep.world
    px, py = w.pos
    hw, hh = width // 2, height // 2
    ascii_rows: list[str] = []
    idx_rows: list[bytes] = []
    for dy in range(-hh, hh + 1):
        y = min(H - 1, max(0, py + dy))
        row = w.g[y]
        chars = []
        idxs = bytearray()
        for dx in range(-hw, hw + 1):
            x = min(W - 1, max(0, px + dx))
            if dx == 0 and dy == 0:
                chars.append("@")
                idxs.append(21)
            else:
                t = row[x]
                chars.append(TILE_CHARS[t])
                idxs.append(t)
        ascii_rows.append("".join(chars))
        idx_rows.append(bytes(idxs))
    return {
        "ascii_grid": ascii_rows,
        "viewport_png_b64": _encode_png(idx_rows, width, height),
    }


class ConsoleState:
    """In-memory operator session backed by an Episode with seed redaction."""

    def __init__(self, task_id: str = "first-night", seed: int = 7, redact_seed: bool = True):
        self.task_id = task_id
        self.seed = int(seed)
        self.redact_seed = redact_seed
        self.reset()

    def reset(self, task_id: str | None = None, seed: int | None = None) -> None:
        if task_id is not None:
            self.task_id = task_id
        if seed is not None:
            self.seed = int(seed)
        self.ep = Episode(self.task_id, self.seed, redact_seed=self.redact_seed)
        self.history: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.last_viewport: dict[str, Any] | None = None
        self.saved_paths: dict[str, str] | None = None
        self.last_error: str = ""

    def step(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        clean_args = dict(args or {})
        win_before = self.ep.world.window
        was_done = self.ep.done
        res = self.ep.step(tool, **clean_args)
        if not was_done and (tool in TOOL_SCHEMAS or tool == "engage_combat_ burst"):
            canonical_tool = "engage_combat_burst" if tool == "engage_combat_ burst" else tool
            self.history.append((canonical_tool, clean_args, {"window": win_before, "phi": self.ep.world.phi(), **res}))
        if tool == "observe_viewport" and res.get("ok"):
            self.last_viewport = res
        if self.ep.done and self.saved_paths is None:
            self.persist()
        return res

    def persist(self, out_dir: Path | None = None) -> dict[str, str]:
        if self.saved_paths is not None:
            return self.saved_paths
        ledger = self.ep.finalize()
        steps = [
            {
                "type": "step",
                "step": idx,
                "window": rec.get("window", 1),
                "tool": tool,
                "args": args,
                "header": rec.get("header", ""),
                "phi": rec.get("phi", 0.0),
            }
            for idx, (tool, args, rec) in enumerate(self.history, 1)
        ]
        base = out_dir or HUMAN_RUNS_DIR
        j_path, l_path = _write_trajectory_files(base / f"{self.task_id}_seed{self.seed}.jsonl", steps, ledger)
        self.saved_paths = {"jsonl_path": str(j_path), "ledger_path": str(l_path), "sha256": ledger["sha256"]}
        return self.saved_paths

    def public_state(self) -> dict[str, Any]:
        w = self.ep.world
        return {
            "header": self.ep.header(),
            "task_id": self.ep.task.id,
            "title": self.ep.task.title,
            "prompt": self.ep.task.prompt,
            "window": w.window,
            "total_windows": self.ep.task.windows,
            "ops_left": self.ep.ops_left,
            "ops_budget": self.ep.window_budget,
            "done": self.ep.done,
            "inventory": dict(sorted(w.inv.items())),
            "equipped": dict(w.eq),
            "tools": TOOL_SCHEMAS,
            "saved": self.saved_paths,
        }


def _render_console_html(state: ConsoleState) -> str:
    ep = state.ep
    vp = state.last_viewport or _readonly_viewport(ep)
    ascii_text = html.escape("\n".join(vp.get("ascii_grid", [])))
    png_b64 = vp.get("viewport_png_b64", "")
    options_html = "\n".join(
        f'<option value="{html.escape(name)}">{html.escape(name)} (cost {TOOL_SCHEMAS[name]["cost"]}) — {html.escape(TOOL_SCHEMAS[name]["description"])}</option>'
        for name in TOOL_NAMES
    )
    inv_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(ep.world.inv.items())
    )
    recent_rows = "".join(
        f"<li><code>{html.escape(t)}({html.escape(json.dumps(a))})</code> &rarr; "
        f"<pre>{html.escape(r.get('text', ''))}</pre></li>"
        for t, a, r in reversed(state.history[-6:])
    )
    error_banner = f'<div class="err">{html.escape(state.last_error)}</div>' if state.last_error else ""
    done_banner = ""
    if ep.done:
        saved = state.saved_paths or {}
        done_banner = (
            f'<div class="done">Episode complete! Sealed ledger saved to '
            f'<code>{html.escape(saved.get("ledger_path", ""))}</code> '
            f'(sha256: <code>{html.escape(saved.get("sha256", "")[:16])}...</code>)</div>'
        )
    img_html = (
        f'<img src="data:image/png;base64,{png_b64}" alt="Viewport" '
        f'style="image-rendering:pixelated;width:366px;height:210px;border:1px solid #334155;" />'
        if png_b64 else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>TerrariaBench Human Operator Console</title>
<style>
body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#0f172a; color:#e2e8f0; margin:20px; }}
.hdr {{ background:#1e293b; border:1px solid #38bdf8; padding:10px 14px; border-radius:6px; font-weight:bold; margin-bottom:14px; }}
.grid {{ display:flex; gap:20px; flex-wrap:wrap; }}
.card {{ background:#1e293b; border:1px solid #334155; border-radius:6px; padding:14px; flex:1; min-width:340px; }}
pre {{ background:#020617; padding:10px; border-radius:4px; overflow-x:auto; font-size:12px; line-height:1.15; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ text-align:left; padding:4px 6px; border-bottom:1px solid #334155; }}
input, select, button {{ background:#020617; color:#f8fafc; border:1px solid #475569; padding:7px 10px; border-radius:4px; }}
button {{ background:#0284c7; border-color:#0284c7; cursor:pointer; font-weight:bold; }}
.err {{ background:#7f1d1d; padding:8px 12px; border-radius:4px; margin-bottom:12px; }}
.done {{ background:#14532d; padding:10px 14px; border-radius:4px; margin-bottom:12px; }}
</style>
</head>
<body>
<h2>TerrariaBench Human Operator Console &mdash; {html.escape(ep.task.title)}</h2>
<div class="hdr">{html.escape(ep.header())}</div>
{error_banner}
{done_banner}
<div class="card" style="margin-bottom:16px;">
  <strong>Briefing &amp; Rules:</strong>
  <pre style="white-space:pre-wrap;">{html.escape(ep.task.prompt)}</pre>
</div>
<div class="grid">
  <div class="card">
    <h3>Execute Tool (24-Tool Surface)</h3>
    <form method="post" action="/step" style="display:flex;flex-direction:column;gap:8px;">
      <label>Tool:<br/><select name="tool" style="width:100%;">{options_html}</select></label>
      <label>JSON Args:<br/><input type="text" name="args_json" value="{{}}" style="width:96%;" /></label>
      <div style="display:flex;gap:8px;">
        <button type="submit">Run Tool</button>
      </div>
    </form>
    <form method="post" action="/reset" style="margin-top:8px;">
      <button type="submit" style="background:#475569;border-color:#475569;">Reset Episode</button>
    </form>
    <h3 style="margin-top:16px;">Equipped &amp; Inventory</h3>
    <p>Weapon: <code>{html.escape(str(ep.world.eq["weapon"]))}</code> | Armor: <code>{html.escape(str(ep.world.eq["armor"]))}</code> | Pickaxe: <code>{html.escape(str(ep.world.eq["pickaxe"]))}</code></p>
    <table><tr><th>Item</th><th>Count</th></tr>{inv_rows}</table>
  </div>
  <div class="card">
    <h3>Viewport (61&times;35)</h3>
    {img_html}
    <pre>{ascii_text}</pre>
  </div>
</div>
<div class="card" style="margin-top:16px;">
  <h3>Recent Tool Calls</h3>
  <ul>{recent_rows}</ul>
</div>
</body>
</html>"""


def create_console_app(task_id: str = "first-night", seed: int = 7) -> FastAPI:
    """Create the FastAPI server-rendered Human Operator Console."""
    state = ConsoleState(task_id=task_id, seed=seed, redact_seed=True)
    app = FastAPI(title="TerrariaBench Human Operator Console")
    app.state.console = state

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_render_console_html(state))

    @app.get("/api/state")
    async def api_state() -> dict[str, Any]:
        return state.public_state()

    @app.post("/step", response_class=HTMLResponse)
    async def step_endpoint(request: Request) -> HTMLResponse:
        raw = (await request.body()).decode("utf-8")
        ctype = request.headers.get("content-type", "")
        state.last_error = ""
        if "application/json" in ctype:
            data = json.loads(raw or "{}")
            tool = str(data.get("tool", ""))
            args = data.get("args") or {}
        else:
            form = parse_qs(raw, keep_blank_values=True)
            tool = (form.get("tool") or [""])[0]
            args_str = (form.get("args_json") or ["{}"])[0].strip() or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError as exc:
                state.last_error = f"Invalid JSON args: {exc}"
                return HTMLResponse(_render_console_html(state))
        if not isinstance(args, dict):
            state.last_error = "JSON args must be an object/dict"
            return HTMLResponse(_render_console_html(state))
        state.step(tool, args)
        return HTMLResponse(_render_console_html(state))

    @app.post("/reset", response_class=HTMLResponse)
    async def reset_endpoint() -> HTMLResponse:
        state.reset()
        return HTMLResponse(_render_console_html(state))

    return app


def run_cli(task_id: str = "first-night", seed: int = 7) -> dict[str, Any]:
    """Run an interactive terminal operator loop over the 24 MCP tools."""
    state = ConsoleState(task_id=task_id, seed=seed, redact_seed=True)
    print(state.ep.task.prompt)
    print("\nType '<tool_name> <json_args>' (or 'tools', 'inv', 'quit'):")
    while not state.ep.done:
        print(f"\n{state.ep.header()}")
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "tools":
            for name in TOOL_NAMES:
                print(f"  {name:28s} cost={TOOL_SCHEMAS[name]['cost']}  {TOOL_SCHEMAS[name]['description']}")
            continue
        if line == "inv":
            print(json.dumps(state.ep.world.inv, sort_keys=True))
            continue
        parts = line.split(None, 1)
        tool = parts[0]
        args: dict[str, Any] = {}
        if len(parts) == 2:
            try:
                args = json.loads(parts[1])
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON args: {exc}")
                continue
        res = state.step(tool, args)
        print(res["text"])

    saved = state.persist()
    print(f"\nEpisode finalized & sealed: {saved['ledger_path']} (sha256={saved['sha256']})")
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerrariaBench pure-Python Human Operator Console")
    parser.add_argument("--task", default="first-night", choices=sorted(TASKS), help="Task ID")
    parser.add_argument("--seed", type=int, default=7, help="World seed")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI bind host")
    parser.add_argument("--port", type=int, default=8080, help="Web UI bind port")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal operator console")
    args = parser.parse_args(argv)

    if args.cli:
        run_cli(args.task, args.seed)
        return 0

    import uvicorn

    uvicorn.run(create_console_app(args.task, args.seed), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
