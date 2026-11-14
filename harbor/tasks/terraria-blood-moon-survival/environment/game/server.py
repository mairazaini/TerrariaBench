"""TerrariaBench game server (Harbor sidecar).

Holds the one metered Episode for this trial. The agent's container never sees
the simulator source, the calibration, or the ledger: it can only call tools,
either over MCP (streamable HTTP at /mcp) or over the plain HTTP mirror
(POST /call) that the `terraria` CLI uses. The verifier finalizes the episode
with a token the agent does not have and reads back the graded score.

    GET  /health                 liveness
    GET  /tools                  the 24 tool schemas
    POST /call {"tool","args"}   one tool call (same metering as MCP)
    POST /finalize               verifier only (X-Verifier-Token)
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Any

import mcp.types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from terraria_bench.grader import grade
from terraria_bench.mcp_server import tool_json_schema
from terraria_bench.provenance import load_calibration
from terraria_bench.tools import TOOL_NAMES, TOOL_SCHEMAS, Episode

TASK_ID = os.environ["TERRARIA_TASK"]
SEED = int(os.environ["TERRARIA_SEED"])
VERIFIER_TOKEN = os.environ["TERRARIA_VERIFIER_TOKEN"]
LOG_PATH = os.environ.get("TERRARIA_ACTION_LOG", "/srv/state/actions.jsonl")

# Fail closed at startup: a stale or tampered calibration means no server, no score.
CALIBRATION = load_calibration()
CELL = CALIBRATION["cells"][f"{TASK_ID}:{SEED}"]

_lock = threading.Lock()
_episode = Episode(TASK_ID, SEED, redact_seed=True)
_result: dict[str, Any] | None = None


def _text(res: dict[str, Any]) -> str:
    body = {k: v for k, v in res.items() if k not in ("header", "text", "viewport_png_b64")}
    return f"{res['header']}\n{json.dumps(body, sort_keys=True)}"


_JSON_CHECKS = {
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list),
}


def _argument_error(name: str, args: dict[str, Any]) -> str | None:
    """Why these arguments cannot be passed to the tool, or None if they can."""
    schema = tool_json_schema(name)
    props = schema["properties"]
    unknown = sorted(set(args) - set(props))
    if unknown:
        return f"unknown parameter(s) {unknown}"
    missing = [p for p in schema["required"] if p not in args]
    if missing:
        return f"missing required parameter(s) {missing}"
    for key, value in list(args.items()):
        kind = props[key]["type"]
        # Forgive the common harmless slips: "12" or 12.0 where an integer is wanted.
        if kind in ("integer", "number") and isinstance(value, str):
            try:
                value = args[key] = float(value) if kind == "number" else int(value)
            except ValueError:
                pass
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            value = args[key] = int(value)
        if value is not None and not _JSON_CHECKS[props[key]["type"]](value):
            return f"parameter '{key}' must be {props[key]['type']}, got {type(value).__name__}"
    return None


def _rejected(name: str, error: str) -> dict[str, Any]:
    body = {"ok": False, "error": f"call to {name} rejected (no operation spent): {error}",
            "expected_params": TOOL_SCHEMAS[name].get("params", {})}
    return {"ok": False, "text": f"{_episode.header()}\n{json.dumps(body, sort_keys=True)}"}


def call_tool(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Run one tool call. Malformed calls are reported to the caller and never reach the world."""
    args = args if isinstance(args, dict) else {}
    with _lock:
        if _result is not None:
            return {"ok": False, "text": "episode is finalized; no further calls are accepted"}
        if name in TOOL_SCHEMAS:
            error = _argument_error(name, args)
            if error:
                return _rejected(name, error)
        try:
            res = _episode.step(name, **args)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
            # The call was metered before it failed. Keep the window boundary honest.
            if _episode.ops_left <= 0 and not _episode.done:
                _episode._settle_current_window()
            body = {"ok": False, "error": f"{name} failed on these arguments: {type(exc).__name__}: {exc}",
                    "expected_params": TOOL_SCHEMAS[name].get("params", {})}
            res = {"ok": False, "header": _episode.header(), **body}
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tool": name, "args": args, "ok": bool(res.get("ok", True))}, default=str) + "\n")
        return {"ok": bool(res.get("ok", True)), "text": _text(res), "done": _episode.done}


def tool_list() -> list[dict[str, Any]]:
    return [{"name": n, "description": f"{TOOL_SCHEMAS[n]['description']} (cost: {TOOL_SCHEMAS[n]['cost']} op)",
             "inputSchema": tool_json_schema(n)} for n in TOOL_NAMES]


# ------------------------------------------------------------------ MCP surface
mcp_server = Server("terraria")


@mcp_server.list_tools()
async def _list_tools() -> list[types.Tool]:
    return [types.Tool(**t) for t in tool_list()]


@mcp_server.call_tool(validate_input=False)
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=call_tool(name, arguments)["text"])]


session_manager = StreamableHTTPSessionManager(app=mcp_server, json_response=True, stateless=True)


# ------------------------------------------------------------- plain HTTP mirror
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "task": TASK_ID})


async def tools(_: Request) -> JSONResponse:
    return JSONResponse({"tools": tool_list()})


async def call(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse({"ok": False, "text": "body must be JSON: {\"tool\": str, \"args\": object}"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"ok": False, "text": "body must be a JSON object"}, status_code=400)
    return JSONResponse(call_tool(str(payload.get("tool", "")), payload.get("args")))


async def finalize(request: Request) -> JSONResponse:
    global _result
    if request.headers.get("x-verifier-token") != VERIFIER_TOKEN:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    with _lock:
        if _result is None:
            ledger = _episode.finalize(CALIBRATION)
            g = grade(ledger, CELL)
            _result = {"reward": float(g["composite"]) if g.get("valid") else 0.0, "grade": g,
                       "ledger": ledger, "actions": len(_episode.actions_log)}
        return JSONResponse(_result)


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    async with session_manager.run():
        yield


http_app = Starlette(
    routes=[Route("/health", health), Route("/tools", tools),
            Route("/call", call, methods=["POST"]), Route("/finalize", finalize, methods=["POST"])],
    lifespan=lifespan,
)


async def app(scope, receive, send):
    if scope["type"] == "http" and scope["path"].rstrip("/") == "/mcp":
        await session_manager.handle_request(scope, receive, send)
    else:
        await http_app(scope, receive, send)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
