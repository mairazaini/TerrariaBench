"""24-tool Model Context Protocol (MCP) server for TerrariaBench.

Supports FastMCP when the optional ``mcp`` package is installed and falls back
cleanly to a zero-dependency stdlib JSON-RPC 2.0 stdio server otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

from .provenance import write_sealed
from .tasks import TASKS
from .tools import TOOL_NAMES, TOOL_SCHEMAS, Episode

_JSON_TYPES = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
    "list": "array",
}


def tool_json_schema(name: str) -> dict[str, Any]:
    """Build a JSON Schema ``inputSchema`` from ``TOOL_SCHEMAS[name]['params']``."""
    spec = TOOL_SCHEMAS[name]
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, pdesc in spec.get("params", {}).items():
        type_part, has_default, _ = str(pdesc).partition("=")
        base_types = [t.strip() for t in type_part.split("|") if t.strip() != "None"]
        jtype = _JSON_TYPES.get(base_types[0] if base_types else "str", "string")
        prop: dict[str, Any] = {"type": jtype}
        if jtype == "array":
            prop["items"] = {"type": "array", "items": {"type": "integer"}}
        properties[pname] = prop
        if not has_default and "|None" not in type_part:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


class MCPToolServer:
    """In-process & stdio JSON-RPC 2.0 MCP server wrapping a single metered Episode."""

    def __init__(self, task_id: str = "first-night", seed: int = 7, redact_seed: bool = False):
        self.task_id = task_id
        self.seed = int(seed)
        self.redact_seed = redact_seed
        self.ep = Episode(self.task_id, self.seed, redact_seed=self.redact_seed)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": TOOL_SCHEMAS[name]["description"],
                "cost": TOOL_SCHEMAS[name]["cost"],
                "inputSchema": tool_json_schema(name),
            }
            for name in TOOL_NAMES
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        res = self.ep.step(name, **(arguments or {}))
        content: list[dict[str, Any]] = [{"type": "text", "text": res["text"]}]
        if "viewport_png_b64" in res:
            content.append({
                "type": "image",
                "data": res["viewport_png_b64"],
                "mimeType": "image/png",
            })
        return {"content": content, "isError": not res.get("ok", True), "raw": res}

    def handle_jsonrpc(self, req: dict[str, Any]) -> dict[str, Any] | None:
        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "initialize":
            if "task" in params or "seed" in params or "redact_seed" in params:
                self.task_id = params.get("task", self.task_id)
                self.seed = int(params.get("seed", self.seed))
                self.redact_seed = bool(params.get("redact_seed", self.redact_seed))
                self.ep = Episode(self.task_id, self.seed, redact_seed=self.redact_seed)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "terraria-bench", "version": "0.1.0"},
                    "header": self.ep.header(),
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}
        if method == "tools/call":
            result = self.call_tool(str(params.get("name", "")), params.get("arguments"))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": result["content"], "isError": result["isError"]}}
        if method == "episode/finalize":
            ledger = self.ep.finalize(params.get("calibration"))
            return {"jsonrpc": "2.0", "id": req_id, "result": ledger}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self, in_stream: TextIO | None = None, out_stream: TextIO | None = None) -> None:
        src = in_stream or sys.stdin
        dst = out_stream or sys.stdout
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as exc:
                resp: dict[str, Any] | None = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            else:
                resp = self.handle_jsonrpc(req)
            if resp is not None:
                dst.write(json.dumps(resp, separators=(",", ":")) + "\n")
                dst.flush()


def build_fastmcp_server(server: MCPToolServer) -> Any:
    """Build a FastMCP instance exposing all 24 tools when the ``mcp`` SDK is installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None

    mcp = FastMCP("terraria-bench")
    for name in TOOL_NAMES:
        desc = TOOL_SCHEMAS[name]["description"]

        def _make_handler(tool_name: str):
            def _handler(**kwargs: Any) -> str:
                return server.call_tool(tool_name, kwargs)["content"][0]["text"]

            _handler.__name__ = tool_name
            return _handler

        mcp.add_tool(_make_handler(name), name=name, description=desc)
    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerrariaBench 24-tool MCP Server")
    parser.add_argument("--task", default="first-night", choices=sorted(TASKS), help="Task ID")
    parser.add_argument("--seed", type=int, default=7, help="World seed")
    parser.add_argument("--redact-seed", action="store_true", help="Mask seed in status header (Appendix F)")
    parser.add_argument("--export-ledger", default=None, help="Write finalized sealed ledger JSON on exit")
    parser.add_argument("--stdio-jsonrpc", action="store_true", help="Force stdlib JSON-RPC 2.0 stdio transport")
    parser.add_argument("--list-tools", action="store_true", help="Print all 24 MCP tool schemas and exit")
    args = parser.parse_args(argv)

    server = MCPToolServer(task_id=args.task, seed=args.seed, redact_seed=args.redact_seed)
    if args.list_tools:
        print(json.dumps({"tools": server.list_tools()}, indent=2))
        return 0

    fastmcp = None if args.stdio_jsonrpc else build_fastmcp_server(server)
    if fastmcp is not None:
        fastmcp.run(transport="stdio")
    else:
        server.serve_stdio()

    if args.export_ledger:
        write_sealed(args.export_ledger, server.ep.finalize())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
