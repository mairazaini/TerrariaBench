"""Harbor agent: the two-tier Gemini player driving a TerrariaBench Harbor task.

A supervisor model (``-m gemini/gemini-4-hydrogen`` by default) writes a menu of
candidate tool calls for each window; the Gemini Decisions API
(``gemini-decision-1``) picks one candidate per operation from the live state.
The Decisions API cannot back a chat harness such as OpenCode (it only answers
structured questions), so this agent is the harness that puts it on the
controls.

    PYTHONPATH=. harbor run \
        -p harbor/tasks/terraria-first-night \
        --agent-import-path terraria_bench.harness.harbor_agent:TwoTierGeminiAgent \
        -m gemini/gemini-4-hydrogen --env-file .env

The models run in the Harbor process; every tool call goes through the task's
agent container to the game sidecar, exactly as the ``terraria`` CLI does, so
metering and grading are identical to any other agent's.
"""
from __future__ import annotations

import asyncio
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from harbor.agents.base import BaseAgent
    from harbor.agents.capabilities import AgentCapabilities
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ImportError:  # Harbor not installed here: the adapter below still works, the agent class does not
    BaseAgent = object  # type: ignore[assignment,misc]
    AgentCapabilities = BaseEnvironment = AgentContext = None  # type: ignore[assignment,misc]

from terraria_bench.harness.gemini_agent import GeminiAgent


def _ensure_genai() -> None:
    """Harbor's own environment may lack google-genai; fall back to the repo virtualenv's copy.

    The path is appended, so Harbor's own dependencies keep precedence.
    """
    try:
        import google.genai  # noqa: F401
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parents[2]
    for site in (root / ".venv" / "Lib" / "site-packages", *root.glob(".venv/lib/python*/site-packages")):
        if (site / "google" / "genai").is_dir():
            sys.path.append(str(site))
            import google
            google.__path__.append(str(site / "google"))  # type: ignore[attr-defined]
            return


_ensure_genai()

GAME_URL = "http://game:8000"
DEFAULT_SUPERVISOR = "gemini-4-hydrogen"
MAX_TURNS = 2000
HISTORY_TURNS = 12
_WIN = re.compile(r"Win:\s*(\d+)/(\d+)")
_OPS = re.compile(r"Ops:\s*(\d+)/(\d+)")


def parse_reply(reply: dict[str, Any]) -> dict[str, Any]:
    """Turn the sidecar's ``{"ok", "text", "done"}`` into the observation dict agents expect."""
    header, _, body = str(reply.get("text", "")).partition("\n")
    try:
        obs = json.loads(body) if body.strip() else {}
    except ValueError:
        obs = {"raw": body}
    if not isinstance(obs, dict):
        obs = {"raw": obs}
    obs["header"] = header
    obs["ok"] = bool(reply.get("ok", obs.get("ok", True)))
    return obs


class RemoteEpisode:
    """The slice of ``Episode`` the live two-tier agent uses, backed by the game sidecar."""

    def __init__(self, call: Any, prompt: str):
        self._call = call  # (tool, args) -> sidecar reply dict
        self.task = SimpleNamespace(prompt=prompt, windows=0)
        self.world = SimpleNamespace(window=0)
        self.done = False
        self.ops_left = 0
        self.window_budget = 0
        self._header = ""

    def header(self) -> str:
        return self._header

    def step(self, tool: str, **args: Any) -> dict[str, Any]:
        reply = self._call(tool, args)
        obs = parse_reply(reply)
        self._header = obs["header"] or self._header
        if (m := _WIN.search(self._header)):
            self.world.window, self.task.windows = int(m.group(1)), int(m.group(2))
        if (m := _OPS.search(self._header)):
            self.ops_left, self.window_budget = int(m.group(1)), int(m.group(2))
        self.done = bool(reply.get("done", False)) or "finalized" in str(reply.get("text", ""))
        return obs


def play(agent: GeminiAgent, ep: RemoteEpisode, on_turn: Any = None, max_turns: int = MAX_TURNS) -> int:
    """Run ``agent`` until the episode is done. Returns the number of tool calls made."""
    obs = ep.step("terraria_status")  # free; gives the opening header
    obs["prompt"] = ep.task.prompt
    history: list[dict[str, Any]] = []
    turns = 0
    while turns < max_turns and not ep.done:
        action = agent.next_action(obs, history[-HISTORY_TURNS:], ep)  # type: ignore[arg-type]
        if not action or not action.get("tool"):
            break
        tool, args = action["tool"], dict(action.get("args") or {})
        obs = ep.step(tool, **args)
        turns += 1
        history.append({"turn": turns, "tool": tool, "args": args, "result": obs})
        if on_turn is not None:
            on_turn(turns, tool, args, obs)
    return turns


class TwoTierGeminiAgent(BaseAgent):
    """Supervisor model plans each window; ``gemini-decision-1`` chooses every action."""

    # The task declares the game as an MCP server; this agent reaches the same server over its HTTP mirror.
    if AgentCapabilities is not None:
        capabilities = AgentCapabilities(mcp_servers=True)

    @staticmethod
    def name() -> str:
        return "terraria-two-tier-gemini"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        loop = asyncio.get_running_loop()
        supervisor = (self.model_name or DEFAULT_SUPERVISOR).split("/")[-1]

        def call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
            payload = json.dumps({"tool": tool, "args": args})
            command = (f"curl -sS --max-time 120 -X POST {GAME_URL}/call "
                       f"-H 'Content-Type: application/json' -d {shlex.quote(payload)}")
            result = asyncio.run_coroutine_threadsafe(environment.exec(command, timeout_sec=180), loop).result()
            try:
                return json.loads(result.stdout or "")
            except ValueError as exc:
                raise RuntimeError(f"game server call failed: {result.stderr or result.stdout!r}") from exc

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        turn_log = (self.logs_dir / "two_tier_turns.jsonl").open("a", encoding="utf-8")

        def on_turn(turn: int, tool: str, args: dict[str, Any], obs: dict[str, Any]) -> None:
            turn_log.write(json.dumps({"turn": turn, "tool": tool, "args": args, "ok": obs.get("ok"),
                                       "header": obs.get("header")}, default=str) + "\n")
            turn_log.flush()

        agent = GeminiAgent(model=supervisor, live=True)
        episode = RemoteEpisode(call, instruction)
        error: str | None = None
        turns = 0
        try:
            turns = await asyncio.to_thread(play, agent, episode, on_turn)
        except Exception as exc:  # keep the trace; the verifier still grades what was played
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            turn_log.close()
            summary = {"supervisor_model": supervisor, "decision_model": agent.decisions.model, "turns": turns,
                       "stats": agent.stats, "error": error, "final_header": episode.header()}
            (self.logs_dir / "two_tier_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            (self.logs_dir / "two_tier_trace.json").write_text(json.dumps(agent.trace, indent=1, default=str),
                                                               encoding="utf-8")
            context.metadata = {**(context.metadata or {}), **summary}
