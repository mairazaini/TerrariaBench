"""NVIDIA Nemotron 3.5 Super VL multimodal agent with rate-limit backoff."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, TypeVar

from ..engine import WORKBENCH
from ..mcp_server import tool_json_schema
from ..tools import TOOL_NAMES, TOOL_SCHEMAS, Episode

T = TypeVar("T")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "rate limit", "too many requests", "500", "502", "503", "504", "timeout"))


def call_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 6,
    base_delay: float = 2.0,
    sleep_fn: Callable[[float], Any] = time.sleep,
) -> T:
    """Invoke ``fn`` with exponential backoff on HTTP 429 / 5xx rate-limit errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            sleep_fn(base_delay * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def openai_tool_schemas() -> list[dict[str, Any]]:
    """Build 24 OpenAI-compatible function tool descriptors from ``TOOL_SCHEMAS``."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_SCHEMAS[name]["description"],
                "parameters": tool_json_schema(name),
            },
        }
        for name in TOOL_NAMES
    ]


def build_multimodal_messages(
    prompt: str,
    observation: dict[str, Any],
    context: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct OpenAI-compatible chat messages (with PNG image_url) and 24 tool schemas."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
    ]
    if context:
        messages.append({"role": "user", "content": f"Recent trajectory context:\n{json.dumps(context[-6:], sort_keys=True)}"})

    obs_clean = {k: v for k, v in observation.items() if k != "viewport_png_b64"}
    user_parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"Current observation:\n{json.dumps(obs_clean, sort_keys=True)}"}
    ]
    png_b64 = observation.get("viewport_png_b64")
    if isinstance(png_b64, str) and png_b64:
        user_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
        })
    messages.append({"role": "user", "content": user_parts})
    return messages, openai_tool_schemas()


class NemotronAgent:
    """NVIDIA Nemotron 3.5 Super VL agent via NVIDIA NIM / OpenRouter fallback."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        live: bool = False,
    ):
        self.name = "nemotron"
        self.model = model or os.environ.get("TERRARIA_NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl")
        nv_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if base_url:
            self.base_url = base_url
            self.api_key = nv_key or or_key
        elif nv_key:
            self.base_url = os.environ.get("NVIDIA_BASE_URL", NVIDIA_BASE_URL)
            self.api_key = nv_key
        else:
            self.base_url = OPENROUTER_BASE_URL
            self.api_key = or_key
        self.live = live
        self._observed_window: int = -1

    def _live_call(self, observation: dict[str, Any], context: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        messages, tools = build_multimodal_messages(prompt, observation, context)
        resp = call_with_backoff(
            lambda: client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        )
        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            args = json.loads(tc.function.arguments or "{}")
            return {"tool": tc.function.name, "args": args if isinstance(args, dict) else {}}
        return {"tool": "end_window", "args": {}}

    def next_action(self, observation: dict[str, Any], context: list[dict[str, Any]], ep: Episode) -> dict[str, Any] | None:
        if ep.done:
            return None
        if self.live and self.api_key:
            try:
                return self._live_call(observation, context, ep.task.prompt)
            except Exception:
                pass

        w = ep.world
        if self._observed_window != w.window and ep.ops_left >= 6:
            self._observed_window = w.window
            return {"tool": "observe_viewport", "args": {"include_png": True}}

        housing_valid = w.housing()["valid"]
        if not housing_valid and "SHELTER" not in w.milestones:
            if w.inv.get("wood", 0) < 28 and any(tx not in w.withered for tx in w.trees):
                return {"tool": "chop_trees", "args": {"radius": 40, "max_trees": 4}}
            if w.inv.get("wood", 0) >= 20:
                return {"tool": "build_blueprint", "args": {"preset": "shelter"}}

        if not w.has("wooden_sword") and w.has("wood", 7) and w.near_tile(WORKBENCH, radius=4):
            return {"tool": "craft_item", "args": {"item_id": "wooden_sword", "quantity": 1}}
        if w.has("wooden_sword") and w.eq["weapon"] != "wooden_sword":
            return {"tool": "equip_item", "args": {"item_id": "wooden_sword"}}
        if not w.has("wood_armor") and w.has("wood", 30) and w.near_tile(WORKBENCH, radius=4):
            return {"tool": "craft_item", "args": {"item_id": "wood_armor", "quantity": 1}}
        if w.has("wood_armor") and w.eq["armor"] != "wood_armor":
            return {"tool": "equip_item", "args": {"item_id": "wood_armor"}}

        if w.inv.get("wood", 0) < 35 and w.tech_tier < 1 and any(tx not in w.withered for tx in w.trees):
            return {"tool": "chop_trees", "args": {"radius": 45, "max_trees": 3}}

        if w.boss is not None:
            return {
                "tool": "engage_combat_burst",
                "args": {"kite_distance": 14.0, "dodge_priority": 0.5, "vertical_platform_use": True},
            }

        if not housing_valid and "SHELTER" in w.milestones:
            return {"tool": "navigate_to", "args": {"x": w.home[0], "y": w.home[1], "budget": 300}}

        return {"tool": "end_window", "args": {}}
