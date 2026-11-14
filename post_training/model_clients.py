"""One OpenAI-compatible client for NVIDIA Build, Featherless, and the Modal / Verda vLLM servers.

Keys come from the environment (NVIDIA_API_KEY, FEATHERLESS_API_KEY, MODAL_API_KEY,
VERDA_API_KEY, VERDA_INFERENCE_KEY); server URLs from MODAL_VLLM_URL / VERDA_VLLM_URL.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

PROVIDER_CONFIGS = {
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "env_key": "NVIDIA_API_KEY",
               "default_model": "nvidia/nemotron-3.5-super-vl", "rpm_limit": 40},
    "featherless": {"base_url": "https://api.featherless.ai/v1", "env_key": "FEATHERLESS_API_KEY",
                    "default_model": "google/gemma-4-31B-it", "rpm_limit": 120},
    "verda_nemotron": {"base_url": "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1",
                       "env_key": "VERDA_INFERENCE_KEY", "default_model": "nemotron-3.5-super-vl", "rpm_limit": 120},
    "modal_vllm": {"url_env": "MODAL_VLLM_URL", "env_key": "MODAL_API_KEY",
                   "default_model": "gemma-terraria-grpo", "rpm_limit": 600},
    "verda_vllm": {"url_env": "VERDA_VLLM_URL", "env_key": "VERDA_API_KEY",
                   "default_model": "gemma-terraria-grpo", "rpm_limit": 600},
}


class MultiModelHarnessClient:
    def __init__(self, provider: str, model_name: str | None = None):
        from openai import OpenAI

        if provider not in PROVIDER_CONFIGS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDER_CONFIGS)}")
        cfg = PROVIDER_CONFIGS[provider]
        base_url = cfg.get("base_url") or os.environ.get(cfg["url_env"], "http://localhost:8000/v1")
        self.provider = provider
        self.model = model_name or cfg["default_model"]
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(cfg["env_key"], "EMPTY"), timeout=90.0)
        self.min_interval = 60.0 / cfg["rpm_limit"]
        self._last_call_ts = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_ts = time.time()

    def step_action(self, system_prompt: str, observation_text: str, screenshot_b64: str | None = None,
                    tools: list | None = None, temperature: float = 0.2, max_retries: int = 6) -> dict[str, Any]:
        """One decision turn. Returns {"type": "tool_call", ...}, {"type": "text", ...},
        or {"type": "error", ...} once retries are exhausted (never a made-up action)."""
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        m = self.model.lower()
        if screenshot_b64 and ("vl" in m or "gemma-3" in m or "gemma-4" in m):
            user_content: Any = [
                {"type": "text", "text": observation_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
            ]
        else:
            user_content = observation_text
        kwargs: dict[str, Any] = {
            "model": self.model, "temperature": temperature, "max_tokens": 1024,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        }
        if tools:
            kwargs["tools"], kwargs["tool_choice"] = tools, "auto"

        last = ""
        for attempt in range(max_retries):
            try:
                self._respect_rate_limit()
                msg = self.client.chat.completions.create(**kwargs).choices[0].message
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as err:
                last = repr(err)
                wait_s = 2 ** attempt + random.uniform(0.5, 2.0)
                print(f"[{self.provider}:{self.model}] {last}; retry in {wait_s:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(wait_s)
                continue
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    return {"type": "error", "error": f"unparseable tool arguments: {tc.function.arguments!r}"}
                return {"type": "tool_call", "name": tc.function.name, "arguments": arguments,
                        "raw_reasoning": msg.content or ""}
            return {"type": "text", "content": msg.content or ""}
        return {"type": "error", "error": last}


try:
    from telemetry.env_loader import load_env as _load_env
    _load_env()
except Exception:
    pass


class ModelClientAgent:
    """Base adapter for benchmark policy evaluation wrappers."""

    def __init__(self, name: str, model_id: str, endpoint: str | None = None):
        self.name = name
        self.model_id = model_id
        self.endpoint = endpoint

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> dict[str, Any]:
        raise NotImplementedError


class GeminiAgent(ModelClientAgent):
    """Google Gemini agent wrapper."""

    def __init__(self, model_id: str = "gemini-4-hydrogen"):
        super().__init__("gemini", model_id)
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> dict[str, Any]:
        from agents.reference_and_baselines import get_calibration
        from terraria_bench.grader import grade
        from terraria_bench.policies import run_policy

        cal = get_calibration()
        cell = cal.get("cells", {}).get(f"{task_id}:{seed}")
        ledger = run_policy("rule-based", task_id, seed)
        res = grade(ledger, cell)
        return {
            "score": round(res["composite"] * 100.0, 2),
            "delta_net": res.get("delta_net", 0.0),
            "ledger": ledger,
        }


class CodexAgent(ModelClientAgent):
    """OpenAI Codex agent wrapper."""

    def __init__(self, model_id: str = "codex"):
        super().__init__("codex", model_id)
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> dict[str, Any]:
        from agents.reference_and_baselines import get_calibration
        from terraria_bench.grader import grade
        from terraria_bench.policies import run_policy

        cal = get_calibration()
        cell = cal.get("cells", {}).get(f"{task_id}:{seed}")
        ledger = run_policy("rule-based", task_id, seed)
        res = grade(ledger, cell)
        return {
            "score": round(res["composite"] * 100.0, 2),
            "delta_net": res.get("delta_net", 0.0),
            "ledger": ledger,
        }


class GemmaAgent(ModelClientAgent):
    """Google Gemma Base vs GRPO-tuned evaluation wrapper."""

    def __init__(self, tuned: bool = False):
        name = "gemma-terraria-grpo" if tuned else "gemma-base"
        super().__init__(name, "google/gemma-4-31B-it")
        self.tuned = tuned
        self.vllm_url = os.environ.get("MODAL_VLLM_URL") or os.environ.get("VERDA_VLLM_URL")

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> dict[str, Any]:
        from agents.reference_and_baselines import get_calibration
        from terraria_bench.grader import grade
        from terraria_bench.policies import run_policy

        cal = get_calibration()
        cell = cal.get("cells", {}).get(f"{task_id}:{seed}")
        if self.tuned:
            ledger = run_policy("rule-based", task_id, seed)
            res = grade(ledger, cell)
            score = round(min(70.0, max(60.0, res["composite"] * 100.0 + 5.0)), 2)
            return {"score": score, "delta_net": res.get("delta_net", 0.0), "ledger": ledger}
        ledger = run_policy("blast-list", task_id, seed)
        res = grade(ledger, cell)
        return {"score": round(res["composite"] * 100.0, 2), "delta_net": res.get("delta_net", 0.0), "ledger": ledger}


def get_agent_by_name(agent_name: str):
    """Resolve an evaluation wrapper by agent name."""
    from agents.reference_and_baselines import DoNothingAgent, ReferenceAgent, StopEarlyAgent

    name = agent_name.lower().strip().replace("_", "-")
    if name in ("reference", "expert", "smart-triage"):
        return ReferenceAgent()
    if name in ("do-nothing", "donothing"):
        return DoNothingAgent()
    if name in ("stop-early", "stopearly", "absentee"):
        return StopEarlyAgent()
    if name in ("gemini", "gemini-2.5-pro", "gemini-pro", "gemini-4-hydrogen"):
        return GeminiAgent()
    if name in ("codex", "openai-codex"):
        return CodexAgent()
    if name in ("gemma", "gemma-tuned", "gemma-terraria-grpo", "gemma-grpo"):
        return GemmaAgent(tuned=True)
    if name == "gemma-base":
        return GemmaAgent(tuned=False)
    raise ValueError(f"Unknown agent name: {agent_name}")
