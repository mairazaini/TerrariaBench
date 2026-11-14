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
                    "default_model": "google/gemma-3-27b-it", "rpm_limit": 120},
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
