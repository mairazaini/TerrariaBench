"""StoreBench §7 Open-Weight Gemma 4 Agent and Evaluation Harness.

This module provides the baseline and LoRA adapter agent for the Terraria benchmark:
- AgentConfig: Dataclass configuring model weights, adapter paths, and generation params.
- StepRecord & Trajectory: Dataclasses tracking rollout state, tokens, log-probs, and clipped rewards.
- GemmaAgent: Agent class supporting unadapted baselines and PEFT LoRA adapters,
  with deterministic and stochastic action sampling and exact per-token log-prob calculation.
- evaluate_agent: Evaluates agent checkpoints across benchmark tasks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore
    PeftModel = None  # type: ignore
    get_peft_model = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

logger = logging.getLogger(__name__)


@runtime_checkable
class BenchmarkEnv(Protocol):
    """Protocol defining the Terraria benchmark environment interface."""

    def reset(self, seed: Optional[int] = None) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...


@dataclass
class AgentConfig:
    """Configuration for Gemma 4 Agent and LoRA evaluation."""

    model_name_or_path: str = "google/gemma-4-9b-it"
    adapter_path: Optional[str] = None
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    max_episode_steps: int = 30
    device: str = "auto"


@dataclass
class StepRecord:
    """Record of a single agent step during an episode rollout."""

    prompt_text: str
    action_text: str
    parsed_tool_call: dict[str, Any]
    prompt_token_ids: list[int]
    action_token_ids: list[int]
    action_log_probs: list[float]
    raw_salvage_reward: float = 0.0
    clipped_reward: float = 0.0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete trajectory of an episode rollout."""

    steps: list[StepRecord] = field(default_factory=list)
    task_id: str = ""
    seed: Optional[int] = None
    total_raw_reward: float = 0.0
    total_clipped_reward: float = 0.0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class GemmaAgent:
    """Gemma 4 Agent supporting baseline evaluation and PEFT LoRA adapters."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or AgentConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = self._resolve_dtype(self.config.torch_dtype)

        self._model = model
        self._tokenizer = tokenizer

        if self._model is not None and self.config.adapter_path:
            self.load_adapter(self.config.adapter_path)

    @property
    def model(self) -> Any:
        if self._model is None:
            self._ensure_loaded()
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._ensure_loaded()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Any) -> None:
        self._tokenizer = value

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested_device)

    def _resolve_dtype(self, requested_dtype: str) -> torch.dtype:
        if requested_dtype in ("bfloat16", "bf16"):
            # CPU fallback for bfloat16 in unit testing / environments without hardware bf16
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if requested_dtype in ("float16", "fp16"):
            return torch.float16
        return torch.float32

    def _ensure_loaded(self) -> None:
        """Lazily loads base model and tokenizer if not injected."""
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers package is required to load models. Please install transformers."
            )

        logger.info("Loading model %s on %s...", self.config.model_name_or_path, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id or 0

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)

        if self.config.adapter_path:
            self.load_adapter(self.config.adapter_path)

    def attach_fresh_lora(self, lora_config: Any) -> Any:
        """Wraps the base model with PEFT get_peft_model."""
        if get_peft_model is None:
            raise ImportError("peft package is required for attach_fresh_lora.")
        self.model = get_peft_model(self.model, lora_config)
        return self.model

    def load_adapter(self, adapter_path: str) -> None:
        """Loads a saved PEFT LoRA checkpoint onto the base model."""
        if PeftModel is None:
            raise ImportError("peft package is required for load_adapter.")
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(adapter_path, adapter_name="default")
        else:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,
            )

    def format_observation(self, obs: Any, history: Optional[list[StepRecord]] = None) -> str:
        """Converts environment observation and history into Gemma's chat format."""
        history_items = []
        if history:
            for idx, rec in enumerate(history[-5:]):
                history_items.append(
                    f"Turn {idx + 1}: Action={rec.action_text}"
                )
        history_str = "\n".join(history_items)

        obs_text = obs if isinstance(obs, str) else json.dumps(obs, default=str)

        prompt = (
            "<start_of_turn>user\n"
            "You are an expert agent playing the Terraria benchmark. "
            "Output your next action strictly as a structured JSON tool call.\n"
        )
        if history_str:
            prompt += f"Recent History:\n{history_str}\n"
        prompt += f"Current Observation:\n{obs_text}\n<end_of_turn>\n<start_of_turn>model\n"
        return prompt

    def parse_action(self, raw_output: str) -> dict[str, Any]:
        """Extracts structured tool call from model output, returning fallback on malformed syntax."""
        text = raw_output.strip()
        if not text:
            return {"action": "no_op", "reason": "empty_output"}

        # 1. Attempt extracting from markdown code block
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 2. Attempt extracting first balanced JSON object
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 3. Direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Graceful fallback: return safe no_op instead of crashing
        return {
            "action": "no_op",
            "reason": "malformed_action",
            "raw": text[:100],
        }

    def act(
        self,
        obs: Any,
        history: Optional[list[StepRecord]] = None,
        deterministic: bool = False,
    ) -> StepRecord:
        """Runs model inference and computes exact per-token action log-probabilities."""
        prompt_text = self.format_observation(obs, history)

        # Tokenize prompt
        if hasattr(self.tokenizer, "encode"):
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        else:
            prompt_token_ids = self.tokenizer(prompt_text)["input_ids"]

        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.squeeze(0).tolist()

        # Generate action token IDs
        action_token_ids: list[int] = []
        if hasattr(self.model, "generate"):
            input_tensor = torch.tensor([prompt_token_ids], device=self.device, dtype=torch.long)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", 0) or 0,
            }
            if deterministic:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = max(self.config.temperature, 1e-4)
                gen_kwargs["top_p"] = self.config.top_p

            with torch.no_grad():
                gen_out = self.model.generate(input_tensor, **gen_kwargs)
            full_gen = gen_out[0].tolist()
            action_token_ids = full_gen[len(prompt_token_ids):]
        else:
            # Fallback for mock models without standard generate method
            curr_ids = list(prompt_token_ids)
            with torch.no_grad():
                for _ in range(min(self.config.max_new_tokens, 8)):
                    inp = torch.tensor([curr_ids], device=self.device, dtype=torch.long)
                    out = self.model(inp)
                    next_logits = out.logits[0, -1, :]
                    if deterministic:
                        next_id = int(torch.argmax(next_logits).item())
                    else:
                        probs = F.softmax(next_logits / max(self.config.temperature, 1e-4), dim=-1)
                        next_id = int(torch.multinomial(probs, num_samples=1).item())
                    action_token_ids.append(next_id)
                    curr_ids.append(next_id)
                    if hasattr(self.tokenizer, "eos_token_id") and next_id == self.tokenizer.eos_token_id:
                        break

        # Compute exact per-token log-probs of action tokens conditioned on prompt tokens
        action_log_probs: list[float] = []
        if len(action_token_ids) > 0:
            full_ids = prompt_token_ids + action_token_ids
            full_tensor = torch.tensor([full_ids], device=self.device, dtype=torch.long)
            with torch.no_grad():
                outputs = self.model(full_tensor)
                logits = outputs.logits  # shape: (1, P + A, vocab_size)

            P = len(prompt_token_ids)
            A = len(action_token_ids)
            # Logits predicting action tokens are at positions P - 1 through P + A - 2 (exclusive end: P + A - 1)
            action_logits = logits[0, P - 1 : P + A - 1, :]
            log_probs = F.log_softmax(action_logits, dim=-1)

            action_tensor = torch.tensor(action_token_ids, device=self.device, dtype=torch.long).unsqueeze(-1)
            token_log_probs = log_probs.gather(dim=-1, index=action_tensor).squeeze(-1)
            action_log_probs = [float(lp.item()) for lp in token_log_probs]

        # Decode action text and parse tool call
        if hasattr(self.tokenizer, "decode"):
            action_text = self.tokenizer.decode(action_token_ids, skip_special_tokens=True)
        else:
            action_text = "".join(chr(t) for t in action_token_ids)

        parsed_tool_call = self.parse_action(action_text)

        return StepRecord(
            prompt_text=prompt_text,
            action_text=action_text,
            parsed_tool_call=parsed_tool_call,
            prompt_token_ids=prompt_token_ids,
            action_token_ids=action_token_ids,
            action_log_probs=action_log_probs,
        )

    def run_episode(
        self,
        env: Any,
        task_id: str,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> Trajectory:
        """Executes a full rollout loop up to max_episode_steps or terminal state."""
        # Clean import of reward calculation with self-contained fallback
        try:
            from terraria_bench.harness.train_grpo import compute_salvage_discounted_reward
        except ImportError:
            def compute_salvage_discounted_reward(
                terminal_score: float = 0.0,
                salvage_credit: float = 0.0,
                steps: int = 0,
                invalid_actions: int = 0,
                discount: float = 0.99,
                step_penalty: float = 0.01,
                invalid_action_penalty: float = 0.05,
                salvage_weight: float = 0.5,
                clip_min: float = -2.0,
                clip_max: float = 1.0,
                raw_return: Optional[float] = None,
            ) -> float:
                if raw_return is not None:
                    raw = float(raw_return)
                else:
                    raw = (
                        (terminal_score + salvage_weight * salvage_credit) * (discount ** steps)
                        - steps * step_penalty
                        - invalid_actions * invalid_action_penalty
                    )
                return float(max(clip_min, min(clip_max, raw)))

        # Reset environment
        if seed is not None:
            try:
                reset_res = env.reset(seed=seed)
            except TypeError:
                reset_res = env.reset()
        else:
            reset_res = env.reset()

        if isinstance(reset_res, tuple):
            obs = reset_res[0]
            info = reset_res[1] if len(reset_res) > 1 and isinstance(reset_res[1], dict) else {}
        else:
            obs = reset_res
            info = {}

        history: list[StepRecord] = []
        step_count = 0
        done = False
        invalid_actions_count = 0
        raw_rewards: list[float] = []
        success = False

        while not done and step_count < self.config.max_episode_steps:
            step_rec = self.act(obs, history, deterministic=deterministic)

            if step_rec.parsed_tool_call.get("action") == "no_op" and "reason" in step_rec.parsed_tool_call:
                invalid_actions_count += 1

            # Step environment
            step_result = env.step(step_rec.parsed_tool_call)

            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                elif len(step_result) == 4:
                    obs, reward, done_flag, info = step_result
                    done = bool(done_flag)
                else:
                    obs = step_result[0]
                    reward = float(step_result[1]) if len(step_result) > 1 else 0.0
                    done = bool(step_result[2]) if len(step_result) > 2 else False
                    info = step_result[3] if len(step_result) > 3 and isinstance(step_result[3], dict) else {}
            else:
                reward = 0.0
                done = True
                info = {}

            if isinstance(info, dict) and info.get("success", False):
                success = True

            raw_rewards.append(float(reward))
            history.append(step_rec)
            step_count += 1

        # Compute terminal salvage-discounted return
        terminal_score = float(info.get("terminal_score", 1.0 if success else 0.0))
        salvage_credit = float(info.get("salvage_credit", info.get("salvage_reward", 0.0)))
        raw_sum = sum(raw_rewards) if raw_rewards else 0.0

        if "salvage_reward" in info or "salvage_credit" in info or "terminal_score" in info:
            total_raw = compute_salvage_discounted_reward(
                terminal_score=terminal_score,
                salvage_credit=salvage_credit,
                steps=step_count,
                invalid_actions=invalid_actions_count,
                clip_min=-float("inf"),
                clip_max=float("inf"),
            )
        else:
            total_raw = raw_sum

        total_clipped = max(-2.0, min(1.0, total_raw))

        for rec in history:
            rec.raw_salvage_reward = total_raw
            rec.clipped_reward = total_clipped
            rec.success = success

        return Trajectory(
            steps=history,
            task_id=task_id,
            seed=seed,
            total_raw_reward=total_raw,
            total_clipped_reward=total_clipped,
            success=success,
            metadata={
                "steps": step_count,
                "invalid_actions": invalid_actions_count,
                "info": info,
            },
        )


def evaluate_agent(
    agent: GemmaAgent,
    env_factory: Callable[[str], Any],
    task_ids: list[str],
    num_seeds: int = 3,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Evaluates agent baseline or LoRA checkpoint across a list of tasks."""
    task_metrics: dict[str, dict[str, float]] = {}
    all_clipped: list[float] = []
    all_raw: list[float] = []
    all_succ: list[float] = []
    trajectories_data: list[dict[str, Any]] = []

    for task_id in task_ids:
        task_clipped = []
        task_raw = []
        task_succ = []

        for seed in range(num_seeds):
            env = env_factory(task_id)
            traj = agent.run_episode(env, task_id=task_id, seed=seed, deterministic=True)

            task_clipped.append(traj.total_clipped_reward)
            task_raw.append(traj.total_raw_reward)
            task_succ.append(1.0 if traj.success else 0.0)

            trajectories_data.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "total_clipped_reward": traj.total_clipped_reward,
                    "total_raw_reward": traj.total_raw_reward,
                    "success": traj.success,
                    "num_steps": len(traj.steps),
                    "steps": [
                        {
                            "action_text": s.action_text,
                            "parsed_tool_call": s.parsed_tool_call,
                        }
                        for s in traj.steps
                    ],
                }
            )

        task_metrics[task_id] = {
            "mean_clipped_reward": float(sum(task_clipped) / len(task_clipped)) if task_clipped else 0.0,
            "mean_raw_reward": float(sum(task_raw) / len(task_raw)) if task_raw else 0.0,
            "success_rate": float(sum(task_succ) / len(task_succ)) if task_succ else 0.0,
        }
        all_clipped.extend(task_clipped)
        all_raw.extend(task_raw)
        all_succ.extend(task_succ)

    overall_metrics = {
        "mean_clipped_reward": float(sum(all_clipped) / len(all_clipped)) if all_clipped else 0.0,
        "mean_raw_reward": float(sum(all_raw) / len(all_raw)) if all_raw else 0.0,
        "success_rate": float(sum(all_succ) / len(all_succ)) if all_succ else 0.0,
        "tasks": task_metrics,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, indent=2)
        with open(out_path / "trajectories.json", "w", encoding="utf-8") as f:
            json.dump(trajectories_data, f, indent=2)

    return overall_metrics
