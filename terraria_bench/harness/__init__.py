"""TerrariaBench multi-model agent evaluation & RL training harness."""
from __future__ import annotations

from .gemini_agent import GeminiAgent, GeminiDecisionsWrapper, synthesize_blueprint_matrix
from .nemotron_agent import NemotronAgent, build_multimodal_messages, call_with_backoff

_RUNNER_EXPORTS = (
    "AgentProtocol",
    "CompactHistory",
    "EpisodeRunner",
    "run_agent_episode",
    "run_eval_standard",
)
_OPTIONAL_TRACK3_EXPORTS = {
    "AgentConfig": ".gemma_agent",
    "GemmaAgent": ".gemma_agent",
    "StepRecord": ".gemma_agent",
    "Trajectory": ".gemma_agent",
    "evaluate_agent": ".gemma_agent",
    "BehaviorCloningTrainer": ".train_grpo",
    "GRPOConfig": ".train_grpo",
    "GRPOTrainer": ".train_grpo",
    "build_lora_config": ".train_grpo",
    "compute_group_advantages": ".train_grpo",
    "compute_salvage_discounted_reward": ".train_grpo",
}

__all__ = [
    "AgentProtocol",
    "CompactHistory",
    "EpisodeRunner",
    "GeminiAgent",
    "GeminiDecisionsWrapper",
    "NemotronAgent",
    "build_multimodal_messages",
    "call_with_backoff",
    "run_agent_episode",
    "run_eval_standard",
    "synthesize_blueprint_matrix",
]


def __getattr__(name: str):
    if name in _RUNNER_EXPORTS:
        from . import runner

        return getattr(runner, name)
    if name in _OPTIONAL_TRACK3_EXPORTS:
        import importlib

        mod = importlib.import_module(_OPTIONAL_TRACK3_EXPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
