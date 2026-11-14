"""TerrariaBench: deterministic windowed MCP benchmark and RL environment."""
from __future__ import annotations

from .engine import BOSSES, ITEMS, MOBS, RECIPES, World, world_hash
from .grader import EXPLOITS, components, dense_rewards, grade, sla_met, training_reward, zero
from .provenance import (
    CALIBRATION_PATH,
    SEED_DATA_PATH,
    ProvenanceError,
    load_calibration,
    load_sealed,
    run_manifest,
    seal,
    source_hash,
    stream,
    verify_seal,
    verify_trial,
    write_sealed,
)
from .tasks import EVAL_TASKS, HELD_OUT_SEEDS, REPORTED_SEEDS, TASKS, TRAIN_TASKS, Task
from .tools import TOOL_NAMES, TOOL_SCHEMAS, Episode

_LAZY_EXPORTS = {
    "ANCHORS": ".policies",
    "HONEST_ANCHORS": ".policies",
    "run_policy": ".policies",
    "run_policy_episode": ".policies",
    "run_exploit": ".exploits",
    "play_exploiter_window": ".exploits",
    "build_calibration": ".calibrate",
    "calibrate_cell": ".calibrate",
    "get_cell": ".calibrate",
    "MCPToolServer": ".mcp_server",
    "tool_json_schema": ".mcp_server",
    "EpisodeRunner": ".harness.runner",
    "run_agent_episode": ".harness.runner",
    "run_eval_standard": ".harness.runner",
    "BridgeSession": ".human_recorder",
    "create_bridge_app": ".human_recorder",
    "record_from_bridge": ".human_recorder",
    "record_trajectory": ".human_recorder",
    "replay_trajectory": ".human_recorder",
    "verify_all_trajectories": ".human_recorder",
    "ConsoleState": ".web_console",
    "create_console_app": ".web_console",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ANCHORS",
    "BOSSES",
    "BridgeSession",
    "CALIBRATION_PATH",
    "ConsoleState",
    "EVAL_TASKS",
    "EXPLOITS",
    "HELD_OUT_SEEDS",
    "HONEST_ANCHORS",
    "ITEMS",
    "MCPToolServer",
    "MOBS",
    "RECIPES",
    "REPORTED_SEEDS",
    "SEED_DATA_PATH",
    "TASKS",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "TRAIN_TASKS",
    "Episode",
    "EpisodeRunner",
    "ProvenanceError",
    "Task",
    "World",
    "build_calibration",
    "calibrate_cell",
    "components",
    "create_bridge_app",
    "create_console_app",
    "dense_rewards",
    "get_cell",
    "grade",
    "load_calibration",
    "load_sealed",
    "play_exploiter_window",
    "record_from_bridge",
    "record_trajectory",
    "replay_trajectory",
    "run_agent_episode",
    "run_eval_standard",
    "run_exploit",
    "run_manifest",
    "run_policy",
    "run_policy_episode",
    "seal",
    "sla_met",
    "source_hash",
    "stream",
    "tool_json_schema",
    "training_reward",
    "verify_all_trajectories",
    "verify_seal",
    "verify_trial",
    "world_hash",
    "write_sealed",
    "zero",
]
