"""
agents/reference_and_baselines.py - Reference & Baseline Agents for AfterQuery Benchmark

Directly wires into TerrariaBench's calibrated policy evaluation engine:
- ReferenceAgent: 'smart-triage' macro policy scoring 100.0 across calibrated seeds.
- DoNothingAgent: 'do-nothing' policy scoring exactly 0.00.
- StopEarlyAgent: 'absentee' policy scoring ~10.0 - 15.0.
"""

from typing import Dict, Any, Optional
from terraria_bench.policies import run_policy
from terraria_bench.grader import grade
from terraria_bench.tools import Episode
import json
import os

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "..", "artifacts", "calibration.json")
_CALIBRATION_DATA = None


def get_calibration():
    global _CALIBRATION_DATA
    if _CALIBRATION_DATA is None:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                _CALIBRATION_DATA = json.load(f)
        else:
            _CALIBRATION_DATA = {"cells": {}}
    return _CALIBRATION_DATA


class BaseTerrariaAgent:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> Dict[str, Any]:
        raise NotImplementedError


class DoNothingAgent(BaseTerrariaAgent):
    """Takes zero world actions. Baseline score: 0.0."""
    def __init__(self):
        super().__init__("do_nothing")

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> Dict[str, Any]:
        cal = get_calibration()
        cell = cal.get("cells", {}).get(f"{task_id}:{seed}")
        ledger = run_policy("do-nothing", task_id, seed)
        res = grade(ledger, cell)
        return {
            "score": round(res["composite"] * 100.0, 2),
            "delta_net": res.get("delta_net", 0.0),
            "ledger": ledger
        }


class StopEarlyAgent(BaseTerrariaAgent):
    """Halts early / absentee baseline policy. Baseline score: ~10.0 - 15.0."""
    def __init__(self):
        super().__init__("stop_early")

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> Dict[str, Any]:
        cal = get_calibration()
        cell = cal.get("cells", {}).get(f"{task_id}:{seed}")
        ledger = run_policy("absentee", task_id, seed)
        res = grade(ledger, cell)
        return {
            "score": round(res["composite"] * 100.0, 2),
            "delta_net": res.get("delta_net", 0.0),
            "ledger": ledger
        }


class ReferenceAgent(BaseTerrariaAgent):
    """
    Deterministic reference expert policy ('smart-triage') that perfectly passes all
    AfterQuery SLA gates, yielding a reproducible 100.0 score across seeds.
    """
    def __init__(self):
        super().__init__("reference")

    def evaluate(self, task_id: str = "first-night", seed: int = 11) -> Dict[str, Any]:
        cal = get_calibration()
        cell = cal.get("cells", {}).get(f"{task_id}:{seed}")
        ledger = run_policy("smart-triage", task_id, seed)
        res = grade(ledger, cell)
        return {
            "score": round(res["composite"] * 100.0, 2),
            "delta_net": res.get("delta_net", 0.0),
            "ledger": ledger
        }
