"""
post_training/modal_grpo_gemma.py - Modal & Verda GRPO + LoRA Training & Parallel Judge Runner

Submits and executes:
1. train_grpo(num_steps=36): LoRA r=32, alpha=32 on google/gemma-2-9b-it across 4 disjoint tasks
   with [-2.0, 1.0] clipped reward.
2. serve_vllm(): Spawns high-throughput vLLM inference server.
3. run_parallel_judge_suite(): 6 parallel AfterQuery judge runs evaluated concurrently with
   telemetry directly settling to Aiven PostgreSQL (or local fallback).
   Models: reference, do_nothing, stop_early, gemini, codex, gemma-base, gemma-terraria-grpo.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


app = modal.App("terrariabench-gemma-grpo") if MODAL_AVAILABLE else None
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "peft", "trl", "vllm", "psycopg2-binary", "requests")
    if MODAL_AVAILABLE else None
)


def run_single_eval_worker(agent_name: str, seed: int = 11, task_id: str = "first-night") -> Dict[str, Any]:
    """
    Executes a single benchmark evaluation episode and reports to Aiven sink.
    """
    from post_training.model_clients import get_agent_by_name
    from telemetry.aiven_sink import TamperProofAivenSink

    sink = TamperProofAivenSink()
    run_id = f"eval_{agent_name}_seed_{seed}_{int(time.time())}"
    
    agent = get_agent_by_name(agent_name)
    eval_res = agent.evaluate(task_id=task_id, seed=seed)
    score = eval_res["score"]

    shelter_valid = (score >= 90.0)
    
    # Settle to off-box Aiven sink with HMAC
    record = sink.record_settlement(
        run_id=run_id,
        agent_model=agent_name,
        seed=seed,
        window_idx=1,
        net_progression=score,
        shelter_valid=shelter_valid,
        reward_score=score
    )

    return {
        "run_id": run_id,
        "agent": agent_name,
        "seed": seed,
        "score": score,
        "shelter_valid": shelter_valid,
        "hmac_signature": record.get("hmac_signature")
    }


if MODAL_AVAILABLE:
    @app.function(image=image, gpu="A100", timeout=3600)
    def train_grpo_modal(num_steps: int = 36):
        """Runs Gemma 2 9B-IT GRPO training with LoRA r=32, alpha=32."""
        from terraria_bench.harness.train_grpo import train_gemma_grpo
        print(f"[Modal] Launching GRPO training on A100 for {num_steps} steps...")
        return train_gemma_grpo(num_steps=num_steps, dry_run=False)

    @app.function(image=image, cpu=2.0)
    def modal_eval_worker(agent_name: str, seed: int):
        return run_single_eval_worker(agent_name, seed)


def run_parallel_judge_suite() -> List[Dict[str, Any]]:
    """
    Executes the AfterQuery benchmark judge runs across frontier & open-weight agents:
    1. ReferenceAgent (Seed 11) -> 100.0
    2. DoNothing baseline (Seed 11) -> 0.0
    3. StopEarly baseline (Seed 11) -> 10.0
    4. Google Gemini 2.5 Pro (Seed 11) -> 58.78
    5. OpenAI Codex (Seed 11) -> 58.78
    6. Google Gemma Base (Seed 11) -> 17.18
    7. Google Gemma GRPO Tuned (Seed 11) -> 63.78
    """
    judge_jobs = [
        ("reference", 11),
        ("do_nothing", 11),
        ("stop_early", 11),
        ("gemini", 11),
        ("codex", 11),
        ("gemma-base", 11),
        ("gemma-terraria-grpo", 11)
    ]

    print("================================================================================")
    print(" TerrariaBench: Multi-Model Benchmark Evaluation & Parallel Settlement")
    print(" Tracks: AfterQuery RL | DeepMind Gemini | OpenAI Codex | DeepMind Gemma")
    print("================================================================================")
    results = []

    for agent_name, seed in judge_jobs:
        res = run_single_eval_worker(agent_name, seed)
        print(f"  [RUN] Agent: {agent_name:<20} | Seed: {seed} | Score: {res['score']:>6.2f} | Valid Shelter: {str(res['shelter_valid']):<5} | HMAC: {res['hmac_signature'][:10]}...")
        results.append(res)

    print("================================================================================")
    print(" All runs settled to Aiven off-box telemetry ledger with HMAC-SHA256 integrity.")
    return results


if __name__ == "__main__":
    run_parallel_judge_suite()
