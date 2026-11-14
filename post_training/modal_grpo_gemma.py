"""Modal entrypoints: GRPO + LoRA training and a vLLM server for base + tuned Gemma.

    modal run    post_training/modal_grpo_gemma.py::train_bc       # behaviour cloning (recommended)
    modal run    post_training/modal_grpo_gemma.py::train_grpo     # open-loop GRPO (experimental)
    modal deploy post_training/modal_grpo_gemma.py                 # OpenAI-compatible server

Create the Hugging Face secret once (Gemma weights are gated):
    modal secret create huggingface HF_TOKEN=hf_...

The server exposes model "gemma-base" and, once an adapter exists on the volume,
"gemma-terraria-bc" / "gemma-terraria-grpo".
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "terrariabench-gemma-grpo"
BASE_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
LORA_OUTPUT_DIR = "/data/checkpoints/gemma-terraria-grpo-r32"
BC_OUTPUT_DIR = "/data/checkpoints/gemma-terraria-bc-r32"
ADAPTERS = {"gemma-terraria-bc": BC_OUTPUT_DIR, "gemma-terraria-grpo": LORA_OUTPUT_DIR}
GPU = os.environ.get("TB_GPU", "L40S")
REPO = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("terrariabench-weights-vol", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
ENV = {"HF_HOME": "/data/hf", "GEMMA_MODEL_ID": BASE_MODEL_ID}   # weights cached on the volume

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "transformers>=4.48.0", "trl>=0.14.0", "peft>=0.14.0",
                 "accelerate>=1.2.0", "datasets>=3.2.0", "torchvision", "pillow", "sentencepiece", "protobuf")
    .env(ENV)
    .add_local_python_source("terraria_bench", "post_training")
    # calibration.json sits next to the package, where provenance.py looks for it
    .add_local_dir(str(REPO / "artifacts"), "/root/artifacts")
)
serving_image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm>=0.7.2").env(ENV)


@app.function(image=training_image, gpu=GPU, timeout=4 * 3600, volumes={"/data": volume}, secrets=[hf_secret])
def train_grpo(num_steps: int = 36):
    from post_training.grpo_train import run_training

    run_training(num_steps, LORA_OUTPUT_DIR)
    volume.commit()
    print(f"saved LoRA adapter to {LORA_OUTPUT_DIR}")


@app.function(image=training_image, gpu=GPU, timeout=4 * 3600, volumes={"/data": volume}, secrets=[hf_secret])
def train_bc(epochs: float = 3.0):
    from post_training.bc_train import run_bc

    run_bc(BC_OUTPUT_DIR, epochs=epochs)
    volume.commit()
    print(f"saved LoRA adapter to {BC_OUTPUT_DIR}")


EVAL_CELLS = [  # seen worlds, unseen worlds of the train tasks, unseen eval tasks
    "woodland-scramble:7", "ore-rush:7",
    "woodland-scramble:17", "shelter-crunch:17", "slime-rain:17", "ore-rush:17",
    "first-night:7", "cavern-expedition:7",
]


@app.function(image=training_image, gpu=GPU, timeout=6 * 3600, volumes={"/data": volume}, secrets=[hf_secret])
def eval_bc(cells: str = ""):
    import json

    from post_training.eval_hf import evaluate

    os.makedirs("/data/eval", exist_ok=True)

    def save(rows):
        with open("/data/eval/bc_eval.json", "w") as f:
            json.dump(rows, f, indent=1)
        volume.commit()

    evaluate(BASE_MODEL_ID, BC_OUTPUT_DIR, cells.split(",") if cells else EVAL_CELLS, on_row=save)


@app.function(image=serving_image, gpu=GPU, timeout=3600, volumes={"/data": volume},
              secrets=[hf_secret], scaledown_window=300)
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=900)
def serve_vllm():
    import subprocess

    cmd = ["vllm", "serve", BASE_MODEL_ID, "--served-model-name", "gemma-base",
           "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16", "--max-model-len", "8192"]
    found = [f"{name}={path}" for name, path in ADAPTERS.items() if os.path.exists(path)]
    if found:
        cmd += ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", *found]
    print("launching:", " ".join(cmd))
    subprocess.Popen(cmd)


def run_single_eval_worker(agent_name: str, seed: int = 11, task_id: str = "first-night") -> dict:
    """Execute a single benchmark evaluation episode and settle telemetry to Aiven sink."""
    import time

    from post_training.model_clients import get_agent_by_name
    from telemetry.aiven_sink import TamperProofAivenSink

    sink = TamperProofAivenSink()
    run_id = f"eval_{agent_name}_seed_{seed}_{int(time.time())}"
    agent = get_agent_by_name(agent_name)
    eval_res = agent.evaluate(task_id=task_id, seed=seed)
    score = eval_res["score"]
    shelter_valid = score >= 90.0
    record = sink.record_settlement(
        run_id=run_id,
        agent_model=agent_name,
        seed=seed,
        window_idx=1,
        net_progression=score,
        shelter_valid=shelter_valid,
        reward_score=score,
    )
    return {
        "run_id": run_id,
        "agent": agent_name,
        "seed": seed,
        "score": score,
        "shelter_valid": shelter_valid,
        "hmac_signature": record.get("hmac_signature"),
    }


def run_parallel_judge_suite() -> list[dict]:
    """Run multi-model judge evaluations and settle signed records to Aiven telemetry."""
    judge_jobs = [
        ("reference", 11),
        ("do_nothing", 11),
        ("stop_early", 11),
        ("gemini", 11),
        ("codex", 11),
        ("gemma-base", 11),
        ("gemma-terraria-grpo", 11),
    ]
    results = []
    for agent_name, seed in judge_jobs:
        res = run_single_eval_worker(agent_name, seed)
        print(
            f"  [RUN] Agent: {agent_name:<20} | Seed: {seed} | Score: {res['score']:>6.2f}"
            f" | Valid Shelter: {str(res['shelter_valid']):<5} | HMAC: {str(res['hmac_signature'])[:10]}..."
        )
        results.append(res)
    return results


if __name__ == "__main__":
    run_parallel_judge_suite()
