"""GRPO + LoRA (rank 32) training loop. Runs anywhere with a GPU.

    python -m post_training.grpo_train --steps 36 --out ./checkpoints/gemma-terraria-grpo-r32

Modal wraps this in post_training/modal_grpo_gemma.py; on a Verda instance run
it directly. Needs HF_TOKEN in the environment (Gemma weights are gated).
"""
from __future__ import annotations

import argparse
import os

from .terraria_reward import grpo_reward, training_rows

BASE_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
GROUP_SIZE = 6          # rollouts per prompt
PROMPTS_PER_STEP = 5    # prompts per optimizer step
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def run_training(num_steps: int = 36, output_dir: str = "./checkpoints/gemma-terraria-grpo-r32",
                 model_id: str = BASE_MODEL_ID) -> str:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(training_rows(num_steps * PROMPTS_PER_STEP))
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=GROUP_SIZE,
        gradient_accumulation_steps=PROMPTS_PER_STEP,   # 5 prompts x 6 rollouts per step
        num_generations=GROUP_SIZE,
        max_completion_length=384,
        max_steps=num_steps,
        beta=0.0,                                       # no KL penalty
        temperature=1.0,
        logging_steps=1,
        save_steps=12,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model_id, reward_funcs=grpo_reward, args=args,
                          train_dataset=dataset, peft_config=peft_config, processing_class=tokenizer)
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--out", default="./checkpoints/gemma-terraria-grpo-r32")
    ap.add_argument("--model", default=BASE_MODEL_ID)
    a = ap.parse_args()
    print("saved", run_training(a.steps, a.out, a.model))
