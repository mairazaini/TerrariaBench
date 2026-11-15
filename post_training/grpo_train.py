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
                 model_id: str = BASE_MODEL_ID, init_adapter: str = "", beta: float = 0.05,
                 lr: float = 2e-5) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    peft_config = None
    if init_adapter and os.path.exists(init_adapter):
        print(f"initializing GRPO from BC adapter: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        names = [n for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in LORA_TARGETS]
        text_targets = [n for n in names if not any(k in n for k in ("vision", "audio", "embed_"))]
        peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
                                 task_type="CAUSAL_LM", target_modules=text_targets or LORA_TARGETS)

    dataset = Dataset.from_list(training_rows(num_steps * PROMPTS_PER_STEP))
    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=GROUP_SIZE,
        gradient_accumulation_steps=PROMPTS_PER_STEP,   # 5 prompts x 6 rollouts per step
        num_generations=GROUP_SIZE,
        max_completion_length=384,
        max_steps=num_steps,
        beta=beta,
        temperature=1.0,
        logging_steps=1,
        save_steps=12,
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, reward_funcs=grpo_reward, args=args,
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
