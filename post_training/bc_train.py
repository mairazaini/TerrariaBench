"""Behaviour-cloning LoRA (rank 32) on the verified closed-loop expert demonstrations.

    python -m post_training.bc_train --out ./checkpoints/gemma-terraria-bc-r32

Loss is taken on the assistant's tool call only; the prompt tokens are masked.
Needs a GPU. Gemma 4 is Apache-2.0 and ungated; older Gemma needs a licensed HF_TOKEN.
"""
from __future__ import annotations

import argparse
import os

from .closed_loop import expert_examples

BASE_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_TOKENS = 6144


def run_bc(output_dir: str = "./checkpoints/gemma-terraria-bc-r32", model_id: str = BASE_MODEL_ID,
           epochs: float = 3.0, lr: float = 1e-4) -> str:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    examples, report = expert_examples()
    assert examples and all(r["kept"] for r in report), report
    print(f"{len(examples)} verified examples from {len(report)} expert episodes")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex):
        prompt = tok.apply_chat_template(ex["messages"][:1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(ex["messages"], tokenize=False)
        assert full.startswith(prompt), "chat template is not prefix-stable"
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = (p + c)[-MAX_TOKENS:]
        labels = ([-100] * len(p) + c)[-MAX_TOKENS:]
        return {"input_ids": ids, "labels": labels}

    data = [encode(e) for e in examples]
    print("longest example:", max(len(d["input_ids"]) for d in data), "tokens")

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))
        return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    except ValueError:   # multimodal checkpoints (Gemma 3/4) are not registered as plain causal LMs
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    # Adapt only the text decoder: skip vision / audio tower projections that share the same names.
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in LORA_TARGETS]
    text = [n for n in names if not any(k in n for k in ("vision", "audio", "embed_"))]
    print(f"LoRA on {len(text)} of {len(names)} matching layers, e.g. {text[:2]}")
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=text))
    model.enable_input_require_grads()   # needed for gradient checkpointing with a frozen base
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
                             per_device_train_batch_size=1, gradient_accumulation_steps=16,
                             gradient_checkpointing=True,
                             lr_scheduler_type="cosine", warmup_steps=6, logging_steps=5,
                             bf16=True, save_strategy="no", report_to="none", seed=7)
    Trainer(model=model, args=args, train_dataset=data, data_collator=collate).train()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="./checkpoints/gemma-terraria-bc-r32")
    ap.add_argument("--model", default=BASE_MODEL_ID)
    ap.add_argument("--epochs", type=float, default=3.0)
    a = ap.parse_args()
    print("saved", run_bc(a.out, a.model, a.epochs))
