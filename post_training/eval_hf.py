"""Base-vs-adapter closed-loop evaluation with local transformers weights (GPU).

Same prompt, same greedy decoding, same cells for both; graded with the official composite.
"""
from __future__ import annotations

import json
import time

from terraria_bench.grader import grade
from terraria_bench.provenance import load_calibration

from .closed_loop import play


def evaluate(model_id: str, adapter_dir: str, cells: list[str], on_row=None) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    except ValueError:
        base = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir).eval()

    def generate(messages) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=96, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    cal = load_calibration()
    rows = []
    for cell in cells:
        task_id, seed = cell.rsplit(":", 1)
        for name in ("base", "tuned"):
            t0 = time.time()
            if name == "base":
                with model.disable_adapter():
                    out = play(generate, task_id, int(seed))
            else:
                out = play(generate, task_id, int(seed))
            g = grade(out["ledger"], cal["cells"][cell])
            row = {"cell": cell, "model": name, "composite": g["composite"], "passed": g.get("passed"),
                   "threshold": cal["cells"][cell]["threshold"], "deaths": out["ledger"]["deaths"],
                   "turns": out["turns"], "bad_replies": out["bad"], "seconds": round(time.time() - t0)}
            print(json.dumps(row), flush=True)
            rows.append(row)
            if on_row:
                on_row(rows)
    return rows
