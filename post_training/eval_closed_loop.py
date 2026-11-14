"""Play closed-loop episodes against any OpenAI-compatible endpoint and grade them
with the official composite (never the training reward).

    python -m post_training.eval_closed_loop --base-url https://api.featherless.ai/v1 \
        --key-env FEATHERLESS_API_KEY --model google/gemma-4-31B-it --cells ore-rush:7 first-night:7
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from terraria_bench.grader import grade
from terraria_bench.provenance import load_calibration

from .closed_loop import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_STRATEGY, play


def make_generate(base_url: str, model: str, api_key: str, temperature: float = 0.2):
    def generate(messages) -> str:
        body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": 96}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, {
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "python-client"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.load(resp)["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"endpoint failed after retries: {err!r}")
    return generate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-env", default="OPENAI_API_KEY")
    ap.add_argument("--cells", nargs="+", required=True, help="task:seed ...")
    ap.add_argument("--no-strategy", action="store_true", help="drop the strategy notes from the prompt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args(argv)

    cal = load_calibration()
    gen = make_generate(a.base_url, a.model, os.environ.get(a.key_env, "EMPTY"))
    system = SYSTEM_PROMPT_NO_STRATEGY if a.no_strategy else SYSTEM_PROMPT

    def run(cell: str) -> dict:
        task_id, seed = cell.rsplit(":", 1)
        t0 = time.time()
        try:
            out = play(gen, task_id, int(seed), system)
        except RuntimeError as exc:
            return {"cell": cell, "error": str(exc)}
        g = grade(out["ledger"], cal["cells"][cell])
        row = {"cell": cell, "composite": g["composite"], "passed": g.get("passed"), "deaths": out["ledger"]["deaths"],
               "turns": out["turns"], "bad_replies": out["bad"], "milestones": out["ledger"]["milestones"],
               "seconds": round(time.time() - t0)}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(a.threads) as pool:
        rows = list(pool.map(run, a.cells))
    ok = [r for r in rows if "composite" in r]
    summary = {"model": a.model, "strategy_notes": not a.no_strategy, "episodes": len(ok),
               "errors": len(rows) - len(ok),
               "mean_composite": round(sum(r["composite"] for r in ok) / len(ok), 4) if ok else None}
    print(json.dumps(summary))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
