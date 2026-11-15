"""Run a chat model against TerrariaBench tasks and keep everything needed to check the result.

    python -m evaluation.run_model --base-url https://inference.verda.com/v1 --key-env VERDA_API_KEY \
        --model kimi-k3-1-pro --tasks first-night eye-of-cthulhu --repeats 3

The model gets exactly what the Harbor/OpenCode agent gets: the task instruction (including every
scored checkpoint), the 24 tools with their parameters, and the full text of every tool result.
It plays one world, one attempt, one call per turn. No strategy hints are given.

Each episode is written to results/runs/<label>/<task>_seed<seed>_run<k>.json with the model id,
the endpoint host, every reply, every call and result, the sealed ledger and the official grade.
The leaderboard in RESULTS.md is built only from these files (evaluation/report.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from terraria_bench.grader import grade
from terraria_bench.provenance import load_calibration
from terraria_bench.tasks import TASKS
from terraria_bench.tools import TOOL_SCHEMAS, Episode

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results" / "runs"
FULL_RESULTS = 14          # most recent tool results kept verbatim in the conversation
RESULT_CHARS = 3500        # one tool result is never longer than this (the PNG is never sent)
MAX_BAD_IN_A_ROW = 6

_TOOL_LIST = "\n".join(
    f"- {name}({', '.join(f'{k}: {v}' for k, v in spec['params'].items())}): {spec['description']}"
    for name, spec in TOOL_SCHEMAS.items()
)
SYSTEM = (
    "You are playing a game through tools. Each turn you make exactly one tool call and then see its result.\n\n"
    "Tools:\n" + _TOOL_LIST + "\n\n"
    "Reply with one JSON object and nothing else:\n"
    '{"why": "<one sentence: what you expect this call to achieve>", "tool": "<tool name>", "args": {...}}'
)


def parse_reply(text: str):
    """Last JSON object in the reply that names a tool -> (tool, args, why), else None."""
    for m in reversed(list(re.finditer(r"\{", text or ""))):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            args = obj.get("args") or {}
            return (str(obj["tool"]), args if isinstance(args, dict) else {}, str(obj.get("why", ""))[:300])
    return None


def make_chat(base_url: str, model: str, api_key: str, temperature: float, max_tokens: int, min_interval: float = 0.0):
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "rate_limited": 0}
    last = [0.0]

    def chat(messages) -> str:
        body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, {
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "python-client"})
        err: Exception | None = None
        attempt = 0
        while attempt < 8:
            wait = last[0] + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.load(resp)
                u = data.get("usage") or {}
                usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                usage["calls"] += 1
                return data["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as exc:
                err = exc
                if exc.code == 429:  # the provider's rate limit is not the model's fault: wait and try again
                    usage["rate_limited"] += 1
                    try:
                        pause = float(json.load(exc)["error"].get("retry_after_s") or 20)
                    except Exception:
                        pause = 20.0
                    time.sleep(min(90.0, pause + 2))
                    if usage["rate_limited"] > 400:
                        break
                    continue
                attempt += 1
                time.sleep(min(60, 3 * 2 ** attempt))
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, OSError) as exc:
                err = exc
                attempt += 1
                time.sleep(min(60, 3 * 2 ** attempt))
        raise RuntimeError(f"endpoint failed after retries: {err!r}")

    chat.usage = usage  # type: ignore[attr-defined]
    return chat


def _result_text(result: dict) -> str:
    text = result.get("text")
    if not isinstance(text, str):
        text = json.dumps({k: v for k, v in result.items() if k != "viewport_png_b64"}, sort_keys=True)
    return text[:RESULT_CHARS]


def _conversation(task_id: str, turns: list[dict], header: str) -> list[dict]:
    """System + task + (compact summary of old turns) + the recent turns verbatim."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASKS[task_id].prompt + f"\n\nNow: {header}\nYour first call:"}]
    played = [t for t in turns if t.get("tool")]
    old, recent = played[:-FULL_RESULTS], played[-FULL_RESULTS:]
    if old:
        lines = [f"{t['n']}. w{t['window']} {t['tool']}({json.dumps(t['args'], separators=(',', ':'))}) -> "
                 f"{'ok' if t.get('ok') else 'FAILED'}: {t['result'][t['result'].find(']') + 1:][:140].strip()}" for t in old]
        msgs.append({"role": "user", "content": "Your earlier calls in this run (oldest first):\n" + "\n".join(lines)})
    for t in recent:
        msgs.append({"role": "assistant", "content": json.dumps({"why": t["why"], "tool": t["tool"], "args": t["args"]})})
        msgs.append({"role": "user", "content": t["result"]})
    if turns and not turns[-1].get("tool"):
        msgs.append({"role": "user", "content": turns[-1]["result"]})
    return msgs


def play(chat, task_id: str, seed: int, progress=None) -> dict:
    ep = Episode(task_id, seed, redact_seed=True)
    task = TASKS[task_id]
    turns: list[dict] = []
    bad_run = bad_total = 0
    max_turns = int(task.windows * task.budget * 1.6) + 20
    while not ep.done and len(turns) < max_turns and bad_run < MAX_BAD_IN_A_ROW:
        window, ops = ep.world.window, ep.ops_left
        t0 = time.time()
        reply = chat(_conversation(task_id, turns, ep.header()))
        row = {"n": len(turns) + 1, "window": window, "ops_left": ops, "reply": reply[:1200],
               "seconds": round(time.time() - t0, 1)}
        action = parse_reply(reply)
        if action is None or action[0] not in TOOL_SCHEMAS:
            bad_run, bad_total = bad_run + 1, bad_total + 1
            row["result"] = ("That was not a valid call (it cost nothing). Reply with one JSON object: "
                             '{"why": "...", "tool": "<one of the listed tools>", "args": {...}}')
            turns.append(row)
            continue
        tool, args, why = action
        try:
            result = ep.step(tool, **args)
            row.update(tool=tool, args=args, why=why, ok=bool(result.get("ok", True)), result=_result_text(result))
            bad_run = 0
        except Exception as exc:  # wrong argument names or types: the server would reject the call the same way
            bad_run, bad_total = bad_run + 1, bad_total + 1
            row.update(tool=tool, args=args, why=why, ok=False,
                       result=f"{ep.header()}\nCall rejected (it cost nothing): {str(exc)[:300]}")
        turns.append(row)
        if progress and len(turns) % 20 == 0:
            progress(f"turn {len(turns)}, window {ep.world.window}/{task.windows}")
    return {"ledger": ep.finalize(), "turns": turns, "bad_replies": bad_total,
            "stopped": "episode finished" if ep.done else ("too many invalid replies" if bad_run >= MAX_BAD_IN_A_ROW
                                                           else "turn limit")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-env", default="OPENAI_API_KEY")
    ap.add_argument("--label", default=None, help="folder name under results/runs (default: the model id)")
    ap.add_argument("--tasks", nargs="+", default=["first-night", "eye-of-cthulhu"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between requests of one episode")
    a = ap.parse_args(argv)

    key = os.environ.get(a.key_env)
    if not key:
        raise SystemExit(f"{a.key_env} is not set")
    cal = load_calibration()
    label = a.label or re.sub(r"[^A-Za-z0-9_.-]+", "_", a.model)
    out_dir = RUNS / label
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(job) -> dict:
        task_id, k = job
        path = out_dir / f"{task_id}_seed{a.seed}_run{k}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["summary"]
        chat = make_chat(a.base_url, a.model, key, a.temperature, a.max_tokens, a.min_interval)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.time()
        try:
            out = play(chat, task_id, a.seed, lambda msg: print(f"[{task_id} run {k}] {msg}", flush=True))
        except RuntimeError as exc:
            print(json.dumps({"task": task_id, "run": k, "error": str(exc)}), flush=True)
            return {"task": task_id, "run": k, "error": str(exc)}
        g = grade(out["ledger"], cal["cells"][f"{task_id}:{a.seed}"])
        summary = {"model": a.model, "label": label, "task": task_id, "seed": a.seed, "run": k,
                   "composite": g["composite"], "eye_progress_pct": g["eye_progress_pct"], "deaths": g["deaths"],
                   "sla": g["sla"], "survival": g["survival"], "business": g["business"],
                   "tool_calls": sum(1 for t in out["turns"] if t.get("tool")), "bad_replies": out["bad_replies"],
                   "stopped": out["stopped"], "minutes": round((time.time() - t0) / 60, 1)}
        record = {"summary": summary,
                  "provenance": {"model": a.model, "endpoint_host": urlparse(a.base_url).netloc, "started_utc": started,
                                 "temperature": a.temperature, "usage": chat.usage, "runner": "evaluation/run_model.py",
                                 "calibration_sha256": cal.get("sha256")},
                  "grade": g, "turns": out["turns"], "ledger": out["ledger"]}
        path.write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
        print(json.dumps(summary), flush=True)
        return summary

    jobs = [(t, k) for t in a.tasks for k in range(1, a.repeats + 1)]
    with ThreadPoolExecutor(a.threads) as pool:
        rows = list(pool.map(run, jobs))
    ok = [r for r in rows if "composite" in r]
    print(json.dumps({"model": a.model, "episodes": len(ok), "errors": len(rows) - len(ok),
                      "mean_composite": round(sum(r["composite"] for r in ok) / len(ok), 4) if ok else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
