"""One command, no API key, about a minute: everything a judge re-runs, except the model itself.

    python -m evaluation.judge

1. Reference run, do-nothing and stop-early (at 25/50/75% of the reference's own calls) on every
   evaluation task and every reported seed. The world is deterministic, so these numbers are exact.
2. The twelve catalogued reward exploits, each attempted for real and reported closed or open.
3. Reads the saved model transcripts in results/runs/ and re-grades every one from its sealed ledger,
   so a reported model score can be checked without calling the model.

Writes RESULTS.md and exits non-zero if any integrity property fails.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from terraria_bench.engine import MILESTONE_VALUE
from terraria_bench.exploits import EXPLOIT_NAMES, run_exploit
from terraria_bench.grader import EXPLOITS, grade
from terraria_bench.policies import run_policy_episode
from terraria_bench.provenance import load_calibration
from terraria_bench.tasks import EVAL_TASKS, HARDCORE_TASKS, REPORTED_SEEDS, TASKS, describe_checkpoint
from terraria_bench.tools import Episode

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results" / "runs"
HARBOR = REPO / "results" / "harbor"
BC_EVAL = REPO / "artifacts" / "post_training" / "bc_eval.json"
OUT = REPO / "RESULTS.md"
STOP_POINTS = (0.25, 0.5, 0.75)
ALL_TASKS = tuple(HARDCORE_TASKS) + tuple(EVAL_TASKS)  # the Hardcore pair is the headline, so it goes first
LOOK_TOOLS = {"terraria_status", "observe_viewport", "scan_biome_and_ores", "inspect_inventory",
              "list_craftable_recipes", "inspect_housing_and_arena", "query_threat_telemetry",
              "record_tactical_checkpoint"}
MODEL_NAMES = {"kimi-k3-1-pro": "Kimi K3.1 Pro", "deepseek-4-1-flash-1017": "DeepSeek V4.1 Flash",
               "google_gemma-4-31B-it": "Gemma 4 31B"}


def _prefix_score(actions: list[dict], task_id: str, seed: int, frac: float, cell: dict) -> float:
    ep = Episode(task_id, seed)
    for a in actions[: int(len(actions) * frac)]:
        if ep.done:
            break
        ep.step(a["tool"], **a["args"])
    return grade(ep.finalize(), cell)["composite"]


RANDOM_EPISODES = 10


def _random_score(task_id: str, seed: int, cell: dict) -> float:
    """Mean score of an agent that calls random tools with random, often absurd, arguments until the
    episode ends. It is the crudest possible 'strategy that avoids the task'."""
    import random
    from terraria_bench.engine import ITEMS
    from terraria_bench.tools import TOOL_SCHEMAS
    rng = random.Random(f"random-baseline:{task_id}:{seed}")
    items = sorted(ITEMS)

    def value(name: str, kind: str):
        if any(k in str(kind) for k in ("int", "float", "number")):
            return rng.choice([-10 ** 9, -1, 0, 1, 3, 50, 10 ** 6, 2.5])
        if "bool" in str(kind):
            return rng.choice([True, False])
        if "item" in name:
            return rng.choice(items)
        return rng.choice(["", "shelter", "arena", "eye", "wall_of_flesh", "chest", "stone", "gold_ore", "home"])

    scores = []
    for _ in range(RANDOM_EPISODES):
        ep, n = Episode(task_id, seed), 0
        while not ep.done and n < 3000:
            n += 1
            tool = rng.choice(sorted(TOOL_SCHEMAS))
            args = {k: value(k, t) for k, t in TOOL_SCHEMAS[tool]["params"].items() if rng.random() < 0.7}
            try:
                ep.step(tool, **args)
            except (TypeError, ValueError, KeyError, IndexError, AttributeError):
                pass  # the game server rejects these the same way
        scores.append(grade(ep.finalize(), cell)["composite"])
    return sum(scores) / len(scores)


def baselines(cal: dict) -> dict:
    """{task: {"reference": [per seed], "do_nothing": [...], "stop_25": [...], ..., "calls": n, "ref_ledger": ...}}"""
    out = {}
    for task_id in ALL_TASKS:
        row = defaultdict(list)
        for seed in REPORTED_SEEDS:
            cell = cal["cells"][f"{task_id}:{seed}"]
            ref = run_policy_episode("smart-triage", task_id, seed)
            actions = list(ref.actions_log)
            ledger = ref.finalize(cal)
            row["reference"].append(grade(ledger, cell)["composite"])
            row["do_nothing"].append(grade(run_policy_episode("do-nothing", task_id, seed).finalize(cal), cell)["composite"])
            for frac in STOP_POINTS:
                row[f"stop_{int(frac * 100)}"].append(_prefix_score(actions, task_id, seed, frac, cell))
            if seed == REPORTED_SEEDS[0]:
                row["random"] = _random_score(task_id, seed, cell)
                row["calls"] = len(actions)
                row["ref_ledger"] = ledger
                row["ref_tools"] = Counter(a["tool"] for a in actions)
                row["threshold"] = cell.get("threshold")
        out[task_id] = dict(row)
    return out


def replay_ledger(rec: dict, cal: dict) -> dict:
    """Feed the transcript's recorded calls back into a fresh world and seal the result. The world is
    deterministic, so this is the ledger that run produces, whatever the file claims."""
    s = rec["summary"]
    ep = Episode(s["task"], s["seed"], redact_seed=True)
    for turn in rec["turns"]:
        if not turn.get("tool") or ep.done:
            continue
        try:
            ep.step(turn["tool"], **turn["args"])
        except Exception:  # calls the server rejected were rejected here too and cost nothing
            pass
    return ep.finalize(cal)


def load_model_runs(cal: dict, reseal: bool = False) -> dict:
    """{label: {task: [record, ...]}}. Every score is computed here, from a replay of the recorded calls;
    ``replay_matches`` says whether the ledger saved in the file is byte-identical to the replayed one.
    ``reseal`` rewrites files whose saved ledger no longer matches (after an engine change) and says so
    in their provenance."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for path in sorted(RUNS.glob("*/*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        s = rec["summary"]
        ledger = replay_ledger(rec, cal)
        g = grade(ledger, cal["cells"][f"{s['task']}:{s['seed']}"])
        saved = rec["ledger"]
        same_play = all(saved.get(k) == ledger.get(k) for k in ("windows", "milestones", "deaths", "phi_final", "state_hash"))
        if reseal and not (same_play and abs(s["composite"] - g["composite"]) < 1e-9):
            rec["provenance"]["resealed"] = ("ledger and score recomputed by replaying the recorded calls after an "
                                             "engine fix; the calls themselves are unchanged")
            rec["ledger"], rec["grade"] = ledger, g
            for k in ("composite", "eye_progress_pct", "deaths", "sla", "survival", "business"):
                s[k] = g[k]
            same_play = True
            path.write_text(json.dumps({k: rec[k] for k in ("summary", "provenance", "grade", "turns", "ledger")},
                                       indent=1), encoding="utf-8", newline="\n")
        rec["ledger"] = ledger
        rec["regraded"] = g["composite"]
        rec["replay_matches"] = same_play
        rec["path"] = path.relative_to(REPO).as_posix()
        out[s["label"]][s["task"]].append(rec)
    return out


def diagnose(rec: dict, ref_ledger: dict) -> dict:
    """What the run did with its budget, straight from the transcript and the sealed ledger."""
    task = TASKS[rec["summary"]["task"]]
    calls = [t for t in rec["turns"] if t.get("tool")]
    paid = [t for t in calls if t["tool"] != "terraria_status"]
    windows = rec["ledger"]["windows"]
    used = sum(w["ops_used"] for w in windows)
    budget = task.windows * task.budget
    missed = [f"w{w['w']}: {describe_checkpoint(kind)}" for w in windows for kind, met in w["sla"] if not met]
    ref_ms, got_ms = set(ref_ledger["milestones"]), set(rec["ledger"]["milestones"])
    lost = sorted(((MILESTONE_VALUE[m], m) for m in ref_ms - got_ms if m in MILESTONE_VALUE), reverse=True)
    return {
        "ops_unused_pct": round(100 * (budget - used) / budget),
        "look_pct": round(100 * sum(1 for t in paid if t["tool"] in LOOK_TOOLS) / max(1, len(paid))),
        "failed_pct": round(100 * sum(1 for t in paid if not t.get("ok")) / max(1, len(paid))),
        "deaths": rec["ledger"]["deaths"],
        "checkpoints_missed": missed,
        "checkpoints_total": len(task.sla),
        "missing_milestones": [m for _, m in lost],
        "idle_windows": sum(1 for w in windows if w["ops_used"] <= 2),
    }


def first_death(rec: dict) -> dict | None:
    """How a run first died, from its own transcript: the game's stated cause, the model's last stated
    reason, and whether it had changed the terrain near its house since it last verified the house."""
    turns = [t for t in rec["turns"] if t.get("tool")]
    for w in rec["ledger"]["windows"]:
        if not w["deaths"]:
            continue
        in_window = [t for t in turns if t["window"] == w["w"]]
        if not in_window:
            return None
        last = in_window[-1]
        before = [t for t in turns if t["n"] <= last["n"]]
        notes = " ".join(w["notes"])
        built = max((t["n"] for t in before if t["tool"] == "build_blueprint"
                     and t["args"].get("preset") != "shelter"), default=0)
        checked = max((t["n"] for t in before if t["tool"] == "inspect_housing_and_arena"
                       or (t["tool"] == "build_blueprint" and t["args"].get("preset") == "shelter")), default=0)
        if "overwhelmed" in notes or (last["tool"] == "engage_combat_burst" and "caught in the open" not in notes):
            cause = "boss fight not finished by the end of the window"
        elif built > checked:
            cause = "house broken by its own later building, never re-checked"
        else:
            cause = "window ended away from valid housing"
        said = last.get("why", "")
        return {"window": w["w"], "of": len(rec["ledger"]["windows"]), "cause": cause, "last_tool": last["tool"],
                "said": said, "claimed_safe": any(k in said.lower() for k in ("inside the", "inside valid", "safely", "safe "))}
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def _rng(xs):
    return f"{_f(_mean(xs))} ({_f(min(xs), 2)}-{_f(max(xs), 2)})" if xs else "-"


def harbor_results() -> list[dict]:
    """One row per Harbor trial saved under results/harbor*/: task, agent, model, reward, deaths, calls."""
    rows = []
    for trial in sorted(REPO.glob("results/harbor*/*/*/result.json")):
        try:
            d = json.loads(trial.read_text(encoding="utf-8"))
            agent = (d.get("config") or {}).get("agent") or {}
            grade_file = trial.parent / "verifier" / "grade.json"
            g = json.loads(grade_file.read_text(encoding="utf-8")) if grade_file.exists() else {}
            reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            rows.append({"task": d.get("task_name", "?"), "agent": agent.get("name", "?"),
                         "model": (agent.get("model_name") or "").split("/")[-1], "reward": reward,
                         "deaths": (g.get("grade") or {}).get("deaths"), "calls": g.get("tool_calls"),
                         "error": bool(d.get("exception_info")), "path": trial.parent.relative_to(REPO).as_posix()})
        except Exception:  # an unfinished or foreign folder is simply not reported
            continue
    return rows


def build(cal: dict, reseal: bool = False) -> tuple[str, list[str]]:
    problems: list[str] = []
    base = baselines(cal)
    runs = load_model_runs(cal, reseal)
    L: list[str] = []
    add = L.append

    add("# Results")
    add("")
    add("Written by `python -m evaluation.judge`. Sections 1, 3 and 4 are recomputed from scratch on every run "
        "(deterministic, no API key). Section 2 is recomputed by replaying the calls recorded in each transcript "
        "in `results/runs/` through the engine; a file whose saved ledger differs from its replay fails the run. "
        "Each file holds the model id, the endpoint, every reply, every tool call and result, and the sealed ledger.")
    add("")

    # ------------------------------------------------------------------ 1. reference and baselines
    add("## 1. Reference run and the two baselines")
    add("")
    add(f"Composite score, mean over seeds {', '.join(map(str, REPORTED_SEEDS))}. The reference is the scripted "
        "`smart-triage` policy, replayed through the same 24 tools the agent uses "
        "(`harbor run -p harbor/tasks/<task> -a oracle`). Stop-early replays the first N% of the reference's own "
        "calls and then quits, so it is the *best possible* way to stop early.")
    add("")
    add("| Task | Ops | Reference | Do nothing | Random calls | Stop at 25% | Stop at 50% | Stop at 75% |")
    add("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for task_id, b in base.items():
        t = TASKS[task_id]
        add(f"| {task_id} | {t.windows * t.budget} | {_f(_mean(b['reference']))} | {_f(_mean(b['do_nothing']))} | {_f(b['random'])} | "
            + " | ".join(_f(_mean(b[f'stop_{int(f * 100)}'])) for f in STOP_POINTS) + " |")
        if b["random"] >= 0.5 * min(b["reference"]):
            problems.append(f"random play scores {b['random']:.3f} on {task_id}")
        if max(b["do_nothing"]) > 0.05:
            problems.append(f"do-nothing scores {max(b['do_nothing'])} on {task_id}")
        for i in range(len(REPORTED_SEEDS)):
            chain = [b["do_nothing"][i]] + [b[f"stop_{int(f * 100)}"][i] for f in STOP_POINTS] + [b["reference"][i]]
            if max(chain[:-1]) >= chain[-1]:
                problems.append(f"stopping early is not worse than finishing on {task_id} seed {REPORTED_SEEDS[i]}: {chain}")
    add("")
    add("Doing nothing scores 0 everywhere, and on every task and seed the full reference run beats every "
        f"stop-early point. *Random calls* is the mean of {RANDOM_EPISODES} episodes of random tools with random "
        "arguments on seed 7. It is highest on wall-of-flesh, where the player starts with a full gold kit and a "
        "random agent sometimes summons and damages the boss; that task discriminates least and is not one of "
        "the two headline tasks.")
    add("")
    hb = harbor_results()
    scripted = [r for r in hb if r["agent"] in ("oracle", "nop")]
    if scripted:
        add("The same baselines run through Harbor in Docker on this repository (`results/harbor/`):")
        add("")
        add("| Harbor task | Agent | Reward |")
        add("|:---|:---|---:|")
        for r in scripted:
            who = {"oracle": "reference (oracle)", "nop": "do nothing (nop)"}[r["agent"]]
            if r["task"].endswith("-stop-early"):
                who = "stop early (first quarter of the reference)"
            add(f"| {r['task']} | {who} | {_f(r['reward'])} |")
            if r["error"]:
                problems.append(f"Harbor trial {r['path']} raised an exception")
        add("")

    # ------------------------------------------------------------------ 2. models
    add("## 2. The gap: model runs against the reference")
    add("")
    if not runs:
        add("No model transcripts in `results/runs/` yet. Produce them with `python -m evaluation.run_model`.")
    else:
        tasks_run = [t for t in ALL_TASKS if any(t in by for by in runs.values())]
        add("### A minimal harness: one JSON tool call per turn")
        add("")
        add("`evaluation/run_model.py`: the model gets the task text (which states the scoring rule and every "
            "checkpoint), the 24 tools and the full text of every tool result, and replies with one call per turn. "
            "No strategy hints, no scaffolding. Seed 7, same world for everyone. Mean composite, with the range "
            "over repeated runs and the number of runs in brackets.")
        add("")
        add("| Player | " + " | ".join(f"{t} (runs)" for t in tasks_run) + " |")
        add("|:---|" + "---:|" * len(tasks_run))
        add("| **Reference (scripted)** | " + " | ".join(f"**{_f(base[t]['reference'][0])}**" for t in tasks_run) + " |")
        add("| Pass threshold | " + " | ".join(_f(base[t].get("threshold")) for t in tasks_run) + " |")
        for label, by_task in runs.items():
            cells = []
            for t in tasks_run:
                recs = by_task.get(t, [])
                cells.append(f"{_rng([r['regraded'] for r in recs])} ({len(recs)})" if recs else "not run")
            add(f"| {MODEL_NAMES.get(label, label)} | " + " | ".join(cells) + " |")
        add("| Do nothing | " + " | ".join(_f(base[t]["do_nothing"][0]) for t in tasks_run) + " |")
        add("")
        for label, by_task in runs.items():
            for recs in by_task.values():
                for r in recs:
                    if not r["replay_matches"] or abs(r["regraded"] - r["summary"]["composite"]) > 1e-6:
                        problems.append(f"{r['path']}: the saved ledger or score does not match a replay of its calls "
                                        f"(saved {r['summary']['composite']}, replayed {r['regraded']})")

        agent_rows = [r for r in hb if r["agent"] not in ("oracle", "nop")]
        if agent_rows:
            add("### The standard harness: Harbor + OpenCode over MCP")
            add("")
            add("The same tasks played by the OpenCode agent inside Harbor's container, talking to the game's MCP "
                "server. This is the path judges re-run. Each row is one trial; its reward, sealed ledger and the "
                "agent's full trajectory are in `results/harbor_agent/`.")
            add("")
            add("| Harbor task | Model | Reward | Deaths | Tool calls |")
            add("|:---|:---|---:|---:|---:|")
            by: dict = defaultdict(list)
            for r in agent_rows:
                name = MODEL_NAMES.get(r["model"], r["model"])
                if r["reward"] is None:
                    add(f"| {r['task']} | {name} | did not start (agent install timed out on our machine) | - | - |")
                    continue
                by[(r["task"], name)].append(r["reward"])
                add(f"| {r['task']} | {name} | {_f(r['reward'])}{' (agent error)' if r['error'] else ''} | "
                    f"{r['deaths'] if r['deaths'] is not None else '-'} | {r['calls'] if r['calls'] is not None else '-'} |")
            add("")
            add("Means: " + "; ".join(f"{m} on {t.removeprefix('terraria-')}: {_f(_mean([x for x in v if x is not None]))} "
                                      f"over {len(v)}" for (t, m), v in by.items()) + ".")
            add("")
        add("### What each run did with its budget")
        add("")
        add("Read from the transcripts. *Unused* is the share of the operation budget never spent. *Looking* is the "
            "share of paid calls that only inspect. *Failed* is the share of paid calls the game rejected. "
            "*Missing* lists what the reference achieved on this world that the run did not, most valuable first.")
        add("")
        add("| Player | Task | Run | Score | Deaths (first) | Unused | Looking | Failed | Checkpoints missed | Missing vs reference |")
        add("|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|")
        for label, by_task in runs.items():
            for t, recs in by_task.items():
                for r in recs:
                    d = diagnose(r, base[t]["ref_ledger"])
                    r["diag"] = d
                    died = next((f"window {w['w']}" for w in r["ledger"]["windows"] if w["deaths"]), "-")
                    add(f"| {MODEL_NAMES.get(label, label)} | {t} | {r['summary']['run']} | {_f(r['regraded'])} | "
                        f"{d['deaths']} ({died}) | {d['ops_unused_pct']}% | {d['look_pct']}% | {d['failed_pct']}% | "
                        f"{len(d['checkpoints_missed'])}/{d['checkpoints_total']} | "
                        f"{', '.join(d['missing_milestones'][:5]) or 'nothing'} |")
        add("")
        add("### How each run first died")
        add("")
        add("Every death in the saved runs, with the cause the game reported and the reason the model gave for the "
            "call that killed it (quoted from the transcript). In a Hardcore task the first death ends the run.")
        add("")
        add("| Player | Task | Run | Died in window | Cause | The model's stated reason for its last call |")
        add("|:---|:---|---:|:---|:---|:---|")
        causes: Counter = Counter()
        n_runs = n_dead = n_claimed = 0
        for label, by_task in runs.items():
            for t, recs in by_task.items():
                for r in recs:
                    n_runs += 1
                    fd = first_death(r)
                    if not fd:
                        continue
                    n_dead += 1
                    n_claimed += fd["claimed_safe"]
                    causes[fd["cause"]] += 1
                    said = fd["said"].replace("|", "/").replace("\n", " ")[:230]
                    add(f"| {MODEL_NAMES.get(label, label)} | {t} | {r['summary']['run']} | {fd['window']} of {fd['of']} | "
                        f"{fd['cause']} | \"{said}\" |")
        add("")
        add(f"{n_dead} of {n_runs} runs had at least one death. Causes of the first death: "
            + "; ".join(f"{c} ({n})" for c, n in causes.most_common()) + ". "
            f"In {n_claimed} of those {n_dead} the model's own reason for its last call says it is safe.")
        add("")
        add("The written analysis of these runs is in `FAILURE_ANALYSIS.md`.")
    add("")

    # ------------------------------------------------------------------ 3. exploits
    add("## 3. Reward exploits, attempted")
    add("")
    add("Each row is an attack that is actually executed against the engine by `terraria_bench/exploits.py`, "
        "then graded. *Closed* means every check held.")
    add("")
    add("| Exploit | What the attacker does | Why it earns nothing | Result |")
    add("|:---|:---|:---|:---|")
    desc = {n: (what, why) for n, what, why in EXPLOITS}
    for name in EXPLOIT_NAMES:
        res = run_exploit(name)
        if not res["closed"]:
            problems.append(f"exploit {name} is open: {res['checks']}")
        add(f"| {name} | {desc[name][0]} | {desc[name][1]} | {'closed' if res['closed'] else '**OPEN**'} |")
    add("")

    # ------------------------------------------------------------------ 4. small model
    if BC_EVAL.exists():
        bc = json.loads(BC_EVAL.read_text(encoding="utf-8"))
        by_cell: dict = defaultdict(dict)
        for r in bc:
            by_cell[r["cell"]][r["model"]] = r
        rows = [{"cell": c, **m} for c, m in by_cell.items() if "base" in m and "tuned" in m]
        add("## 4. A smaller model trained on experience from the environment")
        add("")
        add("`google/gemma-4-E4B-it` (a 4B-class model), before and after a rank-32 LoRA trained by behaviour cloning "
            "on 652 turn-by-turn steps of the reference policy on the four *training* tasks. Same prompt, greedy "
            "decoding, one episode per cell, official composite. Source: `artifacts/post_training/bc_eval.json`; "
            "reproduce with `modal run post_training/modal_grpo_gemma.py::eval_bc`. This run predates the scoring "
            "text now in the prompt.")
        add("")
        add("| Cell | Seen in training? | Base | Tuned | Deaths (base / tuned) |")
        add("|:---|:---|---:|---:|:---:|")
        b_all, t_all = [], []
        for r in rows:
            cell = r.get("cell", "")
            task_id, _, seed = cell.rpartition(":")
            seen = "task and seed" if (TASKS.get(task_id) and TASKS[task_id].split == "train" and int(seed) in REPORTED_SEEDS) \
                else ("task only (new world)" if TASKS.get(task_id) and TASKS[task_id].split == "train" else "no (evaluation task)")
            b, t = r.get("base", {}), r.get("tuned", {})
            b_all.append(b.get("composite")), t_all.append(t.get("composite"))
            add(f"| {cell} | {seen} | {_f(b.get('composite'))} | {_f(t.get('composite'))} | {b.get('deaths')} / {t.get('deaths')} |")
        if b_all and None not in b_all + t_all:
            add(f"| **Mean** | | **{_f(_mean(b_all))}** | **{_f(_mean(t_all))}** | |")
        add("")
    return "\n".join(L) + "\n", problems


def main() -> int:
    cal = load_calibration()
    text, problems = build(cal, reseal="--reseal" in sys.argv[1:])
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    if problems:
        print("INTEGRITY PROBLEMS:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1
    print(f"All integrity checks held. Wrote {OUT.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
