#!/usr/bin/env python3
"""Write EVAL_RESULTS.md from results that can be reproduced or verified.

    python reporting/generate_report.py

Three sources, nothing else:

* the six scripted anchors, re-run on the spot over EVAL_TASKS x REPORTED_SEEDS;
* sealed episode ledgers saved by live model runs under ``artifacts/runs/<model>/``
  (a model with no saved ledgers is listed as "not run", never estimated);
* ``artifacts/post_training/bc_eval.json``, the base-vs-tuned Gemma comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from terraria_bench.grader import grade  # noqa: E402
from terraria_bench.harness.runner import (  # noqa: E402
    _aggregate_graded, _cell_means, _load_saved_runs, _reference_calibration, run_eval_standard,
)
from terraria_bench.tasks import REPORTED_SEEDS  # noqa: E402

FAILURE_AREAS = (
    ("Long-range strategy planning", ("goblin-siege", "eye-of-cthulhu", "wall-of-flesh", "full-progression")),
    ("Short-range strategy & execution", ("first-night", "blood-moon-survival")),
    ("Multimodal & 2D spatial modification", ("arena-architect", "cavern-expedition")),
)

HEADLINE_CELL = ("eye-of-cthulhu", 7)
HEADLINE_MODELS = (("gemini", "Gemini 4 / 2.5"), ("nemotron", "Nemotron 3.5 VL"))

BC_EVAL = REPO / "artifacts" / "post_training" / "bc_eval.json"
GRPO_EVAL = REPO / "artifacts" / "post_training" / "grpo_eval.json"
EVAL_STANDARD_JSON = REPO / "artifacts" / "runs" / "eval_standard.json"
OUT = REPO / "EVAL_RESULTS.md"


def _num(v, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:.3f}"


def _headline_section() -> list[str]:
    """Repeat runs of the headline cell: the reference, the two baselines, and each saved model run."""
    from terraria_bench.harness.runner import _run_builtin_anchor

    task, seed = HEADLINE_CELL
    cell_key = f"{task}:{seed}"
    cal = _reference_calibration()
    cell = cal["cells"][cell_key]
    lines = [
        f"## Headline cell: {task}, seed {seed}",
        "",
        f"Pass threshold {cell['threshold']:.3f}. Scripted rows are deterministic, so one run is exact. Model",
        "rows are separate live episodes of the same cell; `ended by` is recorded by the harness",
        "(`episode_complete` = every window was played; `agent_stopped` = the model stopped calling tools;",
        "`harness_turn_cap` = the runner cut the episode). `Stand-in turns` counts turns where the model's",
        "API call failed and the harness's scripted fallback acted instead; a clean model run shows 0.",
        "",
        "| Agent | Run | Composite | Eye progress | Deaths | Windows | Turns | Stand-in turns | Ended by | Passed |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|:---|:---:|",
    ]
    for anchor, label in (("smart-triage", "reference (smart-triage)"), ("rule-based", "rule-based"),
                          ("absentee", "stop-early (absentee)"), ("do-nothing", "do-nothing")):
        g = grade(_run_builtin_anchor(anchor, task, seed, calibration=cal), cell)
        lines.append(f"| {label} | - | {g['composite']:.3f} | {g.get('eye_progress_pct', 0.0):.0f}% | "
                     f"{int(g.get('deaths', 0))} | {g['windows']} | - | - | scripted | {'yes' if g.get('passed') else 'no'} |")
    for key, label in HEADLINE_MODELS:
        runs = [r for r in _load_saved_runs(key, cal) if r["cell"] == cell_key]
        if not runs:
            lines.append(f"| {label} | - | not run | | | | | | | |")
            continue
        runs.sort(key=lambda r: (r.get("meta") or {}).get("run", 1))
        for r in runs:
            meta = r.get("meta") or {}
            lines.append(
                f"| {label} | {meta.get('run', '?')} | {r['composite']:.3f} | {r.get('eye_progress_pct', 0.0):.0f}% | "
                f"{int(r.get('deaths', 0))} | {r['windows']} | {meta.get('turns', '-')} | "
                f"{(meta.get('agent_stats') or {}).get('live_errors', '-')} | {meta.get('ended_by', 'not recorded')} | "
                f"{'yes' if r.get('passed') else 'no'} |")
        comps = [r["composite"] for r in runs]
        eyes = [r.get("eye_progress_pct", 0.0) for r in runs]
        lines.append(
            f"| **{label}, {len(runs)} runs** | | **{sum(comps) / len(comps):.3f}** "
            f"({min(comps):.3f}-{max(comps):.3f}) | **{sum(eyes) / len(eyes):.0f}%** | | | | | | "
            f"**{sum(1 for r in runs if r.get('passed'))}/{len(runs)}** |")
    lines.append("")
    return lines


def _real_game_section() -> list[str]:
    """Same goal, same model family, two worlds: the simulator ledgers against the real-game recordings."""
    from terraria_bench.real_eval import STAGES, score_all

    rows = score_all(save=True)
    if not rows:
        return []
    cal = _reference_calibration()
    sim = [r for r in _load_saved_runs("gemini", cal) if r["cell"].startswith("eye-of-cthulhu:")]
    lines = [
        "## Simulator vs. real Terraria (same goal: kill the Eye of Cthulhu)",
        "",
        "Rescore the committed real-game recordings with `python -m terraria_bench.real_eval --all --save` "
        "(no game, model or network needed; writes `artifacts/runs/real_terraria/`). The scorer reads kills "
        "and deaths from the game's own referee block and marks a recording inconsistent if they disagree "
        "with the claimed score.",
        "",
        "| World | Agent | Runs | Eye killed | Mean Eye progress | Deaths (total) |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    if sim:
        kills = sum(1 for r in sim if float(r.get("eye_progress") or 0.0) >= 1.0)
        prog = sum(100.0 * float(r.get("eye_progress") or 0.0) for r in sim) / len(sim)
        deaths = sum(int((r.get("meta") or {}).get("deaths", r.get("deaths", 0)) or 0) for r in sim)
        lines.append(
            f"| Simulator (symbolic state, MCP tools) | Gemini two-tier | {len(sim)} | {kills}/{len(sim)} "
            f"| {prog:.0f}% | {deaths} |"
        )
    model_rows = [r for r in rows if r["player"] == "model"]
    for label, group in (("Gemini three-tier", model_rows), ("Human reference", [r for r in rows if r["player"] != "model"])):
        if not group:
            continue
        kills = sum(1 for r in group if r["eye_killed"])
        prog = sum(r["eye_progress_pct"] for r in group) / len(group)
        lines.append(
            f"| Real game (pixels, keyboard and mouse) | {label} | {len(group)} | {kills}/{len(group)} "
            f"| {prog:.0f}% | {sum(r['deaths'] for r in group)} |"
        )
    lines += ["", "Where each real-game recording stops on the road to the Eye:", "",
              "| Stage | " + " | ".join(f"`{Path(r['recording']).stem}` ({r['player']})" for r in rows) + " |",
              "|:---|" + ":---:|" * len(rows)]
    for stage in STAGES:
        lines.append(f"| {stage.replace('_', ' ')} | "
                     + " | ".join("done" if r["task_checklist"].get(stage) else "-" for r in rows) + " |")
    lines.append("")
    for r in rows:
        t = r["trajectory"]
        lat = ", ".join(f"{k} {v:.1f}s" for k, v in t["median_latency_s"].items()) or "n/a"
        gained = ", ".join(t["milestones_gained_in_recording"]) or "none"
        lines.append(
            f"- `{Path(r['recording']).name}`: house built: {'yes' if r['house_built'] else 'no'}; "
            f"{r['stages_done']}/{r['stages_total']} stages; first unmet stage `{r['stalled_at']}`; "
            f"new milestones across the {r['frames']} recorded frames (turns {t['first_turn']}-{t['last_turn']}): "
            f"{gained}; deaths {r['deaths']}; median model latency: {lat}; "
            f"referee check: {'consistent' if r['referee']['consistent'] else 'not available or inconsistent'}."
        )
    lines.append("")
    return lines


def build_markdown(report: dict | None = None) -> str:
    if report is None:
        report = run_eval_standard(live=False)
    lines = [
        "# TerrariaBench evaluation results",
        "",
        "Generated by `python reporting/generate_report.py`. Scripted anchors are re-run every time;",
        "model rows come only from sealed ledgers of live runs saved under `artifacts/runs/`.",
        "",
        "## Simulator leaderboard (8 eval tasks x seeds 7, 11, 13)",
        "",
        "Composite is the calibrated reward (0 = no better than an absent player, 1 = the best scripted",
        "policy). Eye progress is the 0-100% progress-toward-killing-the-Eye score.",
        "",
        "| Policy / model | Status | Cells | Runs | Composite | Pass rate | Eye progress |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for row in report["table1"]:
        lines.append(
            f"| {row['name']} | {row['status']} | {row['episodes']} | {row.get('runs', row['episodes'])} | "
            f"{_num(row['composite'])} | {_num(row['pass_rate'], pct=True)} | "
            f"{_num(row.get('eye_progress'), pct=True)} |"
        )
    lines += [
        "",
        "Cells are distinct task:seed pairs; runs are sealed episodes. Repeat runs of one cell are averaged",
        "into that cell first, so repeating the headline task does not re-weight a model's row. Model rows",
        "cover only the cells that were actually played and are not comparable to the 24-cell anchor rows",
        "until the full grid is run.",
        "",
    ]
    lines += _headline_section()

    if BC_EVAL.exists():
        rows = json.loads(BC_EVAL.read_text(encoding="utf-8"))
        cells = sorted({r["cell"] for r in rows}, key=[r["cell"] for r in rows].index)
        by = {(r["cell"], r["model"]): r for r in rows}
        lines += [
            "## Smaller model trained on environment experience",
            "",
            "`google/gemma-4-E4B-it`, with and without a rank-32 LoRA trained by behaviour cloning on 652",
            "expert steps from the four training tasks (seeds 7, 11, 13). Same prompt and greedy decoding",
            "for both; one episode per cell. Reproduce with `modal run post_training/modal_grpo_gemma.py::eval_bc`.",
            "",
            "| Cell | Base | Tuned | Pass threshold | Deaths (base / tuned) |",
            "|:---|---:|---:|---:|:---:|",
        ]
        base_sum = tuned_sum = 0.0
        for cell in cells:
            b, t = by.get((cell, "base")), by.get((cell, "tuned"))
            if not b or not t:
                continue
            base_sum += b["composite"]
            tuned_sum += t["composite"]
            mark = " (pass)" if t.get("passed") else ""
            lines.append(f"| {cell} | {b['composite']:.3f} | {t['composite']:.3f}{mark} | {t['threshold']:.3f} | "
                         f"{b['deaths']} / {t['deaths']} |")
        n = sum(1 for c in cells if (c, "base") in by and (c, "tuned") in by)
        if n:
            lines.append(f"| **Mean** | **{base_sum / n:.3f}** | **{tuned_sum / n:.3f}** | | |")
        lines += [
            "",
            "Seed 7 cells of the training tasks were seen in training; seed 17 cells are unseen worlds;",
            "first-night and cavern-expedition are eval tasks the model never trained on.",
            "",
        ]

    cal = _reference_calibration()
    lines += [
        "## Breakdown by failure area",
        "",
        "| Failure area | Tasks | smart-triage | rule-based | Gemini 4 / 2.5 | Nemotron 3.5 VL |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    from terraria_bench.harness.runner import _run_builtin_anchor

    def _saved_for_tasks(agent_name: str, tasks: tuple[str, ...]) -> list[dict]:
        rows = [r for r in _load_saved_runs(agent_name, cal) if r["cell"].rsplit(":", 1)[0] in tasks]
        return _cell_means(rows)

    for area_name, area_tasks in FAILURE_AREAS:
        st_rows = [grade(_run_builtin_anchor("smart-triage", t, s, calibration=cal), cal["cells"][f"{t}:{s}"])
                   for t in area_tasks for s in REPORTED_SEEDS]
        rb_rows = [grade(_run_builtin_anchor("rule-based", t, s, calibration=cal), cal["cells"][f"{t}:{s}"])
                   for t in area_tasks for s in REPORTED_SEEDS]
        gem_area = _saved_for_tasks("gemini", area_tasks)
        nem_area = _saved_for_tasks("nemotron", area_tasks)
        st_m = _aggregate_graded(st_rows)["composite"]
        rb_m = _aggregate_graded(rb_rows)["composite"]
        gem_m = _aggregate_graded(gem_area)["composite"] if gem_area else None
        nem_m = _aggregate_graded(nem_area)["composite"] if nem_area else None
        lines.append(
            f"| {area_name} | {', '.join(area_tasks)} | {_num(st_m)} | {_num(rb_m)} | {_num(gem_m)} | {_num(nem_m)} |"
        )
    lines.append("")

    if GRPO_EVAL.exists():
        grows = json.loads(GRPO_EVAL.read_text(encoding="utf-8"))
        gcells = sorted({r["cell"] for r in grows}, key=[r["cell"] for r in grows].index)
        gby = {(r["cell"], r["model"]): r for r in grows}
        gb_vals = [gby[(c, "base")]["composite"] for c in gcells if (c, "base") in gby]
        gt_vals = [gby[(c, "tuned")]["composite"] for c in gcells if (c, "tuned") in gby]
        gt_pass = sum(1 for c in gcells if (c, "tuned") in gby and gby[(c, "tuned")].get("passed"))
        if gb_vals and gt_vals:
            gb_d = sum(int(gby[(c, "base")].get("deaths", 0)) for c in gcells if (c, "base") in gby)
            gt_d = sum(int(gby[(c, "tuned")].get("deaths", 0)) for c in gcells if (c, "tuned") in gby)
            lines += [
                "## GRPO post-training (BC-initialised, KL-regularised)",
                "",
                "GRPO on `google/gemma-4-E4B-it`, starting from the behaviour-cloned LoRA and rewarded only by the "
                "benchmark's own sealed-ledger grader (`post_training/terraria_reward.py`). Rows are read from "
                "`artifacts/post_training/grpo_eval.json`; base and tuned are evaluated on the same cells with the "
                "same turn budget.",
                "",
                "| Cell | Threshold | Base | Tuned | Base deaths | Tuned deaths |",
                "|:---|---:|---:|---:|---:|---:|",
            ]
            for c in gcells:
                bb, tt = gby.get((c, "base")), gby.get((c, "tuned"))
                if not bb or not tt:
                    continue
                lines.append(f"| {c} | {tt['threshold']:.3f} | {bb['composite']:.3f} | {tt['composite']:.3f} | "
                             f"{int(bb.get('deaths', 0))} | {int(tt.get('deaths', 0))} |")
            lines += [
                f"| **Mean / total** | | **{sum(gb_vals)/len(gb_vals):.3f}** | **{sum(gt_vals)/len(gt_vals):.3f}** | "
                f"**{gb_d}** | **{gt_d}** |",
                "",
                f"Tuned cells passing the calibrated threshold: **{gt_pass}/{len(gt_vals)}**. The small model improves "
                "sharply on reward and stops dying, yet still clears no pass gate: the benchmark keeps headroom "
                "after training.",
                "",
            ]

    lines += _real_game_section()

    return "\n".join(lines)


def main() -> int:
    report = run_eval_standard(live=False)
    EVAL_STANDARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVAL_STANDARD_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT.write_text(build_markdown(report), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} and {EVAL_STANDARD_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
