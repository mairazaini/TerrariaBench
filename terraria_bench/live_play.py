"""Gemini Decision 1 Terraria controls engine, playthrough & failure-clip studio, and live tModLoader bridge."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .engine import H, W
from .harness.gemini_agent import GeminiAgent, GeminiDecisionsWrapper
from .human_recorder import BridgeSession
from .provenance import ARTIFACTS_DIR
from .real_game import RealGameController, sanitize_hud
from .tasks import TASKS
from .tools import Episode
from .web_console import _readonly_viewport

RECORDINGS_DIR = ARTIFACTS_DIR / "recordings"
REAL_FRAMES_DIRNAME = "real_terraria_live_frames"
REAL_FRAMES_KEPT = 300
DEFAULT_SUPERVISOR_MODEL = "gemini-4-hydrogen"
DEFAULT_DECISION_MODEL = "gemini-decision-1"

HOTBAR_SLOTS: tuple[tuple[str, str], ...] = (
    ("slot_1_weapon", "Slot 1: Weapon for combat"),
    ("slot_2_pickaxe", "Slot 2: Pickaxe for mining"),
    ("slot_3_axe", "Slot 3: Axe for chopping trees"),
    ("slot_4_wood", "Slot 4: Wood blocks for building"),
    ("slot_5_platform", "Slot 5: Platforms for arena/stairs"),
    ("slot_6_torch", "Slot 6: Torches for light"),
    ("slot_7_station", "Slot 7: Workbench/Furnace/Anvil/Door"),
    ("slot_8_potion", "Slot 8: Healing potion or boss summon"),
)
HOTBAR_INDEX = {key: idx for idx, (key, _) in enumerate(HOTBAR_SLOTS)}

CONTROL_QUESTIONS: dict[str, dict[str, Any]] = {
    "move_x": {
        "type": "choice",
        "instructions": "Which horizontal movement key should the player hold right now?",
        "criteria": {
            "left": "Hold A to move left",
            "idle": "Stand still horizontally",
            "right": "Hold D to move right",
        },
    },
    "jump": {
        "type": "noul",
        "instructions": "Should the player press Space (Jump) right now to clear terrain, climb platforms, or dodge an enemy?",
    },
    "crouch_down": {
        "type": "noul",
        "instructions": "Should the player press S (Down) right now to drop through a platform or dig downward?",
    },
    "use_item": {
        "type": "noul",
        "instructions": "Should the player hold Left Click (Use Item) right now to swing weapon, mine, chop, or place a block?",
    },
    "interact": {
        "type": "noul",
        "instructions": "Should the player press Right Click (Interact) right now to open a door, chest, or station?",
    },
    "grapple": {
        "type": "noul",
        "instructions": "Should the player press E (Grapple) right now to hook onto a platform or wall?",
    },
    "quick_heal": {
        "type": "noul",
        "instructions": "Is HP low enough that the player should press H (Quick Heal) right now?",
    },
    "smart_cursor": {
        "type": "noul",
        "instructions": "Should Smart Cursor (Ctrl) be enabled for automatic tile targeting?",
    },
    "hotbar_slot": {
        "type": "choice",
        "instructions": "Which hotbar slot (1-8) should be selected right now?",
        "criteria": dict(HOTBAR_SLOTS),
    },
    "aim_target": {
        "type": "choice",
        "instructions": "Where should the player aim the Terraria mouse cursor right now?",
        "criteria": {
            "nearest_enemy": "Aim cursor at nearest enemy or boss",
            "nearest_tree": "Aim cursor at nearest tree trunk",
            "nearest_ore": "Aim cursor at nearest ore or stone vein",
            "underfoot": "Aim cursor below feet to dig down",
            "ahead": "Aim cursor horizontally in movement direction",
            "build_site": "Aim cursor at shelter or arena build site",
        },
    },
    "threat_level": {
        "type": "score",
        "instructions": "How dangerous is the current situation for the player?",
        "criteria": [
            "Calm: daytime, safe, no hostiles nearby",
            "Caution: dusk approaching or minor slime nearby",
            "Danger: night unsheltered, multiple enemies, or HP below 60%",
            "Critical: boss enraged, trapped, or near death",
        ],
    },
}

def _as_int(val: Any, default: int = 0) -> int:
    """Coerce int, float, numeric str, or tModLoader DefenseStat dict into an int."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, dict):
        for key in ("flat", "Flat", "base", "Base", "value", "Value", "total", "Total"):
            if key in val and isinstance(val[key], (int, float)):
                return int(val[key])
        return default
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            return default
    return default


def load_dotenv_if_needed(env_path: Path | None = None) -> None:
    """Populate missing env vars from local ``.env`` without printing secrets."""
    path = env_path or (Path(__file__).resolve().parents[1] / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _format_active_keys(ctrl: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    if ctrl["left"]:
        keys.append("A")
    if ctrl["right"]:
        keys.append("D")
    if ctrl["jump"]:
        keys.append("SPACE")
    if ctrl["down"]:
        keys.append("S")
    if ctrl["use_item"]:
        keys.append("LMB")
    if ctrl["use_tile"]:
        keys.append("RMB")
    if ctrl["hook"]:
        keys.append("E:GRPL")
    if ctrl["quick_heal"]:
        keys.append("H:HEAL")
    if ctrl["smart_cursor"]:
        keys.append("CTRL:SMART")
    slot = ctrl.get("hotbar_slot", "slot_1_weapon")
    idx = HOTBAR_INDEX.get(slot, 0) + 1
    short = slot.split("_")[-1].upper()[:4]
    keys.append(f"{idx}:{short}")
    return keys


def _answers_to_controls(answers: dict[str, Any]) -> dict[str, Any]:
    """Translate ``gemini-decision-1`` answer payloads into Terraria input flags."""
    def val(key: str, default: Any = None) -> Any:
        return answers.get(key, {}).get("value", default)

    move_x = str(val("move_x", "idle"))
    jump_p = float(val("jump", 0.0) or 0.0)
    down_p = float(val("crouch_down", 0.0) or 0.0)
    use_p = float(val("use_item", 0.0) or 0.0)
    interact_p = float(val("interact", 0.0) or 0.0)
    grapple_p = float(val("grapple", 0.0) or 0.0)
    heal_p = float(val("quick_heal", 0.0) or 0.0)
    smart_p = float(val("smart_cursor", 0.0) or 0.0)
    hotbar = str(val("hotbar_slot", "slot_1_weapon"))
    aim = str(val("aim_target", "ahead"))
    threat_score = float(val("threat_level", 0.0) or 0.0)
    threat_idx = min(3, max(0, round(threat_score)))
    threat_labels = ("Calm", "Caution", "Danger", "Critical")

    ctrl = {
        "move_x": move_x,
        "left": move_x == "left",
        "right": move_x == "right",
        "up": jump_p >= 0.65,
        "down": down_p >= 0.55,
        "jump": jump_p >= 0.55,
        "use_item": use_p >= 0.50,
        "use_tile": interact_p >= 0.55,
        "hook": grapple_p >= 0.60,
        "quick_heal": heal_p >= 0.60,
        "smart_cursor": smart_p >= 0.55,
        "hotbar_slot": hotbar if hotbar in HOTBAR_INDEX else "slot_1_weapon",
        "hotbar_index": HOTBAR_INDEX.get(hotbar, 0),
        "aim_target": aim,
        "threat_score": round(threat_score, 3),
        "threat_label": threat_labels[threat_idx],
        "decisions": {
            k: {
                "type": v.get("type"),
                "value": v.get("value"),
                "confidence": round(float(v.get("confidence", 0.0) or 0.0), 4),
                "probabilities": v.get("probabilities") or {},
            }
            for k, v in answers.items()
        },
    }
    ctrl["active_keys"] = _format_active_keys(ctrl)
    return ctrl


def _derive_offline_controls(tool: str, args: dict[str, Any], obs: dict[str, Any], ep: Episode, base_answers: dict[str, Any]) -> dict[str, Any]:
    """Tool-aligned deterministic control synthesis for offline/non-live playback."""
    ctrl = _answers_to_controls(base_answers)
    w = ep.world
    left = right = jump = down = use_item = use_tile = hook = quick_heal = smart = False
    hotbar, aim = "slot_1_weapon", "ahead"

    if tool == "chop_trees":
        right, use_item, smart, hotbar, aim = True, True, True, "slot_3_axe", "nearest_tree"
    elif tool in ("mine_vein", "explore_cavern"):
        down = tool == "explore_cavern"
        right = not down
        use_item, smart, hotbar = True, True, "slot_2_pickaxe"
        aim = "underfoot" if down else "nearest_ore"
    elif tool in ("build_blueprint", "place_station_or_furniture"):
        item_id = str(args.get("item_id", ""))
        use_item, smart, aim = True, True, "build_site"
        if "platform" in item_id or args.get("preset") in ("arena", "hellbridge"):
            hotbar = "slot_5_platform"
        elif item_id == "torch":
            hotbar = "slot_6_torch"
        elif item_id in ("workbench", "furnace", "anvil", "door", "campfire"):
            hotbar = "slot_7_station"
        else:
            hotbar = "slot_4_wood"
    elif tool == "navigate_to":
        tx = int(args.get("x", w.pos[0]))
        ty = int(args.get("y", w.pos[1]))
        left, right = tx < w.pos[0], tx > w.pos[0]
        jump, down = ty < w.pos[1], ty > w.pos[1]
        aim = "ahead"
    elif tool == "engage_combat_burst":
        left = (w.tick % 2) == 0
        right = not left
        jump, use_item, hotbar, aim = True, True, "slot_1_weapon", "nearest_enemy"
    elif tool == "deploy_combat_arena_tactics":
        jump, hook, hotbar, aim = True, True, "slot_5_platform", "build_site"
    elif tool in ("retreat_and_stabilize", "consume_buff_or_heal"):
        left = w.pos[0] > w.home[0]
        right = w.pos[0] < w.home[0]
        quick_heal = w.hp < w.max_hp * 0.65
        hotbar, aim = "slot_8_potion", "ahead"
    elif tool == "loot_container_or_crystal":
        use_tile, hotbar, aim = True, "slot_2_pickaxe", "nearest_ore"

    if w.hp < w.max_hp * 0.4 and w.has("lesser_healing_potion"):
        quick_heal = True

    ctrl.update(
        move_x="left" if left else ("right" if right else "idle"),
        left=left,
        right=right,
        up=jump,
        down=down,
        jump=jump,
        use_item=use_item,
        use_tile=use_tile,
        hook=hook,
        quick_heal=quick_heal,
        smart_cursor=smart,
        hotbar_slot=hotbar,
        hotbar_index=HOTBAR_INDEX[hotbar],
        aim_target=aim,
    )
    ctrl["active_keys"] = _format_active_keys(ctrl)
    return ctrl


class DecisionController:
    """Drives both Episode turns and real tModLoader ticks using ``gemini-4-hydrogen`` + ``gemini-decision-1``."""

    def __init__(
        self,
        supervisor_model: str = DEFAULT_SUPERVISOR_MODEL,
        decision_model: str = DEFAULT_DECISION_MODEL,
        live: bool = False,
        vision_model: str | None = None,
        min_confidence: float = 0.35,
        max_replans_per_window: int = 3,
    ):
        self.live = live
        self.agent = GeminiAgent(
            model=supervisor_model,
            live=live,
            decisions=GeminiDecisionsWrapper(model=decision_model, strict=live),
            min_confidence=min_confidence,
            max_replans_per_window=max_replans_per_window,
        )
        self.real = RealGameController(self.agent, live=live, vision_model=vision_model)

    def decide_turn(self, obs: dict[str, Any], context: list[dict[str, Any]], ep: Episode) -> dict[str, Any]:
        vp = _readonly_viewport(ep)
        vp_png = obs.get("viewport_png_b64") or vp["viewport_png_b64"]

        if not self.live:
            action = self.agent._scripted_action(obs, ep)
            tool = (action or {}).get("tool", "end_window")
            args = dict((action or {}).get("args") or {})
            w = ep.world
            state_desc = (
                f"window={w.window} phase={w.phase} sheltered={w.housing()['valid']} "
                f"hp={w.hp}/{w.max_hp} tier={w.tech_tier} boss={bool(w.boss)} tool={tool}"
            )
            dec = self.agent.decisions.decide(CONTROL_QUESTIONS, state_desc, viewport_png_b64=vp_png)
            self.agent.stats["decision_calls"] += 1
            controls = _derive_offline_controls(tool, args, obs, ep, dec["answers"])
            return {
                "tool": tool,
                "args": args,
                "controls": controls,
                "supervisor_strategy": "Offline scripted stand-in + deterministic control telemetry",
                "decision_choice": tool,
                "decision_conf": 1.0,
                "replan_triggered": False,
            }

        window = ep.world.window
        if self.agent._plan is None or self.agent._plan_window != window:
            self.agent._replans = 0
            self.agent._last_choice = None
            self.agent._supervise(obs, context, ep, f"Window {window} is starting. Plan this window.")

        replan_triggered = False
        choice = "end_window"
        conf = 0.0
        controls: dict[str, Any] = {}

        while True:
            assert self.agent._plan is not None
            cands = self.agent._plan["candidates"]
            criteria = {
                cid: f"{c['tool']} {json.dumps(c['args'], sort_keys=True)} - {c['when']}"
                for cid, c in cands.items()
            }
            criteria["end_window"] = "End the current window now: nothing on the menu is still useful this window."
            criteria["replan"] = "Ask the supervisor for a new menu: no candidate fits the current state, or the last action failed."

            questions: dict[str, dict[str, Any]] = {
                "next_action": {
                    "type": "choice",
                    "instructions": (
                        "Given the supervisor strategy, game state, and viewport, pick the single best candidate "
                        "action to execute right now."
                    ),
                    "criteria": criteria,
                },
                **CONTROL_QUESTIONS,
            }
            state_text = (
                f"SUPERVISOR STRATEGY\n{self.agent._plan['strategy']}\n\n"
                f"PREVIOUS CHOICE\n{self.agent._last_choice or 'none'}\n\n"
                f"CURRENT STATE AND LAST ACTION RESULT\n{self.agent._obs_text(obs)}"
            )
            self.agent.stats["decision_calls"] += 1
            res = self.agent.decisions.decide(questions, state_text, viewport_png_b64=vp_png)
            answers = res["answers"]
            controls = _answers_to_controls(answers)
            next_ans = answers["next_action"]
            choice = str(next_ans["value"])
            conf = float(next_ans.get("confidence") or 0.0)
            probs = next_ans.get("probabilities") or {}

            playable = choice in cands or choice == "end_window"
            low = conf < self.agent.min_confidence
            if playable and not low:
                break
            replan_triggered = True
            if self.agent._replans >= self.agent.max_replans_per_window:
                if not playable:
                    real = {k: v for k, v in probs.items() if k in cands}
                    choice = max(real, key=lambda k: real[k]) if real else "end_window"
                break
            self.agent._replans += 1
            self.agent.stats["replans"] += 1
            self.agent.stats["low_confidence"] += int(low)
            self.agent._last_choice = None
            self.agent._supervise(
                obs,
                context,
                ep,
                f"The decision model answered '{choice}' (conf={conf:.2f}). Previous strategy: "
                f"{self.agent._plan['strategy']} Write a new menu that fits the current state.",
            )

        self.agent._last_choice = choice
        cand = self.agent._plan["candidates"].get(choice)
        tool = cand["tool"] if cand else "end_window"
        args = dict(cand["args"]) if cand else {}
        self.agent.trace.append({
            "tier": "decision",
            "window": window,
            "choice": choice,
            "confidence": round(conf, 4),
            "tool": tool,
            "args": args,
            "active_keys": controls.get("active_keys", []),
        })
        return {
            "tool": tool,
            "args": args,
            "controls": controls,
            "supervisor_strategy": self.agent._plan["strategy"],
            "decision_choice": choice,
            "decision_conf": round(conf, 4),
            "replan_triggered": replan_triggered,
        }

    def decide_real_game_tick(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Real Terraria tick: vision -> supervisor -> decision. See ``real_game.RealGameController``."""
        return self.real.tick(payload)


def _snapshot_episode(ep: Episode) -> dict[str, Any]:
    w = ep.world
    threat, alerts = w.alerts()
    housing = w.housing()
    vp = _readonly_viewport(ep)
    return {
        "window": w.window,
        "total_windows": ep.task.windows,
        "phase": w.phase,
        "ops_left": ep.ops_left,
        "ops_budget": ep.window_budget,
        "pos": list(w.pos),
        "home": list(w.home),
        "hp": w.hp,
        "max_hp": w.max_hp,
        "mana": w.mana,
        "defense": w.defense,
        "deaths": w.deaths,
        "last_death": w.last_death,
        "phi": round(w.phi(), 2),
        "tech_tier": w.tech_tier,
        "sheltered": bool(housing["valid"]),
        "threat": threat,
        "alerts": [a["text"] for a in alerts],
        "boss": dict(w.boss) if w.boss else None,
        "equipped": dict(w.eq),
        "milestones": sorted(w.milestones),
        "inventory": dict(sorted(w.inv.items())),
        "ascii_grid": vp["ascii_grid"],
        "viewport_png_b64": vp["viewport_png_b64"],
        "header": ep.header(),
    }


def detect_step_failures(
    before: dict[str, Any],
    after: dict[str, Any],
    tool: str,
    res: dict[str, Any],
    decision_choice: str,
    decision_conf: float,
    replan_triggered: bool,
) -> list[dict[str, Any]]:
    """Identify demo-worthy failure events between ``before`` and ``after`` snapshots."""
    failures: list[dict[str, Any]] = []
    hp_delta = after["hp"] - before["hp"]
    if after["deaths"] > before["deaths"]:
        failures.append({
            "tag": "DEATH",
            "severity": "critical",
            "title": f"Player Died ({after['last_death'] or 'slain'})",
            "description": f"Died during {tool}; lost half of raw materials and 6 ops penalty.",
        })
    if hp_delta <= -15:
        failures.append({
            "tag": "HEAVY_DAMAGE",
            "severity": "high",
            "title": f"Heavy Damage ({hp_delta} HP)",
            "description": f"HP dropped from {before['hp']} to {after['hp']} during {tool}.",
        })
    if after["phase"] == "Night" and not after["sheltered"]:
        failures.append({
            "tag": "NIGHT_UNSHELTERED",
            "severity": "medium",
            "title": "Exposed at Night (Unsheltered)",
            "description": f"Operating outside valid housing at {after['pos']} during Night window {after['window']}.",
        })
    if res.get("ok") is False:
        failures.append({
            "tag": "ACTION_FAILED",
            "severity": "medium",
            "title": f"Tool Failed: {tool}",
            "description": str(res.get("error") or "Operation rejected"),
        })
    if replan_triggered or decision_choice == "replan" or decision_conf < 0.45:
        failures.append({
            "tag": "LOW_CONFIDENCE_REPLAN",
            "severity": "low",
            "title": f"Replan / Low Confidence ({decision_conf:.0%})",
            "description": f"Decision model chose '{decision_choice}' at {decision_conf:.2f} confidence.",
        })
    return failures


def build_failure_clips(frames: list[dict[str, Any]], context_pad: int = 2) -> list[dict[str, Any]]:
    """Index contiguous demo clip windows around detected failure frames."""
    clips: list[dict[str, Any]] = []
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for idx, frame in enumerate(frames):
        events = frame.get("failure_events") or []
        if not events:
            continue
        top = max(events, key=lambda e: severity_rank.get(e["severity"], 0))
        start_idx = max(0, idx - context_pad)
        end_idx = min(len(frames) - 1, idx + context_pad)
        if clips and clips[-1]["tag"] == top["tag"] and idx <= clips[-1]["end_frame"]:
            prev = clips[-1]
            prev["end_frame"] = max(prev["end_frame"], end_idx)
            prev["end_turn"] = frames[prev["end_frame"]]["turn"]
            continue
        clips.append({
            "clip_id": f"clip_{len(clips) + 1}_{top['tag'].lower()}",
            "tag": top["tag"],
            "severity": top["severity"],
            "title": top["title"],
            "description": top["description"],
            "focus_frame": idx,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "start_turn": frames[start_idx]["turn"],
            "end_turn": frames[end_idx]["turn"],
        })
    return clips


def render_player_html(recording: dict[str, Any]) -> str:
    """Render a self-contained HTML5 Canvas Playthrough & Failure-Clip Studio."""
    payload_json = json.dumps(recording, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(f"{recording['task_title']} ({recording['task_id']}, seed {recording['seed']})")
    models = html.escape(f"Supervisor: {recording['supervisor_model']}  |  Controls: {recording['decision_model']}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>TerrariaBench Live Play Studio — {title}</title>
<style>
  :root {{ --bg:#090d16; --panel:#111827; --border:#263248; --accent:#38bdf8; --danger:#ef4444; --warn:#f59e0b; --ok:#22c55e; --text:#e2e8f0; --muted:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:18px; background:var(--bg); color:var(--text); font-family:ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px; padding:12px 16px; background:var(--panel); border:1px solid var(--border); border-radius:10px; }}
  .title {{ font-size:18px; font-weight:700; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:2px; }}
  .layout {{ display:grid; grid-template-columns:minmax(680px, 1.35fr) minmax(340px, 0.9fr); gap:16px; align-items:start; }}
  @media (max-width:1180px) {{ .layout {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }}
  canvas {{ width:100%; height:auto; display:block; border-radius:8px; border:1px solid #334155; background:#020617; }}
  .controls-bar {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:10px; }}
  button, select {{ background:#1e293b; color:var(--text); border:1px solid #334155; padding:7px 12px; border-radius:6px; cursor:pointer; font-weight:600; font-size:13px; }}
  button.primary {{ background:#0284c7; border-color:#0284c7; }}
  button.danger {{ background:#b91c1c; border-color:#dc2626; }}
  button:disabled {{ opacity:.45; cursor:not-allowed; }}
  .scrubber {{ position:relative; margin-top:10px; }}
  input[type=range] {{ width:100%; accent-color:var(--accent); }}
  .markers {{ position:relative; height:10px; margin-top:2px; }}
  .marker {{ position:absolute; width:4px; height:10px; background:var(--danger); border-radius:2px; transform:translateX(-50%); cursor:pointer; }}
  .clip-list {{ display:flex; flex-direction:column; gap:8px; max-height:280px; overflow-y:auto; }}
  .clip-item {{ padding:9px 11px; background:#0b1220; border:1px solid #263248; border-left:4px solid var(--danger); border-radius:6px; display:flex; justify-content:space-between; align-items:center; gap:8px; }}
  .clip-item.sev-medium {{ border-left-color:var(--warn); }}
  .clip-item.sev-low {{ border-left-color:var(--accent); }}
  .prob-row {{ display:grid; grid-template-columns:130px 1fr 48px; align-items:center; gap:8px; font-size:12px; margin:4px 0; }}
  .bar-wrap {{ background:#020617; border-radius:4px; height:10px; overflow:hidden; border:1px solid #1e293b; }}
  .bar-fill {{ height:100%; background:linear-gradient(90deg, #0284c7, #38bdf8); }}
  .bar-fill.hot {{ background:linear-gradient(90deg, #d97706, #f59e0b); }}
  .keys-grid {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }}
  .keycap {{ padding:4px 8px; border-radius:5px; font-family:ui-monospace, monospace; font-size:12px; background:#020617; border:1px solid #334155; color:#64748b; }}
  .keycap.on {{ background:#0ea5e9; border-color:#7dd3fc; color:#020617; font-weight:700; box-shadow:0 0 10px rgba(56,189,248,.45); }}
  .mono {{ font-family:ui-monospace, monospace; font-size:12px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; background:#1e293b; }}
</style>
</head>
<body>
<div class="topbar">
  <div>
    <div class="title">TerrariaBench Playthrough &amp; Failure Clip Studio &mdash; {title}</div>
    <div class="sub">{models} &middot; <span id="runMeta"></span></div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <button id="exportFullBtn" class="primary">Record &amp; Download Full .webm</button>
    <button id="exportClipBtn" class="danger" disabled>Download Active Clip .webm</button>
  </div>
</div>

<div class="layout">
  <div>
    <div class="card">
      <canvas id="stage" width="960" height="600"></canvas>
      <div class="controls-bar">
        <button id="playBtn" class="primary">Play</button>
        <button id="prevBtn">&larr; Prev</button>
        <button id="nextBtn">Next &rarr;</button>
        <label class="mono">Speed:
          <select id="speedSel">
            <option value="1100">0.5x</option>
            <option value="650" selected>1x</option>
            <option value="320">2x</option>
            <option value="150">4x</option>
          </select>
        </label>
        <label class="mono"><input type="checkbox" id="loopClipChk" /> Loop Active Clip</label>
        <span id="frameCounter" class="mono" style="margin-left:auto;"></span>
      </div>
      <div class="scrubber">
        <input id="scrub" type="range" min="0" max="0" value="0" />
        <div id="markers" class="markers"></div>
      </div>
      <div style="margin-top:10px;">
        <div class="sub">Active Terraria Controls (driven by <code>gemini-decision-1</code>)</div>
        <div id="keysHud" class="keys-grid"></div>
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <strong>Demo Failure Clips (<span id="clipCount">0</span>)</strong>
        <span class="sub">Auto-tagged deaths, heavy hits, failed ops, night exposure</span>
      </div>
      <div id="clipList" class="clip-list"></div>
    </div>

    <div class="card">
      <strong>Supervisor Strategy (<code>{html.escape(recording['supervisor_model'])}</code>)</strong>
      <p id="strategyTxt" class="mono" style="white-space:pre-wrap;color:#cbd5e1;margin:8px 0 0;"></p>
    </div>

    <div class="card">
      <strong><code>gemini-decision-1</code> Calibrated Probabilities</strong>
      <div id="decisionBars" style="margin-top:8px;"></div>
    </div>

    <div class="card">
      <strong>Turn Action &amp; World Result</strong>
      <pre id="actionInfo" class="mono" style="white-space:pre-wrap;background:#020617;padding:8px;border-radius:6px;margin:8px 0 0;max-height:190px;overflow:auto;"></pre>
    </div>
  </div>
</div>

<script>
const REC = {payload_json};
const FRAMES = REC.frames || [];
const CLIPS = REC.failure_clips || [];
const TILE_COLORS = {{
  ".": null, "d": "#7c5132", "s": "#7a828e", "W": "#a46d3b", "=": "#d4a359",
  "c": "#c87137", "i": "#b8c0cc", "v": "#d8e4f0", "g": "#facc15", "a": "#4a4348",
  "T": "#2f9e44", "t": "#6b556b", "B": "#c2843e", "F": "#9a3412", "A": "#94a3b8",
  "C": "#f97316", "*": "#fde047", "H": "#eab308", "L": "#f43f5e", "D": "#854d0e",
  "~": "#ef4444", "@": "#38bdf8"
}};
const ALL_KEYS = ["W","A","S","D","SPACE","LMB","RMB","E:GRPL","H:HEAL","CTRL:SMART",
  "1:WEAP","2:PICK","3:AXE","4:WOOD","5:PLAT","6:TORC","7:STAT","8:POTI"];

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const scrub = document.getElementById("scrub");
const playBtn = document.getElementById("playBtn");
const speedSel = document.getElementById("speedSel");
const loopClipChk = document.getElementById("loopClipChk");
const exportClipBtn = document.getElementById("exportClipBtn");
const exportFullBtn = document.getElementById("exportFullBtn");

let idx = 0;
let playing = false;
let timer = null;
let activeClip = null;
let recordingWebm = false;

scrub.max = Math.max(0, FRAMES.length - 1);
document.getElementById("runMeta").textContent =
  `Frames: ${{FRAMES.length}} · Mode: ${{REC.live ? "LIVE API" : "Offline"}} · Composite: ${{REC.score?.composite ?? "N/A"}} · Deaths: ${{REC.final_deaths ?? 0}}`;
document.getElementById("clipCount").textContent = CLIPS.length;

const markersEl = document.getElementById("markers");
CLIPS.forEach((c) => {{
  const m = document.createElement("div");
  m.className = "marker";
  m.title = `${{c.title}} (Turn ${{c.start_turn}}-${{c.end_turn}})`;
  m.style.left = `${{FRAMES.length > 1 ? (c.focus_frame / (FRAMES.length - 1)) * 100 : 0}}%`;
  m.onclick = () => selectClip(c, true);
  markersEl.appendChild(m);
}});

const clipListEl = document.getElementById("clipList");
if (!CLIPS.length) {{
  clipListEl.innerHTML = `<div class="sub">No failures triggered in this run. Try boss/night tasks (e.g. eye-of-cthulhu) for failure clips!</div>`;
}} else {{
  CLIPS.forEach((c) => {{
    const row = document.createElement("div");
    row.className = `clip-item sev-${{c.severity}}`;
    row.innerHTML = `
      <div>
        <div><span class="badge">${{c.tag}}</span> <strong>${{c.title}}</strong></div>
        <div class="sub">Turns ${{c.start_turn}}&ndash;${{c.end_turn}} · ${{c.description}}</div>
      </div>
      <div style="display:flex;gap:6px;flex-shrink:0;">
        <button onclick='window._jumpClip("${{c.clip_id}}")'>Play</button>
        <button class="danger" onclick='window._exportClip("${{c.clip_id}}")'>.webm</button>
      </div>`;
    clipListEl.appendChild(row);
  }});
}}

window._jumpClip = (id) => {{
  const c = CLIPS.find((x) => x.clip_id === id);
  if (c) selectClip(c, true);
}};
window._exportClip = (id) => {{
  const c = CLIPS.find((x) => x.clip_id === id);
  if (c) exportWebm(c.start_frame, c.end_frame, `${{REC.run_id}}_${{c.clip_id}}.webm`);
}};

function selectClip(c, autoplay) {{
  activeClip = c;
  exportClipBtn.disabled = false;
  loopClipChk.checked = true;
  idx = c.start_frame;
  render();
  if (autoplay && !playing) togglePlay();
}}

const FRAME_IMAGES = {{}};
function frameImage(url) {{
  let im = FRAME_IMAGES[url];
  if (!im) {{
    im = new Image();
    im.onload = () => render();
    im.src = url;
    FRAME_IMAGES[url] = im;
  }}
  return im;
}}

function drawCanvas(f) {{
  const st = f.state || {{}};
  const grid = st.ascii_grid || [];
  const rows = grid.length || 35;
  const cols = (grid[0] || "").length || 61;
  const ctrl = f.controls || {{}};
  const fails = f.failure_events || [];

  ctx.fillStyle = "#060a14";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Header strip on canvas (included in exported .webm!)
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, canvas.width, 46);
  ctx.fillStyle = fails.length ? "#fca5a5" : "#38bdf8";
  ctx.font = "bold 15px ui-monospace, monospace";
  ctx.fillText(`Turn ${{f.turn}} | Win ${{st.window}}/${{st.total_windows}} (${{st.phase}}) | Ops ${{st.ops_left}}/${{st.ops_budget}} | HP ${{st.hp}}/${{st.max_hp}} | NetProg ${{st.phi}} | Tier ${{st.tech_tier}}`, 14, 28);

  // Viewport area
  const vpX = 16, vpY = 56, vpW = 671, vpH = 385;
  const isNight = st.phase === "Night";
  const grad = ctx.createLinearGradient(0, vpY, 0, vpY + vpH);
  grad.addColorStop(0, isNight ? "#090d24" : "#1d4ed8");
  grad.addColorStop(1, isNight ? "#111827" : "#0f172a");
  ctx.fillStyle = grad;
  ctx.fillRect(vpX, vpY, vpW, vpH);

  // Real-game recordings: show the exact frame the models were given (lighting and darkness included).
  const shot = st.frame_url ? frameImage(st.frame_url) : null;
  if (shot && shot.complete && shot.naturalWidth) ctx.drawImage(shot, vpX, vpY, vpW, vpH);

  const cw = vpW / cols, ch = vpH / rows;
  for (let r = 0; r < rows; r++) {{
    const line = grid[r] || "";
    for (let c = 0; c < cols; c++) {{
      const chTile = line[c] || ".";
      const color = TILE_COLORS[chTile];
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.fillRect(vpX + c * cw, vpY + r * ch, Math.ceil(cw), Math.ceil(ch));
      if (chTile === "@" || chTile === "*" || chTile === "C" || chTile === "H") {{
        ctx.fillStyle = "#fff";
        ctx.font = "bold 9px monospace";
        ctx.fillText(chTile, vpX + c * cw + 2, vpY + r * ch + 9);
      }}
    }}
  }}

  // Aim vector from player center
  const px = vpX + (cols / 2) * cw, py = vpY + (rows / 2) * ch;
  const aimMap = {{
    nearest_enemy: [56, -18], nearest_tree: [48, -10], nearest_ore: [34, 26],
    underfoot: [0, 44], ahead: [ctrl.left ? -52 : 52, 0], build_site: [-36, -24]
  }};
  let [ax, ay] = aimMap[ctrl.aim_target] || [45, 0];
  if (typeof ctrl.cursor_dx === "number") {{
    const ppt = (st.px_per_tile || 16) * (vpW / (st.shot_width || 640));
    ax = ctrl.cursor_dx * ppt;
    ay = ctrl.cursor_dy * ppt;
  }}
  ctx.strokeStyle = ctrl.use_item ? "#facc15" : "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + ax, py + ay);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px + ax, py + ay, 5, 0, Math.PI * 2);
  ctx.stroke();

  // Boss HP bar
  if (st.boss) {{
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(vpX + 40, vpY + 12, vpW - 80, 22);
    const pct = Math.max(0, Math.min(1, (st.boss.hp || 0) / (st.boss.max_hp || 2800)));
    ctx.fillStyle = "#dc2626";
    ctx.fillRect(vpX + 42, vpY + 14, (vpW - 84) * pct, 18);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 12px monospace";
    ctx.fillText(`BOSS: ${{st.boss.kind}} (${{Math.round(st.boss.hp)}} HP)`, vpX + 52, vpY + 27);
  }}

  // Failure vignette & callout banner on video canvas
  if (fails.length) {{
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 6;
    ctx.strokeRect(vpX + 3, vpY + 3, vpW - 6, vpH - 6);
    ctx.fillStyle = "rgba(185, 28, 28, 0.9)";
    ctx.fillRect(vpX + 16, vpY + vpH - 44, vpW - 32, 32);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px ui-monospace, monospace";
    ctx.fillText(`FAILURE CLIP: ${{fails[0].title}} — ${{fails[0].description.slice(0, 58)}}`, vpX + 26, vpY + vpH - 23);
  }}

  // Right sidebar inside canvas (Tool + HP + Keys + Decision confidence)
  const sx = 702, sw = 242;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(sx, vpY, sw, vpH);
  ctx.strokeStyle = "#263248";
  ctx.lineWidth = 1;
  ctx.strokeRect(sx, vpY, sw, vpH);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "bold 11px monospace";
  ctx.fillText("ACTION MACRO", sx + 12, vpY + 22);
  ctx.fillStyle = f.result_ok === false ? "#f87171" : "#38bdf8";
  ctx.font = "bold 14px monospace";
  ctx.fillText((f.tool || "start").slice(0, 24), sx + 12, vpY + 42);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "11px monospace";
  ctx.fillText(`Choice: ${{f.decision_choice || "-"}} (${{Math.round((f.decision_conf || 0) * 100)}}%)`, sx + 12, vpY + 62);
  ctx.fillText(`Sheltered: ${{st.sheltered ? "YES" : "NO"}} | Threat: ${{ctrl.threat_label || st.threat}}`, sx + 12, vpY + 80);

  // HP Bar
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(`HP ${{st.hp}}/${{st.max_hp}} | Def ${{st.defense}}`, sx + 12, vpY + 112);
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(sx + 12, vpY + 120, sw - 24, 14);
  const hpPct = Math.max(0, Math.min(1, (st.hp || 0) / Math.max(1, st.max_hp || 100)));
  ctx.fillStyle = hpPct > 0.5 ? "#22c55e" : (hpPct > 0.25 ? "#f59e0b" : "#ef4444");
  ctx.fillRect(sx + 12, vpY + 120, (sw - 24) * hpPct, 14);

  // Keys active on canvas
  ctx.fillStyle = "#94a3b8";
  ctx.fillText("GEMINI-DECISION-1 INPUTS", sx + 12, vpY + 162);
  const active = new Set(ctrl.active_keys || []);
  let kx = sx + 12, ky = vpY + 174;
  ALL_KEYS.slice(0, 10).forEach((k) => {{
    const on = active.has(k);
    const w = Math.max(28, k.length * 8 + 12);
    if (kx + w > sx + sw - 8) {{ kx = sx + 12; ky += 28; }}
    ctx.fillStyle = on ? "#0ea5e9" : "#1e293b";
    ctx.fillRect(kx, ky, w, 22);
    ctx.fillStyle = on ? "#020617" : "#64748b";
    ctx.font = "bold 11px monospace";
    ctx.fillText(k, kx + 6, ky + 15);
    kx += w + 5;
  }});

  // Hotbar strip along bottom of canvas
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(16, 456, canvas.width - 32, 128);
  ctx.strokeStyle = "#263248";
  ctx.strokeRect(16, 456, canvas.width - 32, 128);
  const slots = ["1:WEAP","2:PICK","3:AXE","4:WOOD","5:PLAT","6:TORCH","7:STAT","8:POT"];
  slots.forEach((label, i) => {{
    const bx = 28 + i * 114, by = 472;
    const sel = i === (ctrl.hotbar_index || 0);
    ctx.fillStyle = sel ? "#0284c7" : "#1e293b";
    ctx.fillRect(bx, by, 104, 40);
    ctx.strokeStyle = sel ? "#7dd3fc" : "#334155";
    ctx.strokeRect(bx, by, 104, 40);
    ctx.fillStyle = sel ? "#ffffff" : "#94a3b8";
    ctx.font = "bold 12px monospace";
    ctx.fillText(label, bx + 12, by + 25);
  }});
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "12px monospace";
  ctx.fillText(`Strategy (${{REC.supervisor_model}}): ${{(f.supervisor_strategy || "").slice(0, 102)}}`, 28, 542);
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(`Equipped: Wpn=${{st.equipped?.weapon || "-"}} | Arm=${{st.equipped?.armor || "none"}} | Pick=${{st.equipped?.pickaxe || "-"}} | Milestones: ${{(st.milestones || []).join(", ")}}`, 28, 566);
}}

function render() {{
  if (!FRAMES.length) return;
  const f = FRAMES[idx];
  scrub.value = idx;
  document.getElementById("frameCounter").textContent = `Frame ${{idx + 1}}/${{FRAMES.length}} (Turn ${{f.turn}})`;
  drawCanvas(f);

  const ctrl = f.controls || {{}};
  const active = new Set(ctrl.active_keys || []);
  document.getElementById("keysHud").innerHTML = ALL_KEYS.map((k) => {{
    const on = active.has(k);
    return `<span class="keycap ${{on ? "on" : ""}}">${{k}}</span>`;
  }}).join("");

  document.getElementById("strategyTxt").textContent = f.supervisor_strategy || "(initial state)";

  const decs = ctrl.decisions || {{}};
  const bars = [];
  for (const [qName, d] of Object.entries(decs)) {{
    if (d.type === "noul") {{
      const p = Number(d.value || 0);
      bars.push(`<div class="prob-row"><span class="mono">${{qName}}</span><div class="bar-wrap"><div class="bar-fill ${{p >= 0.55 ? "hot" : ""}}" style="width:${{Math.round(p*100)}}%"></div></div><span class="mono">${{Math.round(p*100)}}%</span></div>`);
    }} else if (d.type === "choice") {{
      bars.push(`<div class="prob-row"><span class="mono">${{qName}}</span><div class="bar-wrap"><div class="bar-fill" style="width:${{Math.round((d.confidence||0)*100)}}%"></div></div><span class="mono">${{d.value}}</span></div>`);
    }} else if (d.type === "score") {{
      bars.push(`<div class="prob-row"><span class="mono">${{qName}}</span><div class="bar-wrap"><div class="bar-fill hot" style="width:${{Math.round(((d.value||0)/3)*100)}}%"></div></div><span class="mono">${{Number(d.value||0).toFixed(2)}}</span></div>`);
    }}
  }}
  document.getElementById("decisionBars").innerHTML = bars.join("") || `<div class="sub">No decision telemetry on initial frame.</div>`;

  document.getElementById("actionInfo").textContent = JSON.stringify({{
    turn: f.turn, tool: f.tool, args: f.args, ok: f.result_ok,
    hp_delta: f.hp_delta, failures: f.failure_events, result_summary: f.result_summary
  }}, null, 2);
}}

function stepNext() {{
  if (activeClip && loopClipChk.checked) {{
    idx = idx >= activeClip.end_frame ? activeClip.start_frame : idx + 1;
  }} else {{
    idx = idx >= FRAMES.length - 1 ? 0 : idx + 1;
  }}
  render();
}}

function togglePlay() {{
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
  if (timer) {{ clearInterval(timer); timer = null; }}
  if (playing) timer = setInterval(stepNext, Number(speedSel.value));
}}

playBtn.onclick = togglePlay;
document.getElementById("prevBtn").onclick = () => {{ idx = Math.max(0, idx - 1); render(); }};
document.getElementById("nextBtn").onclick = () => {{ idx = Math.min(FRAMES.length - 1, idx + 1); render(); }};
scrub.oninput = (e) => {{ idx = Number(e.target.value); render(); }};
speedSel.onchange = () => {{ if (playing) {{ togglePlay(); togglePlay(); }} }};
exportFullBtn.onclick = () => exportWebm(0, FRAMES.length - 1, `${{REC.run_id}}_full.webm`);
exportClipBtn.onclick = () => activeClip && exportWebm(activeClip.start_frame, activeClip.end_frame, `${{REC.run_id}}_${{activeClip.clip_id}}.webm`);

async function exportWebm(startF, endF, filename) {{
  if (recordingWebm || !FRAMES.length) return;
  recordingWebm = true;
  if (playing) togglePlay();
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  const rec = new MediaRecorder(stream, {{ mimeType: mime }});
  const chunks = [];
  rec.ondataavailable = (ev) => {{ if (ev.data.size) chunks.push(ev.data); }};
  rec.onstop = () => {{
    const blob = new Blob(chunks, {{ type: "video/webm" }});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    recordingWebm = false;
  }};
  rec.start();
  for (let i = startF; i <= endF; i++) {{
    idx = i;
    render();
    await new Promise((r) => setTimeout(r, 450));
  }}
  rec.stop();
}}

render();
</script>
</body>
</html>"""


def _render_studio_index_html() -> str:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RECORDINGS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rid = data.get("run_id", p.stem)
        clips = len(data.get("failure_clips") or [])
        frames = len(data.get("frames") or [])
        comp = (data.get("score") or {}).get("composite")
        comp_s = f"{comp:.4f}" if isinstance(comp, (int, float)) else "live"
        rows.append(
            f"<tr><td><a href='/play/{html.escape(rid)}'>{html.escape(rid)}</a></td>"
            f"<td>{html.escape(str(data.get('task_id', 'real')))}</td>"
            f"<td>{html.escape(str(data.get('supervisor_model', '')))} + {html.escape(str(data.get('decision_model', '')))}</td>"
            f"<td>{frames}</td><td><strong>{clips}</strong></td><td>{comp_s}</td>"
            f"<td><a href='{html.escape(rid)}.html'>static html</a></td></tr>"
        )
    table_body = "".join(rows) or "<tr><td colspan='7'>No recordings yet.</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<title>TerrariaBench Live Play &amp; Failure Clip Index</title>
<style>body{{font-family:system-ui,sans-serif;background:#090d16;color:#e2e8f0;margin:24px;}}
table{{width:100%;border-collapse:collapse;background:#111827;border:1px solid #263248;}}
th,td{{padding:10px 12px;border-bottom:1px solid #263248;text-align:left;}}a{{color:#38bdf8;}}</style></head>
<body><h1>TerrariaBench Live Play &amp; Failure Clip Studio</h1>
<p>Every playthrough records full frame-by-frame viewports, <code>gemini-decision-1</code> control inputs, and tagged failure clips with 1-click <code>.webm</code> export.</p>
<table><tr><th>Run ID</th><th>Task</th><th>Models</th><th>Frames</th><th>Failure Clips</th><th>Score</th><th>File</th></tr>
{table_body}</table></body></html>"""


def run_recorded_episode(
    task_id: str = "first-night",
    seed: int = 7,
    live: bool = False,
    max_turns: int = 45,
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL,
    decision_model: str = DEFAULT_DECISION_MODEL,
    out_dir: Path | None = None,
    run_id: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run an Episode with ``DecisionController``, capturing every frame & failure clips."""
    if live:
        load_dotenv_if_needed()
    dest = Path(out_dir) if out_dir else RECORDINGS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    mode_tag = "live" if live else "offline"
    rid = run_id or f"{task_id}_seed{seed}_{mode_tag}"

    ep = Episode(task_id, seed, redact_seed=True)
    controller = DecisionController(
        supervisor_model=supervisor_model,
        decision_model=decision_model,
        live=live,
    )
    obs: dict[str, Any] = {
        "ok": True,
        "header": ep.header(),
        "prompt": ep.task.prompt,
        "window": ep.world.window,
        "total_windows": ep.task.windows,
        "ops_left": ep.ops_left,
        "ops_budget": ep.window_budget,
    }
    history: list[dict[str, Any]] = []
    before_snap = _snapshot_episode(ep)
    frames: list[dict[str, Any]] = [{
        "turn": 0,
        "tool": "spawn",
        "args": {},
        "result_ok": True,
        "result_summary": {"message": "Episode started"},
        "hp_delta": 0,
        "controls": {"active_keys": [], "aim_target": "ahead", "hotbar_index": 0, "threat_label": "Calm", "decisions": {}},
        "supervisor_strategy": "",
        "decision_choice": "spawn",
        "decision_conf": 1.0,
        "failure_events": [],
        "state": before_snap,
    }]

    for turn in range(1, max_turns + 1):
        if ep.done:
            break
        dec = controller.decide_turn(obs, history[-8:], ep)
        tool, args = dec["tool"], dec["args"]
        try:
            res = ep.step(tool, **args)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
            res = {"ok": False, "error": f"invalid call {tool}({args}): {exc}", "header": ep.header()}

        after_snap = _snapshot_episode(ep)
        failures = detect_step_failures(
            before_snap,
            after_snap,
            tool,
            res,
            dec["decision_choice"],
            dec["decision_conf"],
            dec["replan_triggered"],
        )
        compact_res = {k: v for k, v in res.items() if k not in ("viewport_png_b64", "ascii_grid")}
        frames.append({
            "turn": turn,
            "tool": tool,
            "args": args,
            "result_ok": bool(res.get("ok", True)),
            "result_summary": compact_res,
            "hp_delta": after_snap["hp"] - before_snap["hp"],
            "controls": dec["controls"],
            "supervisor_strategy": dec["supervisor_strategy"],
            "decision_choice": dec["decision_choice"],
            "decision_conf": dec["decision_conf"],
            "failure_events": failures,
            "state": after_snap,
        })
        if verbose:
            tags = ",".join(f["tag"] for f in failures)
            tag_str = f" [FAIL: {tags}]" if tags else ""
            print(f"[{turn:3d}] {tool}({json.dumps(args, default=str)}) keys={dec['controls']['active_keys']} -> ok={res.get('ok', True)}{tag_str}")

        history.append({"turn": turn, "tool": tool, "args": args, "result": compact_res})
        obs = res
        before_snap = after_snap

    if not ep.done:
        ep.finalize()
        settled_snap = _snapshot_episode(ep)
        settled_failures = detect_step_failures(
            before_snap, settled_snap, "settle_remaining", {"ok": True}, "end_window", 1.0, False
        )
        frames.append({
            "turn": len(frames),
            "tool": "settle_remaining",
            "args": {},
            "result_ok": True,
            "result_summary": {"message": "Remaining windows settled"},
            "hp_delta": settled_snap["hp"] - before_snap["hp"],
            "controls": frames[-1]["controls"],
            "supervisor_strategy": frames[-1]["supervisor_strategy"],
            "decision_choice": "end_window",
            "decision_conf": 1.0,
            "failure_events": settled_failures,
            "state": settled_snap,
        })

    clips = build_failure_clips(frames)
    ledger = ep.finalize()
    recording = {
        "run_id": rid,
        "task_id": task_id,
        "task_title": TASKS[task_id].title,
        "seed": seed,
        "live": live,
        "supervisor_model": supervisor_model,
        "decision_model": decision_model,
        "final_deaths": ep.world.deaths,
        "stats": controller.agent.stats,
        "score": {"windows": len(ledger["windows"]), "phi_final": ledger["phi_final"], "deaths": ledger["deaths"]},
        "failure_clips": clips,
        "frames": frames,
    }
    json_path = dest / f"{rid}.json"
    html_path = dest / f"{rid}.html"
    json_path.write_text(json.dumps(recording), encoding="utf-8")
    html_path.write_text(render_player_html(recording), encoding="utf-8")
    if dest == RECORDINGS_DIR:
        (dest / "index.html").write_text(_render_studio_index_html(), encoding="utf-8")
    return {
        "run_id": rid,
        "json_path": str(json_path),
        "html_path": str(html_path),
        "frames": len(frames),
        "failure_clips": clips,
        "deaths": ep.world.deaths,
        "stats": controller.agent.stats,
    }


def create_live_app(
    live_models: bool = True,
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL,
    decision_model: str = DEFAULT_DECISION_MODEL,
    vision_model: str | None = None,
) -> FastAPI:
    """Create the port 8765 Live Play Studio + real Terraria (tModLoader) bridge."""
    if live_models:
        load_dotenv_if_needed()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    controller = DecisionController(
        supervisor_model=supervisor_model,
        decision_model=decision_model,
        vision_model=vision_model,
        live=live_models and has_key,
    )
    real_turn = [0]
    bridge = BridgeSession(task_id="first-night", seed=7)
    real_frames: list[dict[str, Any]] = []
    real_lock = threading.Lock()
    app = FastAPI(title="TerrariaBench Live Play Studio & tModLoader Bridge (8765)")
    app.state.bridge = bridge
    app.state.controller = controller

    @app.get("/", response_class=HTMLResponse)
    @app.get("/theater", response_class=HTMLResponse)
    async def theater_index() -> HTMLResponse:
        return HTMLResponse(_render_studio_index_html())

    @app.get("/play/{run_id}", response_class=HTMLResponse)
    async def play_run(run_id: str) -> HTMLResponse:
        path = RECORDINGS_DIR / f"{Path(run_id).name}.json"
        if not path.exists():
            return HTMLResponse(f"<h1>Recording not found: {html.escape(run_id)}</h1>", status_code=404)
        return HTMLResponse(render_player_html(json.loads(path.read_text(encoding="utf-8"))))

    @app.get("/api/recordings")
    async def api_recordings() -> dict[str, Any]:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        items = []
        for p in sorted(RECORDINGS_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                items.append({
                    "run_id": d.get("run_id", p.stem),
                    "task_id": d.get("task_id"),
                    "frames": len(d.get("frames") or []),
                    "failure_clips": len(d.get("failure_clips") or []),
                })
            except (OSError, ValueError):
                continue
        return {"ok": True, "recordings": items}

    @app.get("/api/recordings/{run_id}")
    async def api_recording_detail(run_id: str) -> JSONResponse:
        path = RECORDINGS_DIR / f"{Path(run_id).name}.json"
        if not path.exists():
            return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/real/latest")
    async def api_real_latest() -> dict[str, Any]:
        with real_lock:
            return {"ok": True, "frames": len(real_frames), "latest": real_frames[-1] if real_frames else None}

    @app.get("/play/real_terraria_live_frames/{name}")
    async def real_frame_png(name: str) -> Response:
        path = RECORDINGS_DIR / REAL_FRAMES_DIRNAME / Path(name).name
        if path.suffix != ".png" or not path.exists():
            return Response(status_code=404)
        return Response(path.read_bytes(), media_type="image/png")

    @app.post("/real_game_tick")
    def real_game_tick(payload: dict[str, Any]) -> dict[str, Any]:
        # Sync handler: FastAPI runs it in a worker thread, so slow model calls never block the studio pages.
        decision = controller.decide_real_game_tick(payload)
        if decision["source"] == "outdated_mod":
            return decision
        hud = sanitize_hud(payload)
        # Referee block = ground truth for scoring and failure clips only. It is never passed to a model.
        referee = payload.get("referee") if isinstance(payload.get("referee"), dict) else {}
        with real_lock:
            real_turn[0] += 1
            turn_idx = real_turn[0]
            cur_hp = _as_int(hud.get("hp"), 100)
            cur_deaths = _as_int(referee.get("deaths"), 0)
            prev_hp = real_frames[-1]["state"]["hp"] if real_frames else cur_hp
            prev_deaths = real_frames[-1]["state"]["deaths"] if real_frames else cur_deaths
            fails: list[dict[str, Any]] = []
            if cur_deaths > prev_deaths:
                fails.append({"tag": "DEATH", "severity": "critical", "title": "Died in Real Terraria", "description": "Player health dropped to 0."})
            elif cur_hp - prev_hp <= -15:
                fails.append({"tag": "HEAVY_DAMAGE", "severity": "high", "title": f"Heavy Hit ({cur_hp - prev_hp} HP)", "description": f"HP fell from {prev_hp} to {cur_hp}."})
            tiers = decision["tiers"]
            down = [f"{k}={t['status']}" for k, t in tiers.items() if t["status"] not in ("LIVE", "OFFLINE", "WAITING")]
            if down and controller.live:
                fails.append({"tag": "MODEL_TIER_DOWN", "severity": "high", "title": "Model tier not live", "description": ", ".join(down)})

            frame_url = ""
            shot = payload.get("screenshot_b64")
            if shot:
                try:
                    frames_dir = RECORDINGS_DIR / REAL_FRAMES_DIRNAME
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    (frames_dir / f"{turn_idx}.png").write_bytes(base64.b64decode(shot))
                    (frames_dir / f"{turn_idx - REAL_FRAMES_KEPT}.png").unlink(missing_ok=True)
                    frame_url = f"{REAL_FRAMES_DIRNAME}/{turn_idx}.png"
                except (OSError, ValueError):
                    frame_url = ""
            view = hud.get("view") if isinstance(hud.get("view"), dict) else {}
            frame = {
                "turn": turn_idx,
                "tool": decision["macro"]["tool"],
                "args": decision["macro"]["args"],
                "result_ok": decision["source"] in ("decisions_api", "decisions_api_routine"),
                "result_summary": {"source": decision["source"], **decision["hud"]},
                "hp_delta": cur_hp - prev_hp,
                "controls": decision["controls"],
                "supervisor_strategy": decision["strategy"],
                "decision_choice": " ".join(decision["controls"]["active_keys"]) or "no keys",
                "decision_conf": decision["confidence"],
                "failure_events": fails,
                "tiers": tiers,
                "plan": decision["plan"],
                "plan_no": decision["plan_no"],
                "vision_report": decision["vision_report"],
                "source": decision["source"],
                "state": {
                    "window": 1,
                    "total_windows": 1,
                    "phase": "Day" if referee.get("dayTime", True) else "Night",
                    "ops_left": turn_idx,
                    "ops_budget": turn_idx,
                    "hp": cur_hp,
                    "max_hp": _as_int(hud.get("max_hp"), 100),
                    "defense": _as_int(referee.get("defense"), 0),
                    "phi": 0.0,
                    "tech_tier": 0,
                    "sheltered": False,
                    "threat": "n/a",
                    "deaths": cur_deaths,
                    "equipped": {},
                    "milestones": [],
                    "ascii_grid": [],
                    "frame_url": frame_url,
                    "px_per_tile": view.get("px_per_tile"),
                    "shot_width": view.get("shot_width"),
                    "held_item": hud.get("held_item", ""),
                    "inventory_open": bool(hud.get("inventory_open")),
                    "lighting": (decision["vision_report"] or {}).get("lighting"),
                    "referee": referee,
                },
            }
            real_frames.append(frame)
            if len(real_frames) > REAL_FRAMES_KEPT:
                del real_frames[:-REAL_FRAMES_KEPT]
            if turn_idx == 1 or turn_idx % 5 == 0 or fails:
                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                rec = {
                    "run_id": "real_terraria_live",
                    "task_id": "real-terraria",
                    "task_title": "Live tModLoader Session (human-parity, protocol 2)",
                    "seed": 0,
                    "live": controller.live,
                    "supervisor_model": controller.agent.model,
                    "vision_model": controller.real.tiers["vision"].model,
                    "decision_model": controller.agent.decisions.model,
                    "tiers": tiers,
                    "final_deaths": cur_deaths,
                    "stats": controller.agent.stats,
                    "score": {"windows": 1, "phi_final": 0.0, "deaths": cur_deaths},
                    "failure_clips": build_failure_clips(real_frames),
                    "frames": real_frames,
                }
                (RECORDINGS_DIR / "real_terraria_live.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
                (RECORDINGS_DIR / "real_terraria_live.html").write_text(render_player_html(rec), encoding="utf-8")
        return decision

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return {
            "ok": True,
            "task": bridge.ep.task.id,
            "seed": bridge.ep.seed,
            "window": bridge.ep.world.window,
            "ops_left": bridge.ep.ops_left,
            "done": bridge.ep.done,
            "header": bridge.ep.header(),
            "real_frames": len(real_frames),
        }

    @app.post("/")
    async def rpc_endpoint(request: Request) -> Any:
        payload = await request.json()
        if isinstance(payload, list):
            return [bridge.handle_rpc(item) for item in payload]
        return bridge.handle_rpc(payload)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerrariaBench gemini-decision-1 Controls & Failure Clip Studio")
    parser.add_argument("--task", default="first-night", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--live", action="store_true", help="Call live gemini-4-hydrogen + gemini-decision-1")
    parser.add_argument("--max-turns", type=int, default=45)
    parser.add_argument("--supervisor", default=DEFAULT_SUPERVISOR_MODEL)
    parser.add_argument("--decision-model", default=DEFAULT_DECISION_MODEL)
    parser.add_argument("--vision-model", default=None,
                        help="Multimodal model that reads the real-game screenshot (default: $TERRARIA_GEMINI_VISION_MODEL, else the supervisor model)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--serve", action="store_true", help="Serve live studio & real-game bridge on --port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.serve:
        import uvicorn

        load_dotenv_if_needed()
        app = create_live_app(
            live_models=args.live or bool(os.environ.get("GEMINI_API_KEY")),
            supervisor_model=args.supervisor,
            decision_model=args.decision_model,
            vision_model=args.vision_model,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    out = run_recorded_episode(
        task_id=args.task,
        seed=args.seed,
        live=args.live,
        max_turns=args.max_turns,
        supervisor_model=args.supervisor,
        decision_model=args.decision_model,
        run_id=args.run_id,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
