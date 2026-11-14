"""Two-tier Google Gemini agent, Image Blueprint Synthesizer, and Decisions API wrapper.

Architecture (live mode):

* **Supervisor** - a large Gemini model (``TERRARIA_GEMINI_MODEL``) that owns
  strategy. It is called at the start of each window, and again whenever the
  decision model asks for a re-plan, and returns a short strategy plus a menu
  of concrete candidate tool calls.
* **Decision model** - the Gemini Decisions API (``gemini-decision-1``), a
  fast closed-set model. Every turn it picks exactly one action from the
  supervisor's menu (or ``end_window`` / ``replan``) given the latest state.

Offline (``live=False``) the agent is a deterministic scripted stand-in used
by the test suite; it never touches the network and is not a Gemini result.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

from ..engine import AIR, CAMPFIRE, DOOR, PLATFORM, TORCH, WOOD, WORKBENCH
from ..tools import TOOL_NAMES, TOOL_SCHEMAS, Episode


def _verify_shelter_matrix(matrix: list[list[int]]) -> bool:
    if not matrix or len(matrix) < 5 or len(matrix[0]) < 7:
        return False
    rows, cols = len(matrix), len(matrix[0])
    has_wb = any(WORKBENCH in row for row in matrix)
    has_light = any(TORCH in row or CAMPFIRE in row for row in matrix)
    top_bottom_closed = all(t in (WOOD, DOOR) for t in matrix[0]) and all(t in (WOOD, DOOR) for t in matrix[-1])
    sides_closed = all(row[0] in (WOOD, DOOR) and row[-1] in (WOOD, DOOR) for row in matrix)
    interior = (rows - 2) * (cols - 2)
    return has_wb and has_light and top_bottom_closed and sides_closed and interior >= 20


def synthesize_blueprint_matrix(
    structure_type: str = "shelter",
    width: int = 9,
    height: int = 6,
    client: Any = None,
) -> list[list[int]]:
    """Synthesize a validated 2D tile matrix for ``build_blueprint(matrix=...)``."""
    image_model = os.environ.get("TERRARIA_GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image-preview")
    if client is not None and hasattr(client, "models"):
        try:
            prompt = (
                f"Generate a {height}x{width} integer 2D tile matrix JSON for a Terraria {structure_type} "
                f"using tile codes AIR=0, WOOD=3, PLATFORM=4, WORKBENCH=12, CAMPFIRE=15, TORCH=16, DOOR=19."
            )
            resp = client.models.generate_content(model=image_model, contents=prompt)
            match = re.search(r"\[\s*\[.*?\]\s*\]", getattr(resp, "text", "") or "", re.S)
            if match:
                cand = json.loads(match.group(0))
                if structure_type != "shelter" or _verify_shelter_matrix(cand):
                    return cand
        except Exception:
            pass

    if structure_type == "arena":
        w = max(26, int(width))
        h = max(7, int(height))
        mat = [[-1 for _ in range(w)] for _ in range(h)]
        for r in (0, 3, h - 1):
            mat[r] = [PLATFORM] * w
        mat[2][w // 2] = CAMPFIRE
        return mat

    # Default enclosed shelter: 6x9 => 7x4 = 28 interior tiles, DOOR at (4, 0), WORKBENCH at (4, 2), TORCH at (2, 4)
    w = max(9, int(width))
    h = max(6, int(height))
    mat = [[AIR for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                mat[r][c] = WOOD
    mat[min(4, h - 2)][0] = DOOR
    mat[min(4, h - 2)][2] = WORKBENCH
    mat[min(2, h - 2)][w // 2] = TORCH
    return mat


class GeminiLiveError(RuntimeError):
    """A live Gemini call failed. Live mode never falls back to the scripted policy."""


def _retry(fn: Any, attempts: int = 4, base_delay: float = 2.0) -> Any:
    """Call ``fn`` retrying transient 429 / 5xx errors with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            transient = code == 429 or (isinstance(code, int) and 500 <= code < 600)
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _api_question(spec: dict[str, Any]) -> dict[str, Any]:
    """Translate a question spec into the Decisions API schema.

    The API takes ``type`` in {"noul", "choice", "score"}, a required
    ``instructions`` string, and ``criteria`` (option-id -> description map for
    choice, ordered level descriptions for score). Legacy ``bool`` and
    ``choices`` / ``max_score`` spellings are accepted and converted.
    """
    q_type = {"bool": "noul"}.get(spec.get("type", "choice"), spec.get("type", "choice"))
    out: dict[str, Any] = {"type": q_type, "instructions": spec.get("instructions") or "Pick the best option."}
    if q_type == "choice":
        crit = spec.get("criteria")
        if not isinstance(crit, dict):
            crit = {str(c): str(c) for c in (spec.get("choices") or ["shelter", "gather", "combat"])}
        out["criteria"] = {str(k): str(v) for k, v in crit.items()}
    elif q_type == "score":
        crit = spec.get("criteria")
        if not isinstance(crit, (list, tuple)):
            crit = [f"level {i}" for i in range(int(spec.get("max_score", 4)) + 1)]
        out["criteria"] = [str(c) for c in crit]
    return out


def _normalize_answer(q_type: str, ans: Any) -> dict[str, Any]:
    """Flatten a Decisions API answer to ``{type, value, confidence, probabilities}``."""
    out = {"type": q_type, "value": _field(ans, q_type), "confidence": float(_field(ans, "confidence", 0.0) or 0.0)}
    probs = _field(ans, "probabilities")
    if probs is not None:
        out["probabilities"] = dict(probs) if isinstance(probs, dict) else list(probs)
    return out


class GeminiDecisionsWrapper:
    """Wrapper around the Gemini Decisions API (``gemini-decision-1``).

    ``decide`` returns ``{"model", "source", "answers"}`` where every answer is
    normalised to ``{type, value, confidence[, probabilities]}``. With no client
    and no ``GEMINI_API_KEY`` it uses a deterministic local stand-in. With
    ``strict=True`` an unavailable or failing API raises ``GeminiLiveError``
    instead of silently using the stand-in.
    """

    def __init__(self, model: str | None = None, client: Any = None, strict: bool = False):
        self.model = model or os.environ.get("TERRARIA_GEMINI_DECISION_MODEL", "gemini-decision-1")
        self.client = client
        self.strict = strict
        self.last_error: str | None = None
        self.calls = 0

    def _get_live_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        try:
            from google import genai

            self.client = genai.Client(http_options={"api_version": "v1beta"})
            return self.client
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def decide(
        self,
        questions: dict[str, dict[str, Any]],
        context_text: str,
        viewport_png_b64: str | None = None,
    ) -> dict[str, Any]:
        client = self._get_live_client()
        if client is not None and hasattr(client, "decisions"):
            api_questions = {name: _api_question(spec) for name, spec in questions.items()}
            blocks: list[dict[str, Any]] = [{"type": "text", "text": context_text}]
            if viewport_png_b64:
                blocks.append({"type": "image", "data": viewport_png_b64, "mime_type": "image/png", "resolution": "low"})
            try:
                resp = _retry(lambda: client.decisions.create(model=self.model, input=blocks, questions=api_questions))
                raw = _field(resp, "answers") or {}
                self.calls += 1
                return {
                    "model": self.model,
                    "source": "decisions_api",
                    "answers": {name: _normalize_answer(q["type"], raw[name]) for name, q in api_questions.items()},
                }
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if self.strict:
                    raise GeminiLiveError(f"Decisions API call failed: {self.last_error}") from exc
        elif self.strict:
            raise GeminiLiveError(
                "Decisions API unavailable: need GEMINI_API_KEY and google-genai>=3.2 "
                f"(client.decisions). {self.last_error or ''}".strip()
            )

        # Deterministic offline stand-in with the same normalised answer shape
        lower = context_text.lower()
        answers: dict[str, Any] = {}
        for q_name, q_spec in questions.items():
            api_q = _api_question(q_spec)
            q_type = api_q["type"]
            if q_type == "choice":
                opts = list(api_q["criteria"])
                weights = []
                for opt in opts:
                    o = str(opt).lower()
                    score = 1.0
                    if "unsheltered" in lower and ("shelter" in o or "wood" in o):
                        score += 4.0
                    elif "tier:0" in lower and ("craft" in o or "gear" in o or "mine" in o):
                        score += 3.0
                    elif "boss" in lower and ("combat" in o or "arena" in o):
                        score += 3.5
                    weights.append(score)
                total = sum(weights) or 1.0
                probs = {str(o): round(w / total, 4) for o, w in zip(opts, weights)}
                winner = max(probs, key=lambda k: probs[k])
                answers[q_name] = {
                    "type": "choice",
                    "value": winner,
                    "confidence": round(probs[winner], 4),
                    "probabilities": probs,
                }
            elif q_type == "noul":
                val = "ready" in lower or ("sheltered" in lower and "night" not in lower)
                answers[q_name] = {"type": "noul", "value": 0.85 if val else 0.15, "confidence": 0.85}
            else:
                max_lvl = len(api_q["criteria"]) - 1
                val = min(max_lvl, 3 if "sheltered" in lower else 1)
                answers[q_name] = {"type": "score", "value": val, "confidence": 0.82}
        return {"model": self.model, "source": "deterministic_fallback", "answers": answers}


SUPERVISOR_SYSTEM = (
    "You are the strategy supervisor for an agent playing TerrariaBench, a headless Terraria-like survival "
    "benchmark. You do not act directly. A small, fast decision model executes: every turn it picks exactly ONE "
    "entry from the candidate menu you write, using the latest game state. Each tool call costs one operation "
    "from the window budget (terraria_status is free); the window ends when operations run out or end_window is "
    "called.\n"
    "Write a menu that covers the rest of this window: the main line of play in order, plus contingencies "
    "(low HP, night falls, a craft fails for missing materials, a boss appears). Every candidate must be a "
    "complete tool call with concrete arguments. The decision model can pick the same candidate repeatedly.\n"
    "Return JSON only: {\"strategy\": \"<2-4 sentences>\", \"candidates\": [{\"id\": \"short_snake_case\", "
    "\"tool\": \"<tool name>\", \"args\": {...}, \"when\": \"<one sentence: the state in which this is the right "
    "next action>\"}]} with 3 to 12 candidates."
)

DECISION_INSTRUCTIONS = (
    "A supervisor wrote a strategy and a menu of candidate actions for a player in a Terraria-like survival "
    "game. Given the current game state and the result of the last action, which single candidate should the "
    "player execute right now?"
)

MAX_CANDIDATES = 12
_PARAM_CASTS = {"int": int, "float": float, "str": str}


def _tool_catalog() -> str:
    return "\n".join(
        f"- {name}({', '.join(f'{k}: {v}' for k, v in TOOL_SCHEMAS[name]['params'].items())}): "
        f"{TOOL_SCHEMAS[name]['description']}"
        for name in TOOL_NAMES
    )


def sanitize_tool_args(tool: str, args: Any) -> dict[str, Any]:
    """Keep only arguments the tool declares, coercing simple scalar types."""
    params = TOOL_SCHEMAS[tool]["params"]
    out: dict[str, Any] = {}
    for key, val in (args.items() if isinstance(args, dict) else ()):
        if key not in params or val is None:
            continue
        base = str(params[key]).split("=")[0].split("|")[0].strip()
        try:
            if base == "bool":
                val = val if isinstance(val, bool) else str(val).strip().lower() in ("1", "true", "yes")
            elif base in _PARAM_CASTS:
                val = _PARAM_CASTS[base](val)
        except (TypeError, ValueError):
            continue
        out[key] = val
    return out


def parse_supervisor_plan(text: str) -> dict[str, Any]:
    """Parse and validate the supervisor's JSON plan. Raises ``ValueError`` if unusable."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("supervisor returned no JSON object")
    data = json.loads(match.group(0))
    candidates: dict[str, dict[str, Any]] = {}
    for i, cand in enumerate(data.get("candidates") or []):
        if not isinstance(cand, dict) or cand.get("tool") not in TOOL_SCHEMAS:
            continue
        cid = re.sub(r"[^A-Za-z0-9_-]", "_", str(cand.get("id") or f"a{i}"))[:48] or f"a{i}"
        if cid in ("end_window", "replan") or cid in candidates:
            cid = f"{cid}_{i}"
        candidates[cid] = {
            "tool": cand["tool"],
            "args": sanitize_tool_args(cand["tool"], cand.get("args")),
            "when": str(cand.get("when") or "")[:300],
        }
        if len(candidates) >= MAX_CANDIDATES:
            break
    if not candidates:
        raise ValueError("supervisor plan has no valid candidates")
    return {"strategy": str(data.get("strategy") or "")[:1200], "candidates": candidates}


class GeminiAgent:
    """Two-tier Gemini agent: a supervisor model plans, the Decisions API acts.

    ``supervisor_fn(prompt_text, png_b64) -> str`` and ``decisions`` can be
    injected for tests; by default both talk to the Gemini Developer API.
    """

    def __init__(
        self,
        model: str | None = None,
        use_decisions_api: bool = True,
        live: bool = False,
        supervisor_fn: Any = None,
        decisions: GeminiDecisionsWrapper | None = None,
        min_confidence: float = 0.35,
        max_replans_per_window: int = 3,
    ):
        self.name = "gemini"
        self.model = model or os.environ.get("TERRARIA_GEMINI_MODEL", "gemini-4-hydrogen")
        self.use_decisions_api = use_decisions_api
        self.live = live
        self.decisions = decisions or GeminiDecisionsWrapper(strict=live)
        self.supervisor_fn = supervisor_fn
        self.min_confidence = float(min_confidence)
        self.max_replans_per_window = int(max_replans_per_window)
        self._client: Any = None
        self._plan: dict[str, Any] | None = None
        self._plan_window = -1
        self._replans = 0
        self._last_choice: str | None = None
        self.stats = {"supervisor_calls": 0, "decision_calls": 0, "replans": 0, "low_confidence": 0}
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------- live tier 1
    def _genai_client(self) -> Any:
        if self._client is None:
            if not os.environ.get("GEMINI_API_KEY"):
                raise GeminiLiveError("GEMINI_API_KEY is not set")
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiLiveError("google-genai is not installed (pip install 'google-genai>=3.2')") from exc
            self._client = genai.Client(http_options={"api_version": "v1beta"})
            if self.decisions.client is None:
                self.decisions.client = self._client
        return self._client

    def _call_supervisor_model(
        self,
        prompt_text: str,
        png_b64: str | None,
        system: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> str:
        """Call a generate_content model with a JSON response. ``system``/``model`` default to the supervisor's.

        ``thinking_level`` (e.g. "low") is a latency hint. If the model rejects it, it is dropped for the rest of
        the run and the call is repeated without it.
        """
        from google.genai import types

        model = model or self.model
        system = system or SUPERVISOR_SYSTEM
        client = self._genai_client()
        parts: list[Any] = [prompt_text]
        if png_b64:
            parts.append(types.Part.from_bytes(data=base64.b64decode(png_b64), mime_type="image/png"))
        rejected = self.__dict__.setdefault("_thinking_rejected", set())
        if thinking_level and (model, thinking_level) not in rejected:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                    ),
                )
                return resp.text or ""
            except Exception:
                rejected.add((model, thinking_level))
        try:
            resp = _retry(lambda: client.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                ),
            ))
        except Exception as exc:
            raise GeminiLiveError(f"model '{model}' call failed: {type(exc).__name__}: {exc}") from exc
        return resp.text or ""

    @staticmethod
    def _obs_text(observation: dict[str, Any], limit: int = 6000) -> str:
        clean = {k: v for k, v in observation.items() if k not in ("viewport_png_b64", "text", "prompt")}
        return json.dumps(clean, sort_keys=True, default=str)[:limit]

    def _supervise(self, observation: dict[str, Any], context: list[dict[str, Any]], ep: Episode, reason: str) -> None:
        status = ep.step("terraria_status")  # free: costs no operation
        prompt_text = (
            f"TASK\n{ep.task.prompt}\n\nTOOLS\n{_tool_catalog()}\n\n"
            f"WHY YOU ARE BEING CALLED\n{reason}\n\n"
            f"STATUS\n{self._obs_text(status)}\n\n"
            f"LAST ACTION RESULT\n{self._obs_text(observation)}\n\n"
            f"RECENT HISTORY\n{json.dumps(context[-8:], sort_keys=True, default=str)[:12000]}"
        )
        call = self.supervisor_fn or self._call_supervisor_model
        last_exc: Exception | None = None
        for _ in range(2):  # one retry on an unparseable plan
            try:
                self.stats["supervisor_calls"] += 1
                self._plan = parse_supervisor_plan(call(prompt_text, observation.get("viewport_png_b64")))
                self._plan_window = ep.world.window
                self.trace.append({"tier": "supervisor", "window": ep.world.window, "reason": reason,
                                   "strategy": self._plan["strategy"], "candidates": list(self._plan["candidates"])})
                return
            except GeminiLiveError:
                raise
            except Exception as exc:
                last_exc = exc
        raise GeminiLiveError(f"supervisor model '{self.model}' gave no usable plan: {last_exc}") from last_exc

    # ------------------------------------------------------------- live tier 2
    def _decide(self, observation: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
        assert self._plan is not None
        cands = self._plan["candidates"]
        if not self.use_decisions_api:  # supervisor-only: walk the menu in order
            order = list(cands)
            nxt = order[order.index(self._last_choice) + 1] if self._last_choice in order[:-1] else (
                "end_window" if self._last_choice == order[-1] else order[0])
            return nxt, 1.0, {}
        criteria = {
            cid: f"{c['tool']} {json.dumps(c['args'], sort_keys=True)} - {c['when']}" for cid, c in cands.items()
        }
        criteria["end_window"] = "End the current window now: nothing on the menu is still useful this window."
        criteria["replan"] = ("Ask the supervisor for a new menu: no candidate fits the current state, or the "
                              "last action failed in a way the menu does not cover.")
        state = (
            f"SUPERVISOR STRATEGY\n{self._plan['strategy']}\n\n"
            f"PREVIOUS CHOICE\n{self._last_choice or 'none'}\n\n"
            f"CURRENT STATE AND LAST ACTION RESULT\n{self._obs_text(observation)}"
        )
        self.stats["decision_calls"] += 1
        res = self.decisions.decide(
            {"next_action": {"type": "choice", "instructions": DECISION_INSTRUCTIONS, "criteria": criteria}},
            state,
            viewport_png_b64=observation.get("viewport_png_b64"),
        )
        ans = res["answers"]["next_action"]
        probs = ans.get("probabilities")
        return str(ans["value"]), float(ans.get("confidence") or 0.0), probs if isinstance(probs, dict) else {}

    def _live_action(self, observation: dict[str, Any], context: list[dict[str, Any]], ep: Episode) -> dict[str, Any]:
        window = ep.world.window
        if self._plan is None or self._plan_window != window:
            self._replans = 0
            self._last_choice = None
            self._supervise(observation, context, ep, f"Window {window} is starting. Plan this window.")
        while True:
            choice, conf, probs = self._decide(observation)
            playable = choice in self._plan["candidates"] or choice == "end_window"
            low = conf < self.min_confidence
            if playable and not low:
                break
            if self._replans >= self.max_replans_per_window:
                if not playable:  # re-plan budget spent: take the likeliest real action
                    real = {k: v for k, v in probs.items() if k in self._plan["candidates"]}
                    choice = max(real, key=lambda k: real[k]) if real else "end_window"
                break
            self._replans += 1
            self.stats["replans"] += 1
            self.stats["low_confidence"] += int(low)
            self._last_choice = None
            self._supervise(
                observation, context, ep,
                f"The decision model answered '{choice}' with confidence {conf:.2f}. Previous strategy: "
                f"{self._plan['strategy']} Write a new menu that fits the current state.",
            )
        self._last_choice = choice
        cand = self._plan["candidates"].get(choice)
        action = {"tool": cand["tool"], "args": dict(cand["args"])} if cand else {"tool": "end_window", "args": {}}
        self.trace.append({"tier": "decision", "window": window, "choice": choice, "confidence": round(conf, 4),
                           "tool": action["tool"], "args": action["args"]})
        return action

    # ---------------------------------------------------------------- dispatch
    def next_action(self, observation: dict[str, Any], context: list[dict[str, Any]], ep: Episode) -> dict[str, Any] | None:
        if ep.done:
            return None
        if self.live:
            return self._live_action(observation, context, ep)
        return self._scripted_action(observation, ep)

    def _scripted_action(self, observation: dict[str, Any], ep: Episode) -> dict[str, Any]:
        """Deterministic offline stand-in (no network). Not a Gemini result."""
        w = ep.world
        housing_valid = w.housing()["valid"]

        if self.use_decisions_api and not w.notes and ep.ops_left >= 5:
            state_desc = f"window={w.window} phase={w.phase} sheltered={housing_valid} tier={w.tech_tier} boss={bool(w.boss)}"
            decision = self.decisions.decide(
                {
                    "macro_priority": {
                        "type": "choice",
                        "instructions": "Select primary window objective",
                        "choices": ["build_shelter_and_wood", "craft_tier1_gear", "arena_or_boss_combat"],
                    }
                },
                state_desc,
                viewport_png_b64=observation.get("viewport_png_b64"),
            )
            ans = decision["answers"]["macro_priority"]
            return {
                "tool": "record_tactical_checkpoint",
                "args": {
                    "hypothesis": f"Gemini Decisions priority={ans['value']}",
                    "plan": state_desc,
                    "confidence": float(ans["confidence"]),
                },
            }

        if ep.ops_left <= 1 and not housing_valid and "SHELTER" in w.milestones and w.pos != w.home:
            return {"tool": "navigate_to", "args": {"x": w.home[0], "y": w.home[1], "budget": 300}}

        if not housing_valid and "SHELTER" not in w.milestones:
            action = self._scripted_shelter(ep)
            if action is not None:
                return action

        if not w.has("wooden_sword") and w.has("wood", 7) and w.near_tile(WORKBENCH, radius=4):
            return {"tool": "craft_item", "args": {"item_id": "wooden_sword", "quantity": 1}}
        if w.has("wooden_sword") and w.eq["weapon"] != "wooden_sword":
            return {"tool": "equip_item", "args": {"item_id": "wooden_sword"}}
        if not w.has("wood_armor") and w.has("wood", 30) and w.near_tile(WORKBENCH, radius=4):
            return {"tool": "craft_item", "args": {"item_id": "wood_armor", "quantity": 1}}
        if w.has("wood_armor") and w.eq["armor"] != "wood_armor":
            return {"tool": "equip_item", "args": {"item_id": "wood_armor"}}

        if "wood" not in w.inv or (w.inv.get("wood", 0) < 35 and w.tech_tier < 1 and any(tx not in w.withered for tx in w.trees)):
            return {"tool": "chop_trees", "args": {"radius": 45, "max_trees": 3}}

        if "arena" in ep.task.objectives and "ARENA" not in w.milestones and w.inv.get("wood", 0) >= 20:
            return {"tool": "build_blueprint", "args": {"preset": "arena"}}

        if w.boss is not None:
            return {
                "tool": "engage_combat_burst",
                "args": {"kite_distance": 16.0, "dodge_priority": 0.6, "vertical_platform_use": True},
            }

        if not housing_valid and "SHELTER" in w.milestones:
            return {"tool": "navigate_to", "args": {"x": w.home[0], "y": w.home[1], "budget": 300}}

        return {"tool": "end_window", "args": {}}

    @staticmethod
    def _scripted_shelter(ep: Episode) -> dict[str, Any] | None:
        """Build the synthesized cabin, paying for every tile.

        Matrix tiles are charged from inventory, so the door, torch and
        workbench are crafted first; the workbench is placed where the blueprint
        wants it (dx=-2) and reused in place. Needs 42 wood and 1 gel in total;
        without the materials it falls back to the ``shelter`` preset.
        """
        w = ep.world
        px, py = w.pos
        wb_placed = px - 2 >= 0 and w.g[py][px - 2] == WORKBENCH
        trees_left = any(tx not in w.withered for tx in w.trees)
        if not wb_placed and not w.has("workbench"):
            if w.inv.get("wood", 0) < 42 and trees_left:
                return {"tool": "chop_trees", "args": {"radius": 40, "max_trees": 4}}
            if w.has("wood", 42) and (w.has("gel", 1) or w.has("torch")):
                return {"tool": "craft_item", "args": {"item_id": "workbench", "quantity": 1}}
        elif not wb_placed:
            return {"tool": "place_station_or_furniture", "args": {"item_id": "workbench", "dx": -2, "dy": 0}}
        else:
            if not w.has("door") and w.has("wood", 6):
                return {"tool": "craft_item", "args": {"item_id": "door", "quantity": 1}}
            if not w.has("torch") and w.has("wood", 1) and w.has("gel", 1):
                return {"tool": "craft_item", "args": {"item_id": "torch", "quantity": 1}}
            if w.has("door") and w.has("torch") and w.has("wood", 25):
                return {
                    "tool": "build_blueprint",
                    "args": {"matrix": synthesize_blueprint_matrix("shelter"), "origin_dx": -4, "origin_dy": -4},
                }
        if w.inv.get("wood", 0) >= 16 or w.has("workbench"):
            return {"tool": "build_blueprint", "args": {"preset": "shelter"}}
        return None
