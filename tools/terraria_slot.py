"""Bridge and execution slot for Terraria headless & visual environment.

Provides coordinate-gridded frames, checkpoint log streaming, and deterministic
low-level / macro action execution.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from terraria_bench.engine import World
from terraria_bench.tasks import TASKS, Task
from terraria_bench.tools import Episode


class TerrariaSlot:
    """Manages an active Terraria execution slot, viewport frames, and action dispatch."""

    def __init__(self, task_id: str = "woodland-scramble", seed: int = 7, run_dir: Path | str | None = None):
        self.task_id = task_id
        self.seed = seed
        self.run_dir = Path(run_dir) if run_dir else Path(f"artifacts/runs/{task_id}_seed{seed}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = self.run_dir / "checkpoints.jsonl"
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.ep = Episode(task_id, seed)
        self.step_idx = 0
        self._record_checkpoint(initial=True)

    def _record_checkpoint(self, initial: bool = False, action_info: dict[str, Any] | None = None) -> dict[str, Any]:
        w = self.ep.world
        px, py = w.pos if hasattr(w, "pos") and isinstance(w.pos, (list, tuple)) else (100, 25)
        wood_val = w.inv.get("wood", 0) if hasattr(w, "inv") else 0
        obsidian_val = (w.inv.get("obsidian", 0) > 0) if hasattr(w, "inv") else False
        housing_val = (wood_val >= 10) or bool(getattr(w, "housing", False))

        record = {
            "step": self.step_idx,
            "timestamp": time.time(),
            "task_id": self.task_id,
            "seed": self.seed,
            "player_x": px,
            "player_y": py,
            "hp": getattr(w, "hp", 100),
            "mana": getattr(w, "mana", 20),
            "net_wood": wood_val,
            "obsidian_bridged": obsidian_val,
            "housing_valid": housing_val,
            "reward_score": float(w.phi()) if hasattr(w, "phi") else 0.0,
            "action": action_info or ("init" if initial else "step"),
            "done": self.ep.done,
        }
        with open(self.checkpoints_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def get_latest_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoints_path.exists():
            return self._record_checkpoint(initial=True)
        last_line = ""
        with open(self.checkpoints_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        return json.loads(last_line) if last_line else self._record_checkpoint(initial=True)

    def get_frame(self, include_grid: bool = True) -> dict[str, Any]:
        """Returns the current viewport frame with optional coordinate grid overlay."""
        viewport = self.ep.step("observe_viewport")
        raw_text = viewport.get("viewport", "")
        grid_lines = []
        if include_grid:
            grid_lines.append("+----[ X: 0..60 | Y: 0..34 ]----+")
            for idx, line in enumerate(raw_text.splitlines()[:35]):
                grid_lines.append(f"{idx:02d}| {line}")
            annotated_frame = "\n".join(grid_lines)
        else:
            annotated_frame = raw_text

        frame_file = self.frames_dir / f"frame_{self.step_idx:04d}.png"
        if not frame_file.exists():
            frame_file.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
            )

        return {
            "step": self.step_idx,
            "header": self.ep.header(),
            "viewport_ascii": annotated_frame,
            "frame_path": str(frame_file),
            "state": self.get_latest_checkpoint(),
        }

    def execute_action(self, action_type: str, x: int = 0, y: int = 0, key: str = "", duration_ms: int = 100) -> dict[str, Any]:
        """Dispatches deterministic actions into the Terraria environment."""
        self.step_idx += 1
        res: dict[str, Any] = {"ok": True, "action_type": action_type}
        action_desc = {"type": action_type, "x": x, "y": y, "key": key, "duration_ms": duration_ms}

        if action_type == "click":
            if y < 15:
                res = self.ep.step("chop_trees")
            else:
                res = self.ep.step("mine_vein", ore="copper")
        elif action_type == "move":
            px = self.ep.world.pos[0] if hasattr(self.ep.world, "pos") else 100
            target_x = max(0, min(self.ep.world.w - 1 if hasattr(self.ep.world, "w") else 200, x if x != 0 else px + 2))
            res = self.ep.step("navigate_to", x=target_x)
        elif action_type == "key":
            if key in ("e", "i"):
                res = self.ep.step("inspect_inventory")
            elif key == "c":
                res = self.ep.step("craft_item", item="work_bench")
            elif key == "space":
                res = self.ep.step("retreat_and_stabilize")
            else:
                res = self.ep.step("terraria_status")
        elif action_type in ("mouse_down", "batch"):
            res = self.ep.step("chop_trees")
        else:
            res = self.ep.step("terraria_status")

        checkpoint = self._record_checkpoint(action_info=action_desc)
        return {
            "result": res,
            "checkpoint": checkpoint,
            "step": self.step_idx,
            "done": self.ep.done,
            "header": self.ep.header(),
        }
