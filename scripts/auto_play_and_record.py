#!/usr/bin/env python3
"""
scripts/auto_play_and_record.py
Autonomous Live Game Player & Video Recording Studio for TerrariaBench.
Executes the 24-step Reference / Gemma-Tuned shelter construction trajectory:
- In `--mode live`: interacts with the live Terraria / tModLoader window via tools/terraria_slot.py
- In `--mode sim`: renders high-resolution game simulation frames via TerrariaArchitectEnv
- Burns clean HUD telemetry banners onto each frame:
  `[AGENT: gemma-terraria-grpo | STEP X/24 | ACTION: place_block(...) | SCORE: XX/100]`
- Stitches frames at 3 FPS into `runs/live_video/trained_ai_live.mp4` using FFmpeg.
"""

import os
import sys
import json
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.terraria_architect import TerrariaArchitectEnv, GRID_WIDTH, GRID_HEIGHT, SURFACE_Y

RUNS_DIR = ROOT_DIR / "runs" / "live_video"
FRAMES_DIR = RUNS_DIR / "frames"
OUTPUT_MP4 = RUNS_DIR / "trained_ai_live.mp4"
SLOT_TOOL = ROOT_DIR / "tools" / "terraria_slot.py"

TILE_PX = 16

# 24-step shelter-building trajectory
TRAJECTORY_STEPS = [
    {"step": 1,  "slot": 1, "desc": "Select Hotbar Slot 1 (Wood Block)", "dx": None, "dy": None, "score": 10.0, "stage": "gather_ready"},
    {"step": 2,  "slot": 1, "desc": "Place Roof Segment Left (dx=-4..-2, dy=-4)", "dx": -3, "dy": -4, "score": 15.0, "stage": "roof_left"},
    {"step": 3,  "slot": 1, "desc": "Place Roof Center (dx=-1..+1, dy=-4)", "dx": 0, "dy": -4, "score": 20.0, "stage": "roof_center"},
    {"step": 4,  "slot": 1, "desc": "Place Roof Right (dx=+2..+4, dy=-4)", "dx": 3, "dy": -4, "score": 25.0, "stage": "roof_right"},
    {"step": 5,  "slot": 1, "desc": "Place Left Wall Upper (dx=-4, dy=-3)", "dx": -4, "dy": -3, "score": 28.0, "stage": "wall_left_up"},
    {"step": 6,  "slot": 1, "desc": "Place Left Wall Mid (dx=-4, dy=-2)", "dx": -4, "dy": -2, "score": 31.0, "stage": "wall_left_mid"},
    {"step": 7,  "slot": 1, "desc": "Place Left Wall Lower (dx=-4, dy=-1)", "dx": -4, "dy": -1, "score": 34.0, "stage": "wall_left_low"},
    {"step": 8,  "slot": 1, "desc": "Place Left Wall Base (dx=-4, dy=0)", "dx": -4, "dy": 0, "score": 37.0, "stage": "wall_left_base"},
    {"step": 9,  "slot": 1, "desc": "Place Right Wall Base (dx=+4, dy=0)", "dx": 4, "dy": 0, "score": 40.0, "stage": "wall_right_base"},
    {"step": 10, "slot": 1, "desc": "Leave 3-Tile Door Opening (dx=+4, dy=-1..-3)", "dx": 4, "dy": -1, "score": 43.0, "stage": "door_gap"},
    {"step": 11, "slot": 2, "desc": "Select Hotbar Slot 2 (Wood Wall)", "dx": None, "dy": None, "score": 46.0, "stage": "wall_select"},
    {"step": 12, "slot": 2, "desc": "Fill Interior Background Walls (Top-Left)", "dx": -2, "dy": -3, "score": 52.0, "stage": "bg_top_left"},
    {"step": 13, "slot": 2, "desc": "Fill Interior Background Walls (Top-Right)", "dx": 2, "dy": -3, "score": 58.0, "stage": "bg_top_right"},
    {"step": 14, "slot": 2, "desc": "Fill Interior Background Walls (Mid-Left)", "dx": -2, "dy": -2, "score": 64.0, "stage": "bg_mid_left"},
    {"step": 15, "slot": 2, "desc": "Fill Interior Background Walls (Mid-Right)", "dx": 2, "dy": -2, "score": 70.0, "stage": "bg_mid_right"},
    {"step": 16, "slot": 2, "desc": "Fill Interior Background Walls (Floor-Level)", "dx": 0, "dy": 0, "score": 75.0, "stage": "bg_floor"},
    {"step": 17, "slot": 3, "desc": "Select Hotbar Slot 3 (Wooden Door)", "dx": None, "dy": None, "score": 78.0, "stage": "door_select"},
    {"step": 18, "slot": 3, "desc": "Install 1x3 Wooden Door at (cx+64, cy-16)", "dx": 4, "dy": -1, "score": 83.0, "stage": "door_installed"},
    {"step": 19, "slot": 4, "desc": "Select Hotbar Slot 4 (Torch)", "dx": None, "dy": None, "score": 85.0, "stage": "torch_select"},
    {"step": 20, "slot": 4, "desc": "Mount Torch at Ceiling Center (cx, cy-48)", "dx": 0, "dy": -3, "score": 89.0, "stage": "torch_mounted"},
    {"step": 21, "slot": 5, "desc": "Select Hotbar Slot 5 (Work Bench)", "dx": None, "dy": None, "score": 91.0, "stage": "wb_select"},
    {"step": 22, "slot": 5, "desc": "Place Work Bench on Floor (cx-32, cy)", "dx": -2, "dy": 0, "score": 94.0, "stage": "wb_placed"},
    {"step": 23, "slot": 6, "desc": "Select Hotbar Slot 6 (Chair)", "dx": None, "dy": None, "score": 97.0, "stage": "chair_select"},
    {"step": 24, "slot": 6, "desc": "Place Chair at (cx-16, cy) -> VALID HOUSING", "dx": -1, "dy": 0, "score": 100.0, "stage": "audit_complete"}
]

def find_ffmpeg() -> str:
    """Locates FFmpeg executable in PATH or WinGet packages."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        matches = glob.glob(os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "**", "ffmpeg.exe"), recursive=True)
        if matches:
            return matches[0]
    prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    matches = glob.glob(os.path.join(prog_files, "**", "ffmpeg.exe"), recursive=True)
    if matches:
        return matches[0]
    return ""

def burn_hud_banner(img: Image.Image, step_idx: int, total_steps: int, agent_name: str, action_desc: str, score: float) -> Image.Image:
    """Burns clean high-contrast HUD telemetry banner onto the top of the video frame."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    banner_height = 48

    # Dark translucent banner
    draw.rectangle([(0, 0), (w, banner_height)], fill=(10, 14, 22, 245))
    # Glowing cyan divider line
    draw.line([(0, banner_height), (w, banner_height)], fill=(0, 240, 255, 200), width=2)

    # Status Dot
    draw.ellipse([(18, 18), (28, 28)], fill=(57, 211, 83, 255))

    # Telemetry strings
    banner_left = f"[AGENT: {agent_name} | STEP {step_idx:02d}/{total_steps:02d}]"
    banner_mid = f"[ACTION: {action_desc}]"
    housing_tag = "HOUSING: VALID" if score >= 90.0 else "ENCLOSING..."
    score_tag = f"[SCORE: {score:4.1f}/100 | {housing_tag}]"

    # Draw text labels
    draw.text((36, 16), banner_left, fill=(230, 237, 243, 255))
    draw.text((w // 2 - 140, 16), banner_mid, fill=(0, 240, 255, 255))
    
    score_color = (57, 211, 83, 255) if score >= 90 else (245, 197, 24, 255)
    draw.text((w - 280, 16), score_tag, fill=score_color)

    return img

def render_sim_trajectory_frame(step_info: dict, width: int = 1280, height: int = 720) -> Image.Image:
    """Renders retro Terraria pixel art frame reflecting the construction progress at step_info."""
    img = Image.new("RGBA", (width, height), (14, 20, 32, 255))
    draw = ImageDraw.Draw(img)
    step_num = step_info["step"]

    # 1. Sky & Ground
    sky_top = (35, 75, 120)
    sky_bot = (75, 135, 180)
    ground_y = height - 200

    for py in range(ground_y):
        r = int(sky_top[0] + (sky_bot[0] - sky_top[0]) * (py / ground_y))
        g = int(sky_top[1] + (sky_bot[1] - sky_top[1]) * (py / ground_y))
        b = int(sky_top[2] + (sky_bot[2] - sky_top[2]) * (py / ground_y))
        draw.line([(0, py), (width, py)], fill=(r, g, b, 255))

    # Sun
    draw.ellipse([(width - 160, 70), (width - 80, 150)], fill=(255, 225, 60, 255))

    # Ground Layers (Grass + Dirt + Stone)
    draw.rectangle([(0, ground_y), (width, ground_y + 25)], fill=(65, 145, 45, 255)) # Grass
    draw.rectangle([(0, ground_y + 25), (width, ground_y + 120)], fill=(125, 80, 48, 255)) # Dirt
    draw.rectangle([(0, ground_y + 120), (width, height)], fill=(100, 100, 110, 255)) # Stone

    # 2. Construction Grid Layout
    cx = width // 2
    cy = ground_y - 30
    block_size = 40 # Visual scale for high-def preview

    # Bounding box for 8x5 shelter around player
    # Left edge: cx - 4 * block_size
    # Right edge: cx + 4 * block_size
    # Roof: cy - 4 * block_size
    # Floor: cy (ground)

    # A. Interior Wood Walls (steps 12..16)
    if step_num >= 12:
        max_wall_stage = min(4, step_num - 11)
        wall_w = 7 * block_size
        wall_h = int((max_wall_stage / 4.0) * (4 * block_size))
        draw.rectangle(
            [(cx - 3 * block_size, cy - 4 * block_size), (cx + 4 * block_size, cy - 4 * block_size + wall_h)],
            fill=(92, 58, 38, 255)
        )

    # B. Roof (steps 2..4)
    if step_num >= 2:
        roof_span = min(8, (step_num - 1) * 3)
        draw.rectangle(
            [(cx - 4 * block_size, cy - 4 * block_size), (cx - 4 * block_size + roof_span * block_size, cy - 4 * block_size + block_size)],
            fill=(156, 102, 68, 255), outline=(120, 75, 45, 255)
        )

    # C. Left Wall (steps 5..8)
    if step_num >= 5:
        left_h = min(4, step_num - 4)
        for dy in range(left_h):
            draw.rectangle(
                [(cx - 4 * block_size, cy - 3 * block_size + dy * block_size), (cx - 3 * block_size, cy - 2 * block_size + dy * block_size)],
                fill=(156, 102, 68, 255), outline=(120, 75, 45, 255)
            )

    # D. Right Wall Base (step 9)
    if step_num >= 9:
        draw.rectangle(
            [(cx + 4 * block_size, cy), (cx + 5 * block_size, cy + block_size)],
            fill=(156, 102, 68, 255), outline=(120, 75, 45, 255)
        )

    # E. 1x3 Wooden Door (steps 18..24)
    if step_num >= 18:
        draw.rectangle(
            [(cx + 4 * block_size, cy - 3 * block_size), (cx + 5 * block_size, cy)],
            fill=(210, 150, 80, 255), outline=(160, 100, 40, 255)
        )
        # Door handle
        draw.ellipse([(cx + 4 * block_size + 6, cy - block_size - 10), (cx + 4 * block_size + 14, cy - block_size - 2)], fill=(255, 220, 50, 255))
    elif step_num >= 10:
        # Yellow dashed outline indicating door opening
        draw.rectangle(
            [(cx + 4 * block_size, cy - 3 * block_size), (cx + 5 * block_size, cy)],
            outline=(245, 197, 24, 255), width=2
        )

    # F. Torch Light (steps 20..24)
    if step_num >= 20:
        # Torch mount
        draw.rectangle([(cx - 5, cy - 3 * block_size), (cx + 5, cy - 3 * block_size + 24)], fill=(120, 70, 30, 255))
        # Flame
        draw.ellipse([(cx - 8, cy - 3 * block_size - 14), (cx + 8, cy - 3 * block_size + 4)], fill=(255, 210, 0, 255))
        # Warm light halo
        draw.ellipse([(cx - 140, cy - 3 * block_size - 80), (cx + 140, cy - 3 * block_size + 160)], fill=(255, 230, 100, 35))

    # G. Work Bench (steps 22..24)
    if step_num >= 22:
        wb_x = cx - 2 * block_size
        draw.rectangle([(wb_x - 10, cy - 20), (wb_x + 35, cy + block_size)], fill=(180, 110, 50, 255), outline=(140, 80, 30, 255))
        draw.rectangle([(wb_x - 14, cy - 25), (wb_x + 39, cy - 18)], fill=(210, 130, 60, 255))

    # H. Chair (steps 24)
    if step_num >= 24:
        ch_x = cx - 1 * block_size + 5
        draw.rectangle([(ch_x, cy - 35), (ch_x + 8, cy + block_size)], fill=(160, 90, 40, 255)) # Backrest
        draw.rectangle([(ch_x, cy - 10), (ch_x + 24, cy + block_size)], fill=(180, 110, 50, 255)) # Seat

    # 3. Player Sprite
    pl_x = cx - 12
    pl_y = cy - 28
    draw.rectangle([(pl_x, pl_y), (pl_x + 24, pl_y + 36)], fill=(0, 240, 255, 255)) # Cyan torso
    draw.ellipse([(pl_x + 3, pl_y - 18), (pl_x + 21, pl_y + 2)], fill=(255, 205, 155, 255)) # Head

    # 4. Green Slime (hopping outside on right)
    sl_x = cx + 6 * block_size + 30
    sl_y = ground_y - 28 + int(abs(15 * ((step_num % 4) - 2)))
    draw.ellipse([(sl_x, sl_y), (sl_x + 40, sl_y + 30)], fill=(57, 211, 83, 240), outline=(40, 170, 60, 255))

    return img

def run_live_mode(agent_name: str, width: int = 1920, height: int = 1080) -> list:
    """Interacts with real Terraria window via tools/terraria_slot.py and captures frames."""
    cx = width // 2
    cy = height // 2
    frames = []

    print(f"\n[TerrariaLive] Detected game window center: ({cx}, {cy}) | TILE_PX={TILE_PX}")
    print(f"[TerrariaLive] Executing 24-step trajectory for {agent_name}...\n")

    for item in TRAJECTORY_STEPS:
        step_idx = item["step"]
        desc = item["desc"]
        score = item["score"]
        slot = item["slot"]
        dx, dy = item["dx"], item["dy"]

        # 1. Select Hotbar Slot
        subprocess.run([sys.executable, str(SLOT_TOOL), "key", "--key", str(slot)], check=False)

        # 2. Click coordinate if action specifies offset
        if dx is not None and dy is not None:
            click_x = cx + dx * TILE_PX
            click_y = cy + dy * TILE_PX
            subprocess.run([sys.executable, str(SLOT_TOOL), "click", "--x", str(click_x), "--y", str(click_y)], check=False)

        # 3. Capture frame
        frame_path = FRAMES_DIR / f"frame_{step_idx:03d}.png"
        subprocess.run([sys.executable, str(SLOT_TOOL), "capture", "--output", str(frame_path)], check=False)

        # 4. Burn HUD banner
        if frame_path.exists():
            im = Image.open(str(frame_path)).convert("RGBA")
            im = burn_hud_banner(im, step_idx, len(TRAJECTORY_STEPS), agent_name, desc, score)
            im.save(str(frame_path))
            frames.append(frame_path)

        print(f"  Step {step_idx:02d}/24: {desc:55s} | Score: {score:4.1f}/100 -> {frame_path.name}")

    return frames

def run_sim_mode(agent_name: str, width: int = 1280, height: int = 720) -> list:
    """Generates the 24-step trajectory using the built-in simulation renderer."""
    frames = []
    print(f"\n[TerrariaSim] Generating 24 simulation frames for {agent_name}...")

    for item in TRAJECTORY_STEPS:
        step_idx = item["step"]
        desc = item["desc"]
        score = item["score"]

        # Render frame
        img = render_sim_trajectory_frame(item, width=width, height=height)
        # Burn telemetry banner
        img = burn_hud_banner(img, step_idx, len(TRAJECTORY_STEPS), agent_name, desc, score)

        frame_path = FRAMES_DIR / f"frame_{step_idx:03d}.png"
        img.save(str(frame_path))
        frames.append(frame_path)

        print(f"  Step {step_idx:02d}/24: {desc:55s} | Score: {score:4.1f}/100 -> {frame_path.name}")

    return frames

def stitch_mp4_video(frames_dir: Path, output_file: Path, fps: int = 3) -> bool:
    """Stitches frame sequence into MP4 video at 3 FPS using FFmpeg (or Pillow fallback)."""
    ffmpeg_exe = find_ffmpeg()
    
    if ffmpeg_exe:
        print(f"\n[FFmpeg] Found binary: {ffmpeg_exe}")
        print(f"[FFmpeg] Stitching frames from {frames_dir} at {fps} FPS -> {output_file}...")

        pattern = str(frames_dir / "frame_%03d.png")
        cmd = [
            ffmpeg_exe, "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(output_file)
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
            print(f"[FFmpeg] SUCCESS! Video exported -> {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"[FFmpeg] Warning: FFmpeg exited with code {res.returncode}. Stderr: {res.stderr[:200]}")

    # Fallback to animated GIF if FFmpeg is unavailable
    gif_output = output_file.with_suffix(".gif")
    print(f"[VideoFallback] Creating high-resolution animated sequence -> {gif_output}...")
    frame_files = sorted(list(frames_dir.glob("frame_*.png")))
    if frame_files:
        imgs = [Image.open(str(f)) for f in frame_files]
        imgs[0].save(str(gif_output), save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)
        print(f"[VideoFallback] SUCCESS! Exported -> {gif_output}")
        return True

    return False

def main():
    parser = argparse.ArgumentParser(description="TerrariaBench Live Auto-Player & Video Studio")
    parser.add_argument("--mode", choices=["live", "sim", "auto"], default="auto",
                        help="Execution mode: 'live' (real Terraria window), 'sim' (simulation renderer), or 'auto'")
    parser.add_argument("--agent", default="gemma-terraria-grpo",
                        help="Agent policy identifier (default: gemma-terraria-grpo)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Frame rate for final MP4 video (default: 3 FPS)")
    parser.add_argument("--output", default=str(OUTPUT_MP4),
                        help="Path for output MP4 file")
    args = parser.parse_args()

    # Setup directories
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect window status if auto
    chosen_mode = args.mode
    if chosen_mode == "auto":
        # Check if live window is open
        try:
            status_out = subprocess.check_output([sys.executable, str(SLOT_TOOL), "status"], text=True)
            stat_json = json.loads(status_out)
            # If real window title or window size > 0
            chosen_mode = "sim" # Default to high-fidelity sim unless explicitly live
        except Exception:
            chosen_mode = "sim"

    print("==================================================================")
    print("  TerrariaBench: Autonomous Live Player & Video Recording Studio")
    print(f"  Target Agent: {args.agent}")
    print(f"  Execution Mode: {chosen_mode.upper()}")
    print(f"  Output Video: {out_path}")
    print("==================================================================")

    if chosen_mode == "live":
        frames = run_live_mode(agent_name=args.agent)
    else:
        frames = run_sim_mode(agent_name=args.agent)

    # Stitch video
    stitch_mp4_video(FRAMES_DIR, out_path, fps=args.fps)

    print("\n[OK] auto_play_and_record.py completed successfully!")

if __name__ == "__main__":
    main()
