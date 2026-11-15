#!/usr/bin/env python3
"""
cloud/publish_demo.py
Non-invasive, standalone demo publisher using Modal (@app.function + @modal.asgi_app)
Serves docs/index.html and static benchmark run frames at a public https://...modal.run URL
for the Junction submission without modifying any core TerrariaBench files.

Usage:
  Deploy to public URL:
    modal deploy cloud/publish_demo.py

  Ephemeral dev test:
    modal serve cloud/publish_demo.py

  Local preview (no Modal token required):
    python cloud/publish_demo.py --local [port]
"""

import sys
import os
from pathlib import Path

# Paths
CLOUD_DIR = Path(__file__).resolve().parent
ROOT_DIR = CLOUD_DIR.parent
DOCS_DIR = ROOT_DIR / "docs"
INDEX_FILE = DOCS_DIR / "index.html"

# Try importing modal; if unavailable or in --local mode, fallback cleanly
try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    # Modal App Definition
    app = modal.App("terrariabench-demo")

    # Remote debian image with web dependencies and local docs mounted
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("fastapi", "uvicorn", "pydantic")
        .add_local_dir(str(DOCS_DIR), remote_path="/root/docs")
    )

    @app.function(
        image=image,
        timeout=600,
        scaledown_window=120,
    )
    @modal.asgi_app()
    def serve():
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware

        web_app = FastAPI(
            title="TerrariaBench Demo Publisher",
            description="Junction Hackathon Live Demo Dashboard & AfterQuery Rubric",
            version="1.0.0",
            docs_url=None,
            redoc_url=None,
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        remote_docs = Path("/root/docs")
        index_path = remote_docs / "index.html"
        if not index_path.exists():
            index_path = INDEX_FILE

        @web_app.get("/", response_class=HTMLResponse)
        async def root():
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
            return HTMLResponse(
                content="<h1>TerrariaBench docs/index.html not found</h1>",
                status_code=404
            )

        @web_app.get("/healthz")
        async def health():
            return {
                "status": "healthy",
                "service": "terrariabench-demo-publisher",
                "benchmark": "tModLoader 1.4+ / Harbor",
                "fair_controller": "active"
            }

        @web_app.get("/api/rubric")
        async def get_rubric():
            return {
                "benchmark": "TerrariaBench",
                "framework": "Harbor + AfterQuery Rubric",
                "tiers": [
                    {"tier": 1, "name": "Basic Gathering & Shelter", "tasks": 25, "gap": "+64.8%", "failure_mode": "Unsealed shelter ceiling gap"},
                    {"tier": 2, "name": "Mining & Metallurgy", "tasks": 25, "gap": "+88.0%", "failure_mode": "Cavern disorientation & station prereqs"},
                    {"tier": 3, "name": "Exploration & Stat Upgrades", "tasks": 20, "gap": "+90.0%", "failure_mode": "Boulder/Dart traps & fall damage"},
                    {"tier": 4, "name": "Preparation & Summoning", "tasks": 15, "gap": "+86.6%", "failure_mode": "Nocturnal Demon Eye farming deaths"},
                    {"tier": 5, "name": "Grand Challenge (Eye of Cthulhu)", "tasks": 15, "gap": "+74.0%", "failure_mode": "Phase 2 kinetic dodging latency"}
                ],
                "reward_integrity": {
                    "source": "checkpoints.jsonl",
                    "verifier": "C# tModLoader Bridge (NPC.downedBoss1 & tile diffs)",
                    "anti_cheat": "direct_tile_control_used = true flagged & rejected"
                }
            }

        # Mount static files if present
        if remote_docs.exists():
            web_app.mount("/static", StaticFiles(directory=str(remote_docs)), name="static")

        return web_app


def run_local(port: int = 8000):
    """Fallback local server using Python's built-in http.server."""
    import http.server
    import socketserver

    os.chdir(str(DOCS_DIR))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            super().end_headers()

    print(f"\n========================================================")
    print(f"  TerrariaBench Demo Publisher (Local Preview)")
    print(f"  Serving docs/index.html at:")
    print(f"  -> http://localhost:{port}")
    print(f"========================================================\n")

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down local server.")


if __name__ == "__main__":
    if "--local" in sys.argv or not HAS_MODAL:
        port = 8000
        for arg in sys.argv:
            if arg.isdigit():
                port = int(arg)
        run_local(port)
    else:
        print("[INFO] Modal app configured. Run with 'modal deploy cloud/publish_demo.py'")
