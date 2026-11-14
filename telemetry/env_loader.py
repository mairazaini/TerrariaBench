"""
telemetry/env_loader.py - Pure Python .env Loader

Loads key-value pairs from .env into os.environ without requiring external dependencies.
"""

import os
from pathlib import Path


def load_env(env_path: str = ".env"):
    p = Path(env_path)
    if not p.is_file():
        # Check parent directory as fallback
        p = Path(__file__).resolve().parent.parent / ".env"
    
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k:
                    os.environ[k] = v
