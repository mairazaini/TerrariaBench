#!/usr/bin/env python3
"""
make.py - Cross-platform runner for Makefile targets on Windows and Linux.
Usage:
  python make.py eval
  python make.py test-anti-cheat
  python make.py serve-local-demo
"""

import sys
import subprocess

TARGETS = {
    "eval": [sys.executable, "reporting/generate_eval_report.py"],
    "test-anti-cheat": [sys.executable, "-m", "pytest", "tests/test_reward_integrity.py", "-v"],
    "serve-local-demo": [sys.executable, "cloud/modal_app.py", "--local", "8080"],
    "deploy-modal": [sys.executable, "-m", "modal", "deploy", "cloud/modal_app.py"],
    "train-grpo": [sys.executable, "-m", "modal", "run", "cloud/modal_app.py::train_grpo"],
}

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if target not in TARGETS:
        print(f"Unknown target '{target}'. Available targets: {list(TARGETS.keys())}")
        sys.exit(1)

    cmd = TARGETS[target]
    print(f"[make.py] Running: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    sys.exit(ret)

if __name__ == "__main__":
    main()
