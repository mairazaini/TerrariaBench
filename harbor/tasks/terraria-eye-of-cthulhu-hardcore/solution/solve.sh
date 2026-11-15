#!/bin/bash
# Reference solution: replay the tool calls of the scripted `smart-triage`
# anchor policy. The world is deterministic, so the replay reproduces its run.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

python3 - "$DIR/actions.jsonl" <<'PY'
import json
import subprocess
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    step = json.loads(line)
    out = subprocess.run(["terraria", step["tool"], json.dumps(step["args"])], capture_output=True, text=True)
    print(out.stdout.splitlines()[0] if out.stdout else out.stderr.strip())
PY
