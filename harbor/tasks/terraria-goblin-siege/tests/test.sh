#!/bin/bash
# Verifier: finalize the episode on the game server and record its composite
# score (0..1). Windows the agent never reached are settled unattended, so
# quitting early is graded, not excused. Any failure here scores 0.

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

python3 - <<'PY'
import json
import urllib.request

req = urllib.request.Request(
    "http://game:8000/finalize", data=b"{}",
    headers={"X-Verifier-Token": "2a7a55560b4f50a83f297fb2ab53c4d0", "Content-Type": "application/json"})
result = json.loads(urllib.request.urlopen(req, timeout=120).read())

reward = min(1.0, max(0.0, float(result["reward"])))
with open("/logs/verifier/grade.json", "w") as fh:
    json.dump({"reward": reward, "grade": result["grade"], "tool_calls": result["actions"]}, fh, indent=1)
with open("/logs/verifier/ledger.json", "w") as fh:
    json.dump(result["ledger"], fh, indent=1, sort_keys=True)
with open("/logs/verifier/reward.txt", "w") as fh:
    fh.write(f"{reward:.4f}\n")
print(json.dumps({"reward": reward, **result["grade"]}, indent=1))
PY
