"""Determinism and provenance (StoreBench Appendix G).

* purpose-keyed random streams: every stochastic draw names its purpose, so
  adding a draw in one subsystem never shifts another subsystem's stream;
* canonical SHA-256 hashing of JSON artifacts and source files;
* the run manifest (source commit, image digest, seed-data and calibration
  hashes) and the fail-closed loaders used by the engine and the verifier.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
ARTIFACTS_DIR = Path(os.environ.get("TERRARIA_BENCH_ARTIFACTS", REPO_DIR / "artifacts"))
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.json"
SEED_DATA_PATH = ARTIFACTS_DIR / "seed_data.json"

# Files whose content defines the benchmark. A change to any of them
# invalidates the calibration artifact.
BENCHMARK_SOURCES = ("tasks.py", "engine.py", "tools.py", "grader.py", "policies.py", "provenance.py")


class ProvenanceError(RuntimeError):
    """Raised when an artifact fails verification. Callers fail closed."""


def stream(seed: int, *purpose) -> random.Random:
    """A random stream keyed by (seed, purpose...). Mersenne Twister seeded from
    SHA-256, so it is stable across platforms and Python versions."""
    key = "|".join([str(int(seed)), *map(str, purpose)]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical(obj))


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def source_hash() -> str:
    h = hashlib.sha256()
    for name in BENCHMARK_SOURCES:
        h.update(name.encode())
        p = PACKAGE_DIR / name
        h.update((sha256_file(p) if p.exists() else "missing").encode())
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def seal(payload: dict) -> dict:
    """Return payload with a ``sha256`` field covering everything else."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return {**body, "sha256": sha256_json(body)}


def verify_seal(payload: dict) -> bool:
    return isinstance(payload, dict) and payload.get("sha256") == sha256_json(
        {k: v for k, v in payload.items() if k != "sha256"})


def write_sealed(path, payload: dict) -> dict:
    sealed = seal(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return sealed


def load_sealed(path) -> dict:
    """Load a sealed artifact; missing or corrupt artifacts raise (fail closed)."""
    path = Path(path)
    if not path.exists():
        raise ProvenanceError(f"missing artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"corrupt artifact: {path}: {exc}") from exc
    if not verify_seal(payload):
        raise ProvenanceError(f"artifact hash mismatch: {path}")
    return payload


def load_calibration(path=None, check_source: bool = True) -> dict:
    cal = load_sealed(path or CALIBRATION_PATH)
    if check_source and cal.get("source_hash") != source_hash():
        raise ProvenanceError("calibration is stale: benchmark source changed since it was computed; "
                              "run `python -m terraria_bench.calibrate`")
    return cal


def pinned_seed_hash(seed: int):
    """The pinned world hash for a seed, or None when no pin file exists yet."""
    if not SEED_DATA_PATH.exists():
        return None
    try:
        pins = load_sealed(SEED_DATA_PATH)
    except ProvenanceError:
        return "corrupt"
    if pins.get("source_hash") != source_hash():
        return None  # pins predate a source change; calibrate rewrites them
    return pins.get("seeds", {}).get(str(seed))


def run_manifest(calibration: dict | None = None) -> dict:
    return {
        "source_commit": git_commit(),
        "source_hash": source_hash(),
        "image_digest": os.environ.get("TERRARIA_BENCH_IMAGE_DIGEST", "unpinned-local-run"),
        "calibration_sha256": (calibration or {}).get("sha256", "none"),
        "seed_data_sha256": sha256_file(SEED_DATA_PATH) if SEED_DATA_PATH.exists() else "none",
        "python": platform.python_version(),
    }


def verify_trial(trial: dict, calibration: dict) -> bool:
    """A trial is gradable only if it ran against this exact calibration and source."""
    m = trial.get("manifest", {})
    return m.get("calibration_sha256") == calibration.get("sha256") and m.get("source_hash") == calibration.get("source_hash")
