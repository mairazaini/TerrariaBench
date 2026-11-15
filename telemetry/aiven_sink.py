#!/usr/bin/env python3
"""
telemetry/aiven_sink.py
Off-box tamper-proof checkpoint settlement sink using Aiven PostgreSQL.
Streams HMAC-SHA256 signed checkpoints (run_id, agent_name, seed, score, housing_valid, hmac_sig).
Falls back to a secure local SQLite database (telemetry/local_aiven_audit.sqlite) when AIVEN_PG_URL is unset.
"""

import os
import sys
import json
import hmac
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

SECRET_KEY = os.environ.get("TERRARIA_HMAC_SECRET", "junction_2026_afterquery_secret_key").encode("utf-8")
LOCAL_DB_PATH = Path(__file__).resolve().parent / "local_aiven_audit.sqlite"

def compute_signature(payload: dict) -> str:
    """Computes HMAC-SHA256 signature over canonical JSON payload."""
    serialized = json.dumps(payload, sort_keys=True)
    return hmac.new(SECRET_KEY, serialized.encode("utf-8"), hashlib.sha256).hexdigest()

def verify_checkpoint(payload: dict, signature: str) -> bool:
    """Verifies HMAC signature of a checkpoint record."""
    expected = compute_signature(payload)
    if not hmac.compare_digest(expected, signature):
        raise ValueError(f"HMAC tamper detected! Expected {expected}, got {signature}")
    return True

class AivenSink:
    def __init__(self, pg_url: str = None):
        self.pg_url = pg_url or os.environ.get("AIVEN_PG_URL")
        self.use_postgres = False
        self.conn = None
        
        if self.pg_url:
            try:
                import psycopg2
                self.conn = psycopg2.connect(self.pg_url)
                self.use_postgres = True
                self._init_pg_table()
                print("[AivenSink] Connected to Aiven Managed PostgreSQL.")
            except Exception as e:
                print(f"[AivenSink] PostgreSQL connection failed ({e}). Falling back to local SQLite audit sink.")
                self.use_postgres = False

        if not self.use_postgres:
            self._init_sqlite()

    def _init_sqlite(self):
        LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_conn = sqlite3.connect(str(LOCAL_DB_PATH), check_same_thread=False)
        with self.sqlite_conn:
            self.sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    score REAL NOT NULL,
                    housing_valid INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_sig TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def _init_pg_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_checkpoints (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(128) NOT NULL,
                    agent_name VARCHAR(128) NOT NULL,
                    seed INT NOT NULL,
                    score FLOAT NOT NULL,
                    housing_valid BOOLEAN NOT NULL,
                    payload_json JSONB NOT NULL,
                    hmac_sig VARCHAR(64) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            self.conn.commit()

    def settle_checkpoint(self, run_id: str, agent_name: str, seed: int, score: float, housing_valid: bool, extra: dict = None) -> str:
        payload = {
            "run_id": run_id,
            "agent_name": agent_name,
            "seed": seed,
            "score": round(float(score), 2),
            "housing_valid": bool(housing_valid),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extra": extra or {}
        }
        sig = compute_signature(payload)

        if self.use_postgres:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO audit_checkpoints (run_id, agent_name, seed, score, housing_valid, payload_json, hmac_sig)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (run_id, agent_name, seed, score, housing_valid, json.dumps(payload), sig))
                    self.conn.commit()
                return sig
            except Exception as e:
                print(f"[AivenSink] Failed to insert to Postgres ({e}), falling back to SQLite.")

        with self.sqlite_conn:
            self.sqlite_conn.execute("""
                INSERT INTO audit_checkpoints (run_id, agent_name, seed, score, housing_valid, payload_json, hmac_sig)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (run_id, agent_name, seed, score, 1 if housing_valid else 0, json.dumps(payload), sig))
        return sig

    def get_recent_checkpoints(self, limit: int = 10) -> list:
        if self.use_postgres:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT run_id, agent_name, seed, score, housing_valid, hmac_sig, created_at 
                        FROM audit_checkpoints ORDER BY id DESC LIMIT %s;
                    """, (limit,))
                    rows = cur.fetchall()
                    return [{"run_id": r[0], "agent": r[1], "seed": r[2], "score": r[3], "valid": r[4], "sig": r[5], "time": str(r[6])} for r in rows]
            except Exception:
                pass

        with self.sqlite_conn:
            cur = self.sqlite_conn.cursor()
            cur.execute("""
                SELECT run_id, agent_name, seed, score, housing_valid, hmac_sig, created_at 
                FROM audit_checkpoints ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cur.fetchall()
            return [{"run_id": r[0], "agent": r[1], "seed": r[2], "score": r[3], "valid": bool(r[4]), "sig": r[5], "time": str(r[6])} for r in rows]

# Global singleton
_sink = None
def get_sink():
    global _sink
    if _sink is None:
        _sink = AivenSink()
    return _sink
