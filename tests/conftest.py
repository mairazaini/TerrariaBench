"""Shared test fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _keep_committed_recordings(tmp_path, monkeypatch):
    """Point the live-play server at a scratch folder.

    Posting a tick to ``/real_game_tick`` rewrites ``real_terraria_live.json``. The redirect guarantees
    that no test, present or future, can overwrite the committed real-game recordings, which are evidence.
    """
    from terraria_bench import live_play

    monkeypatch.setattr(live_play, "RECORDINGS_DIR", tmp_path / "recordings")
