"""The Harbor two-tier agent's sidecar adapter, driven against an in-process Episode."""
import json

from terraria_bench.harness.gemini_agent import GeminiAgent, GeminiDecisionsWrapper
from terraria_bench.harness.harbor_agent import RemoteEpisode, parse_reply, play
from terraria_bench.tools import Episode


def _sidecar(ep: Episode):
    """Mimic the game server's POST /call reply shape."""
    def call(tool, args):
        res = ep.step(tool, **args)
        body = {k: v for k, v in res.items() if k not in ("header", "text", "viewport_png_b64")}
        return {"ok": bool(res.get("ok", True)), "text": f"{res['header']}\n{json.dumps(body, default=str)}",
                "done": ep.done}
    return call


def test_parse_reply_splits_header_and_body():
    obs = parse_reply({"ok": False, "text": "[Win: 1/4 | Ops: 3/20]\n{\"error\": \"x\"}"})
    assert obs == {"error": "x", "header": "[Win: 1/4 | Ops: 3/20]", "ok": False}
    assert parse_reply({"ok": True, "text": "episode is finalized"})["header"] == "episode is finalized"


def test_remote_episode_tracks_window_ops_and_done():
    ep = Episode("first-night", 7, redact_seed=True)
    remote = RemoteEpisode(_sidecar(ep), "prompt")
    remote.step("terraria_status")
    assert (remote.world.window, remote.task.windows) == (ep.world.window, 4)
    assert (remote.ops_left, remote.window_budget) == (ep.ops_left, ep.window_budget)
    while not remote.done:
        remote.step("end_window")
    assert ep.done


def test_two_tier_agent_plays_to_the_end_through_the_adapter():
    plan = json.dumps({"strategy": "gather", "candidates": [
        {"id": "chop", "tool": "chop_trees", "args": {"radius": 60, "max_trees": 2}, "when": "always"}]})

    class Decisions:
        def create(self, model, input, questions):
            assert model == "gemini-decision-1"
            return {"answers": {"next_action": {"choice": "chop", "confidence": 0.9, "probabilities": {"chop": 0.9}}}}

    class Client:
        decisions = Decisions()

    ep = Episode("first-night", 7, redact_seed=True)
    agent = GeminiAgent(model="gemini-4-hydrogen", live=True, supervisor_fn=lambda text, png: plan,
                        decisions=GeminiDecisionsWrapper(client=Client(), strict=True))
    turns = play(agent, RemoteEpisode(_sidecar(ep), "prompt"))
    assert ep.done and turns == sum(w["ops_used"] for w in ep.finalize()["windows"])
    assert agent.stats["decision_calls"] == turns and agent.stats["supervisor_calls"] == 4
