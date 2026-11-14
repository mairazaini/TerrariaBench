import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from terraria_bench.context_crawler import (
    API_URL,
    DEFAULT_START_URL,
    DOMAIN_PATTERN,
    MAX_DEPTH,
    MAX_PAGES,
    MAX_STOP_AFTER_MS,
    crawl_website,
    retrieve_wiki_context,
)
from terraria_bench.real_game import RealGameController


def test_crawl_uses_native_context_api_and_saves_markdown(tmp_path: Path):
    seen = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "request_id": "req-test",
            "partial": False,
            "metadata": {"numUrls": 1, "numSucceeded": 1},
            "results": [{"markdown": "# Eye of Cthulhu\nSummoned at night.", "metadata": {
                "url": "https://terraria.wiki.gg/wiki/Eye_of_Cthulhu",
                "title": "Eye of Cthulhu", "success": True,
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = crawl_website(output_dir=tmp_path, api_key="test-key", client=client)

    assert seen["url"] == API_URL
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == {
        "url": DEFAULT_START_URL,
        "maxPages": MAX_PAGES,
        "maxDepth": MAX_DEPTH,
        "urlRegex": DOMAIN_PATTERN,
        "followSubdomains": True,
        "useMainContentOnly": True,
        "includeLinks": True,
        "stopAfterMs": MAX_STOP_AFTER_MS,
    }
    assert result.pages[0].markdown.startswith("# Eye")
    assert Path(result.pages[0].path).exists()
    assert (tmp_path / "manifest.json").exists()


def test_retrieval_returns_relevant_pages_and_source(tmp_path: Path):
    (tmp_path / "eye.md").write_text("<!-- source: https://terraria.wiki.gg/wiki/Eye_of_Cthulhu -->\n\n# Eye of Cthulhu\nSummon at night.", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"pages": [{
        "url": "https://terraria.wiki.gg/wiki/Eye_of_Cthulhu",
        "title": "Eye of Cthulhu", "success": True, "path": "eye.md",
    }]}), encoding="utf-8")

    context = retrieve_wiki_context("How should I fight Eye of Cthulhu?", corpus_dir=tmp_path)

    assert "Source: https://terraria.wiki.gg/wiki/Eye_of_Cthulhu" in context
    assert "Summon at night." in context


def test_supervisor_receives_saved_wiki_context(tmp_path: Path):
    (tmp_path / "eye.md").write_text("<!-- source: https://terraria.wiki.gg/wiki/Eye_of_Cthulhu -->\n\n# Eye of Cthulhu\nSummon at night.", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"pages": [{
        "url": "https://terraria.wiki.gg/wiki/Eye_of_Cthulhu",
        "title": "Eye of Cthulhu", "success": True, "path": "eye.md",
    }]}), encoding="utf-8")
    seen = {}
    agent = SimpleNamespace(model="test-supervisor", decisions=SimpleNamespace(model="test-decision"), stats={"supervisor_calls": 0})
    def supervisor(prompt, shot):
        seen["prompt"] = prompt
        return '{"strategy":"Use verified wiki mechanics.","objective":"prepare"}'
    controller = RealGameController(agent, live=False, wiki_dir=str(tmp_path), supervisor_fn=supervisor)

    controller._supervise(1, {}, {"notes": "visible"}, None, "test")

    assert "TERRARIA WIKI REFERENCE" in seen["prompt"]
    assert "Source: https://terraria.wiki.gg/wiki/Eye_of_Cthulhu" in seen["prompt"]


@pytest.mark.parametrize("url", ["https://example.com/", "https://terraria.wiki.gg.evil.test/", "file:///wiki"])
def test_crawl_rejects_urls_outside_wiki_scope(url):
    with pytest.raises(ValueError):
        crawl_website(url, api_key="test-key")
