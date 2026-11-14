"""Context.dev crawler and local retrieval for Terraria Wiki Markdown."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

DEFAULT_START_URL = "https://terraria.wiki.gg/"
API_URL = "https://api.context.dev/v1/web/crawl"
MAX_PAGES = 500
MAX_DEPTH = 10_000
MAX_STOP_AFTER_MS = 110_000
DOMAIN_PATTERN = r"^https?://([a-z0-9-]+\.)*terraria\.wiki\.gg(?::\d+)?/"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "terraria_wiki"


class ContextCrawlError(RuntimeError):
    """Raised when Context.dev cannot complete a crawl request."""


@dataclass
class CrawlPage:
    url: str
    title: str
    markdown: str
    success: bool
    path: str
    metadata: dict[str, Any]


@dataclass
class CrawlResult:
    start_url: str
    pages: list[CrawlPage]
    partial: bool
    metadata: dict[str, Any]
    output_dir: str
    request_id: str | None = None


def _validate_start_url(start_url: str) -> str:
    parsed = urlparse(start_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not (host == "terraria.wiki.gg" or host.endswith(".terraria.wiki.gg")):
        raise ValueError("start URL must be on terraria.wiki.gg or one of its subdomains")
    if parsed.username or parsed.password:
        raise ValueError("credentials are not allowed in the start URL")
    return start_url


def _safe_stem(title: str, url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:72] or "page"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def crawl_website(
    start_url: str = DEFAULT_START_URL,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> CrawlResult:
    """Crawl up to Context.dev's maximum 500 pages and save each as Markdown.

    The returned pages include the complete Markdown and the output path. Network
    access is opt-in through an explicit key or CONTEXT_DEV_API_KEY.
    """
    start_url = _validate_start_url(start_url)
    key = api_key or os.environ.get("CONTEXT_DEV_API_KEY")
    if not key:
        raise RuntimeError("CONTEXT_DEV_API_KEY is required; set it in the server environment")

    payload = {
        "url": start_url,
        "maxPages": MAX_PAGES,
        "maxDepth": MAX_DEPTH,
        "urlRegex": DOMAIN_PATTERN,
        "followSubdomains": True,
        "useMainContentOnly": True,
        "includeLinks": True,
        "stopAfterMs": MAX_STOP_AFTER_MS,
    }
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0))
    try:
        response = http.post(
            API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(120.0, connect=15.0),
        )
    except httpx.HTTPError as exc:
        raise ContextCrawlError(f"Context.dev request failed: {type(exc).__name__}") from exc
    finally:
        if owns_client:
            http.close()

    if response.is_error:
        # Do not include request headers or credentials in diagnostics.
        try:
            detail = response.json()
            message = str(detail.get("message") or detail.get("error_code") or "request rejected")
        except (ValueError, AttributeError):
            message = "request rejected"
        raise ContextCrawlError(f"Context.dev returned HTTP {response.status_code}: {message[:240]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise ContextCrawlError("Context.dev returned invalid JSON") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[CrawlPage] = []
    for item in data.get("results", []):
        meta = item.get("metadata") or {}
        url = str(meta.get("url") or meta.get("finalUrl") or "")
        title = str(meta.get("title") or url or "Untitled")
        markdown = str(item.get("markdown") or "")
        # The API URL pattern is a crawl constraint; independently validate returned redirects.
        if url:
            _validate_start_url(url)
        filename = _safe_stem(title, url or f"result-{len(pages)}") + ".md"
        path = out_dir / filename
        header = f"<!-- source: {url}\ntitle: {title}\nretrieved: {datetime.now(timezone.utc).isoformat()} -->\n\n"
        path.write_text(header + markdown, encoding="utf-8")
        pages.append(CrawlPage(
            url=url,
            title=title,
            markdown=markdown,
            success=bool(meta.get("success", bool(markdown))),
            path=str(path),
            metadata=meta,
        ))

    manifest = {
        "start_url": start_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "partial": bool(data.get("partial", False)),
        "metadata": data.get("metadata") or {},
        "request_id": data.get("request_id"),
        "pages": [
            {"url": p.url, "title": p.title, "success": p.success,
             "path": Path(p.path).name, "characters": len(p.markdown)}
            for p in pages
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return CrawlResult(
        start_url=start_url,
        pages=pages,
        partial=manifest["partial"],
        metadata=manifest["metadata"],
        output_dir=str(out_dir),
        request_id=manifest["request_id"],
    )


_WORD = re.compile(r"[a-z0-9]{3,}", re.I)
_STOP_WORDS = {"the", "and", "for", "from", "with", "that", "this", "are", "you", "your", "what", "when", "where", "how", "now", "then", "have", "has", "into", "about", "would", "could", "should", "player", "terraria", "cthulhu"}


_META_PREFIXES = (
    "category:", "template:", "help:", "terraria wiki:", "module:", "user:",
    "legacy:", "editing ", "database data", "information for", "pages that link", "creating ",
)

_PAGE_BOOSTS: tuple[tuple[set[str], set[str], int], ...] = (
    ({"eye of cthulhu"}, {"eye", "boss", "summon", "suspicious", "lens", "cthulhu"}, 50),
    ({"house"}, {"house", "housing", "shelter", "door", "chair", "table", "wall", "walls", "npc", "guide"}, 35),
    ({"background walls"}, {"house", "housing", "shelter", "wall", "walls"}, 24),
    ({"guide:getting started", "guide:walkthrough"}, {"wood", "tree", "trees", "chop", "axe", "start", "house", "night", "early"}, 26),
    ({"work benches", "crafting stations", "guide:crafting 101"}, {"workbench", "bench", "furnace", "anvil", "craft", "crafting", "recipe"}, 25),
    ({"ores", "bars", "underground", "cavern"}, {"ore", "ores", "bar", "bars", "mine", "mining", "pickaxe", "cave", "caves", "underground", "cavern"}, 24),
    ({"chests", "health", "potions"}, {"chest", "chests", "loot", "heart", "crystal", "health", "hp", "potion", "potions", "heal"}, 24),
    ({"blocks", "tools"}, {"block", "blocks", "place", "placement", "tree", "trees", "pickaxe", "axe", "dig"}, 20),
    ({"ranged weapons", "weapons", "armor", "bosses"}, {"weapon", "weapons", "bow", "arrow", "arrows", "ranged", "armor", "combat", "fight", "arena", "boss"}, 18),
    ({"day and night cycle"}, {"night", "day", "dawn", "dusk", "summon", "eye"}, 16),
)

_PAGE_TOPIC_TERMS: dict[str, set[str]] = {
    "eye of cthulhu": {"summon", "summoned", "suspicious", "lens", "lenses", "night", "dawn", "phase", "charge", "charges", "despawn", "health", "life", "arena", "platforms"},
    "house": {"house", "housing", "light", "chair", "table", "door", "walls", "wall", "tiles", "width", "height", "workbench"},
    "background walls": {"wall", "walls", "house", "housing", "hammer", "safe", "natural", "wood"},
    "blocks": {"blocks", "block", "place", "placement", "tree", "trees", "support", "reach", "platform", "platforms"},
    "guide:getting started": {"tree", "trees", "wood", "axe", "pickaxe", "house", "workbench", "torch", "torches", "gel", "night", "shelter", "platform"},
    "guide:walkthrough": {"tree", "wood", "house", "cave", "caves", "ore", "furnace", "anvil", "life", "crystal", "chest", "arena", "eye", "suspicious"},
    "work benches": {"work", "bench", "wood", "craft", "crafting", "furnace", "chair", "door", "wall", "armor"},
    "crafting stations": {"workbench", "bench", "furnace", "anvil", "demon", "altar", "suspicious", "looking", "eye", "lens"},
    "ores": {"ore", "ores", "pickaxe", "copper", "tin", "iron", "lead", "silver", "tungsten", "gold", "platinum", "furnace", "bar"},
    "chests": {"chest", "chests", "loot", "underground", "surface", "cavern", "potion", "hermes", "cloud"},
    "health": {"life", "crystal", "crystals", "heart", "hearts", "max", "hp", "20", "potion", "healing", "campfire"},
    "ranged weapons": {"bow", "bows", "arrow", "arrows", "flaming", "frostburn", "jester", "shuriken", "eye"},
    "bosses": {"arena", "platforms", "campfire", "buff", "potion", "summon", "night"},
}


def retrieve_wiki_context(query: str, *, corpus_dir: str | Path = DEFAULT_OUTPUT_DIR,
                          max_pages: int = 4, max_chars: int = 7200) -> str:
    """Return relevant saved wiki pages for a model prompt; never performs network I/O."""
    root = Path(corpus_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    terms = {w.lower() for w in _WORD.findall(query) if w.lower() not in _STOP_WORDS}
    terms |= {word[:-1] for word in tuple(terms) if word.endswith("s") and len(word) > 4}
    ranked: list[tuple[int, dict[str, Any]]] = []
    seen_urls: set[str] = set()
    for page in manifest.get("pages", []):
        if not page.get("success") or not page.get("path"):
            continue
        parsed_url = urlparse(str(page.get("url") or ""))
        canonical_url = urlunparse(parsed_url._replace(fragment="", query=""))
        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        title = str(page.get("title", "")).lower()
        # Terraria has a post-Plantera yoyo named "The Eye of Cthulhu"; skip it and wiki maintenance pages.
        if title == "the eye of cthulhu" or title.startswith(_META_PREFIXES):
            continue
        title_terms = {w.lower() for w in _WORD.findall(f"{page.get('title', '')} {page.get('url', '')}")}
        title_terms |= {word[:-1] for word in tuple(title_terms) if word.endswith("s") and len(word) > 4}
        score = len(terms & title_terms) * 5
        for page_titles, match_terms, bonus in _PAGE_BOOSTS:
            if title in page_titles and (terms & match_terms):
                score += bonus
        if not score:
            # Give core progression/mechanics pages a chance when the query is broad.
            if any(x in title for x in ("eye of cthulhu", "getting started", "walkthrough", "house", "ores", "chests", "health")):
                score = 1
        if score:
            ranked.append((score, page))
    ranked.sort(key=lambda x: (-x[0], str(x[1].get("title", ""))))

    chunks: list[str] = []
    used = 0
    per_page_chars = max(1200, max_chars // max(1, max_pages))
    for _, page in ranked[:max_pages]:
        try:
            text = (root / page["path"]).read_text(encoding="utf-8")
        except OSError:
            continue
        # Drop our file metadata comment; the source URL is supplied as a citation line.
        text = re.sub(r"^<!--.*?-->\s*", "", text, count=1, flags=re.S).strip()
        remaining = max_chars - used
        if remaining <= 300:
            break
        title = str(page.get("title", "")).lower()
        topic_terms = _PAGE_TOPIC_TERMS.get(title, set())
        lines = text.splitlines()
        title_line = next((line for line in lines if line.startswith("# ")), "")
        candidates: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped == title_line or len(stripped) < 35:
                continue
            line_terms = {w.lower() for w in _WORD.findall(stripped)}
            score = len(terms & line_terms) + 3 * len(topic_terms & line_terms)
            if score:
                candidates.append((score, index, stripped[:1000]))
        # The page's infobox and navigation often precede the useful prose.
        # Select relevant statements throughout the page instead of truncating
        # the first 1,800 characters and losing the boss mechanics near the end.
        selected_lines = [entry[2] for entry in sorted(candidates, key=lambda x: (-x[0], x[1]))[:10]]
        selected_lines = sorted(set(selected_lines), key=lambda line: next(
            (i for i, candidate in enumerate(lines) if candidate.strip() == line), len(lines)
        ))
        if selected_lines:
            text = "\n\n".join(([title_line] if title_line else []) + selected_lines)
        page_limit = min(remaining, per_page_chars)
        if len(text) > page_limit:
            text = text[:page_limit].rsplit("\n", 1)[0].rstrip()
        chunk = f"### {page.get('title') or 'Terraria Wiki'}\nSource: {page.get('url', '')}\n\n{text}"
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    # Match the project's normal CLI behavior: read .env locally, never print it.
    from .live_play import load_dotenv_if_needed
    load_dotenv_if_needed()
    parser = argparse.ArgumentParser(description="Crawl Terraria Wiki pages to Markdown with Context.dev")
    parser.add_argument("url", nargs="?", default=DEFAULT_START_URL, help=f"Starting URL (default: {DEFAULT_START_URL})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    try:
        result = crawl_website(args.url, output_dir=args.output_dir)
    except (ContextCrawlError, RuntimeError, ValueError) as exc:
        print(f"crawl failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "start_url": result.start_url,
        "pages_returned": len(result.pages),
        "successful_pages": sum(p.success for p in result.pages),
        "partial": result.partial,
        "metadata": result.metadata,
        "request_id": result.request_id,
        "output_dir": result.output_dir,
        "manifest": str(Path(result.output_dir) / "manifest.json"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
