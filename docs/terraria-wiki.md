# Terraria Wiki knowledge for the live agent

The live real-game supervisor can use locally saved Terraria Wiki Markdown as
grounded reference context. The fast control loop stays network-free; when the
supervisor makes or updates a plan, it retrieves a few relevant pages from the
local crawl corpus and includes the source URLs in its prompt.

## Configure and crawl

Set `CONTEXT_DEV_API_KEY` in the repository's ignored `.env` file. The key is
only sent from the server-side Python process to Context.dev.

Run the crawl with the default Terraria Wiki start URL:

```powershell
.\.venv\Scripts\python.exe -m terraria_bench.context_crawler
```

To start at a different page on the Terraria Wiki:

```powershell
.\.venv\Scripts\python.exe -m terraria_bench.context_crawler "https://terraria.wiki.gg/wiki/Eye_of_Cthulhu"
```

Each returned page is saved as a Markdown file under `data/terraria_wiki/`;
`manifest.json` records page URLs, titles, status, and crawl metadata. The crawl
follows links on `terraria.wiki.gg` and its subdomains, requests up to 500 pages,
and has a 110-second soft deadline. Context.dev charges one credit per page.
Use batches for a larger corpus.

`crawl_website()` also returns the page Markdown and metadata to Python callers.
The saved corpus is ignored by Git. Set `TERRARIA_WIKI_DIR` to use another local
corpus directory for the real-game supervisor.
