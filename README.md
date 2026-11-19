# AstroNewsFEED

An automated daily astronomy and space exploration news briefing powered by the **Google Gemini API** (`google-genai` SDK with **Google Search grounding**) and published as a static landing page via **GitHub Pages**.

- **Live Landing Page:** https://guidoyak.github.io/AstroNewsFEED/
- **Tracked Websites:** [`sites.md`](sites.md)
- **Structured Archive:** [`data/news.json`](data/news.json)

## What Each Dispatch Includes

Every notable astronomy or space exploration article added to `index.html` contains:

1. **Website Source** — Publication name and direct link to the source site and original article.
2. **Date** — Publication date (`YYYY-MM-DD`).
3. **Title of the News** — Clear headline linking to the full article.
4. **Resume (~100 words)** — Concise overview of the discovery, mission, or research event.
5. **Main Facts** — 3–5 concrete factual bullet points with key measurements, instruments, or milestones.
6. **Implications & Research Development** — Scientific/technological significance and expected next steps.

## How It Works

1. Every week on **Monday at 07:00 UTC** (or on-demand via **Actions → Weekly Astro News → Run workflow**), GitHub Actions executes [`scripts/astro_agent.py`](scripts/astro_agent.py) with a 7-day lookback window.
2. The script reads all website URLs listed in [`sites.md`](sites.md).
3. For each website, it calls Gemini (`gemini-4-hydrogen` by default) via the `google-genai` SDK with `GoogleSearch` grounding enabled to discover genuinely notable recent astronomy and space exploration news.
4. New stories are deduplicated against [`data/news.json`](data/news.json) (by normalized URL and headline similarity) and appended to the archive.
5. [`scripts/render_html.py`](scripts/render_html.py) rebuilds [`index.html`](index.html) cleanly from `data/news.json`, and the workflow commits and pushes the updated files to `main`, automatically refreshing GitHub Pages.

## Adding More Websites

Edit [`sites.md`](sites.md) and add any URL (as a bullet item, Markdown link, or plain URL on its own line):

```markdown
- https://www.astronomy.com/news/
- https://www.space.com/
- https://science.nasa.gov/universe/
```

The next weekly run will pick up the new sources automatically.

## Setup Checklist (GitHub Repository)

1. **Repository Secret:** Under **Settings → Secrets and variables → Actions**, make sure your Gemini API key is saved as `GEMINI_API_KEY`.
2. **Workflow Permissions:** Under **Settings → Actions → General → Workflow permissions**, ensure workflows have read/write permissions (the workflow also explicitly requests `contents: write`).
3. **GitHub Pages:** Under **Settings → Pages**, deploy from branch `main` and folder `/ (root)`.

## Running Locally

```bash
pip install -r requirements.txt

# Run a full scan and update data/news.json + index.html
export GEMINI_API_KEY="your-api-key"
python scripts/astro_agent.py

# Preview discovered stories without writing files
python scripts/astro_agent.py --dry-run

# Re-render index.html from existing data/news.json (no API key required)
python scripts/astro_agent.py --render-only
```

You can override the model with `--model <model-id>` or the `GEMINI_MODEL` environment variable.
