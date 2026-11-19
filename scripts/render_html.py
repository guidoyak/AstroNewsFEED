"""Render the AstroNewsFEED landing page (index.html) from structured JSON data."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def sanitize_url(url: str | None, fallback: str = "#") -> str:
    """Allow only http/https URLs in rendered href attributes."""
    if not url:
        return fallback
    candidate = url.strip()
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return candidate
    return fallback


def domain_label(url: str) -> str:
    """Return a readable domain label from a URL (e.g. 'astronomy.com')."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or url


def format_display_date(date_str: str) -> str:
    """Format an ISO YYYY-MM-DD date string for editorial date headers."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return f"{dt.strftime('%A, %B')} {dt.day}, {dt.year}"
    except (ValueError, TypeError):
        return date_str or "Undated"


def format_timestamp(iso_str: str | None) -> str:
    """Format an ISO UTC timestamp for the masthead telemetry line."""
    if not iso_str:
        return "Awaiting first scan"
    try:
        normalized = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized).astimezone(timezone.utc)
        return dt.strftime("%b %d, %Y · %H:%M UTC")
    except ValueError:
        return iso_str


def _render_entry_card(entry: dict[str, Any], index: int) -> str:
    """Render a single astronomy/space news dispatch article."""
    title = escape(str(entry.get("title", "Untitled Dispatch")).strip())
    source_name = escape(
        str(
            entry.get("website_source")
            or domain_label(str(entry.get("source_url", "")))
        ).strip()
    )
    source_url = sanitize_url(entry.get("source_url"), fallback="#")
    article_url = sanitize_url(entry.get("article_url"), fallback=source_url)
    date_raw = str(entry.get("date", "")).strip()
    date_escaped = escape(date_raw)
    category = escape(str(entry.get("category", "Astronomy & Space")).strip())
    resume = escape(str(entry.get("resume", "")).strip())
    implications = escape(
        str(entry.get("implications_and_development", "")).strip()
    )

    facts_raw = entry.get("main_facts") or []
    facts_items = "\n".join(
        f"              <li>{escape(str(fact).strip())}</li>"
        for fact in facts_raw
        if str(fact).strip()
    )
    if not facts_items:
        facts_items = "              <li>Key observational details pending further reporting.</li>"

    search_blob = escape(
        " ".join(
            [
                str(entry.get("title", "")),
                str(entry.get("website_source", "")),
                str(entry.get("category", "")),
                str(entry.get("resume", "")),
                " ".join(str(f) for f in facts_raw),
                str(entry.get("implications_and_development", "")),
            ]
        ).lower()
    )

    source_key = escape(
        str(entry.get("website_source") or domain_label(source_url)).strip().lower()
    )
    category_key = escape(
        str(entry.get("category", "Astronomy & Space")).strip().lower()
    )
    entry_id = escape(str(entry.get("id") or f"dispatch-{index}"))

    return f"""          <article
            id="{entry_id}"
            class="dispatch-card"
            data-source="{source_key}"
            data-category="{category_key}"
            data-search="{search_blob}"
          >
            <div class="dispatch-meta">
              <div class="dispatch-meta-left">
                <a
                  class="source-badge"
                  href="{escape(source_url)}"
                  target="_blank"
                  rel="noopener noreferrer"
                >{source_name}</a>
                <span class="category-tag">{category}</span>
              </div>
              <time class="dispatch-date" datetime="{date_escaped}">{date_escaped}</time>
            </div>

            <h3 class="dispatch-title">
              <a href="{escape(article_url)}" target="_blank" rel="noopener noreferrer">{title}</a>
            </h3>

            <section class="dispatch-resume" aria-label="Resume">
              <p class="dispatch-section-label">Resume</p>
              <p>{resume}</p>
            </section>

            <div class="dispatch-details">
              <section class="dispatch-facts" aria-label="Main Facts">
                <p class="dispatch-section-label">Main Facts</p>
                <ul class="facts-list">
{facts_items}
                </ul>
              </section>

              <section class="dispatch-implications" aria-label="Main Implications and Research Development">
                <p class="dispatch-section-label">Implications &amp; Research Development</p>
                <p>{implications}</p>
              </section>
            </div>

            <footer class="dispatch-footer">
              <span class="source-origin">
                Source: <a href="{escape(source_url)}" target="_blank" rel="noopener noreferrer">{escape(domain_label(source_url))}</a>
              </span>
              <a
                class="read-link"
                href="{escape(article_url)}"
                target="_blank"
                rel="noopener noreferrer"
              >
                Read original article <span aria-hidden="true">&rarr;</span>
              </a>
            </footer>
          </article>"""


def render_html(
    news_data: dict[str, Any],
    tracked_sites: list[str],
    repo_url: str = "https://github.com/guidoyak/AstroNewsFEED",
) -> str:
    """Build the complete HTML document for the AstroNewsFEED landing page."""
    entries: list[dict[str, Any]] = list(news_data.get("entries") or [])
    entries.sort(
        key=lambda item: (
            str(item.get("date", "")),
            str(item.get("added_at", "")),
        ),
        reverse=True,
    )

    # Group entries chronologically by publication date (newest first)
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for entry in entries:
        day = str(entry.get("date") or "Undated")[:10]
        grouped.setdefault(day, []).append(entry)

    # Collect unique sources and categories for filter buttons
    sources_seen: OrderedDict[str, str] = OrderedDict()
    categories_seen: OrderedDict[str, str] = OrderedDict()
    for entry in entries:
        src_name = str(
            entry.get("website_source")
            or domain_label(str(entry.get("source_url", "")))
        ).strip()
        if src_name:
            sources_seen.setdefault(src_name.lower(), src_name)
        cat_name = str(entry.get("category", "Astronomy & Space")).strip()
        if cat_name:
            categories_seen.setdefault(cat_name.lower(), cat_name)

    source_buttons_html = "\n".join(
        f'              <button type="button" class="filter-btn" data-filter-source="{escape(key)}">{escape(label)}</button>'
        for key, label in sources_seen.items()
    )
    category_buttons_html = "\n".join(
        f'              <button type="button" class="filter-btn" data-filter-category="{escape(key)}">{escape(label)}</button>'
        for key, label in categories_seen.items()
    )

    if entries:
        sections_html_parts: list[str] = []
        running_index = 1
        for day, day_entries in grouped.items():
            cards = []
            for item in day_entries:
                cards.append(_render_entry_card(item, running_index))
                running_index += 1
            count_word = "dispatch" if len(day_entries) == 1 else "dispatches"
            sections_html_parts.append(
                f"""      <section class="date-section" data-date-group="{escape(day)}">
        <div class="date-heading-bar">
          <h2 class="date-heading">{escape(format_display_date(day))}</h2>
          <span class="date-count">{len(day_entries)} {count_word}</span>
        </div>
        <div class="entries-stack">
{chr(10).join(cards)}
        </div>
      </section>"""
            )
        feed_body_html = "\n".join(sections_html_parts)
    else:
        feed_body_html = """      <section class="empty-state" id="empty-state-initial">
        <p class="empty-state-eyebrow">Observatory Standby</p>
        <h2>Awaiting the First Weekly Astronomy Scan</h2>
        <p>
          The Gemini AI news agent is configured to scan every website listed in
          <code>sites.md</code> once a week using Google Search grounding and publish
          structured research summaries right here.
        </p>
        <p>
          Trigger the <strong>Weekly Astro News</strong> workflow in GitHub Actions
          or run the script locally to populate your first dispatches.
        </p>
        <div class="empty-state-code">python scripts/astro_agent.py</div>
      </section>"""

    tracked_sites_html = "\n".join(
        f'            <li><a href="{escape(sanitize_url(site))}" target="_blank" rel="noopener noreferrer">{escape(domain_label(site))}</a></li>'
        for site in tracked_sites
    )
    if not tracked_sites_html:
        tracked_sites_html = "            <li>No websites configured in sites.md yet.</li>"

    last_updated_text = escape(format_timestamp(news_data.get("last_updated")))
    model_name = escape(str(news_data.get("model") or "gemini-4-hydrogen"))
    total_entries = len(entries)
    total_sources = len(tracked_sites)
    sites_md_url = escape(f"{repo_url}/blob/main/sites.md")

    controls_block = ""
    if entries:
        category_group = (
            f"""          <div class="filter-pills" role="group" aria-label="Filter by topic">
            <button type="button" class="filter-btn is-active" data-filter-category="all">All Topics</button>
{category_buttons_html}
          </div>"""
            if len(categories_seen) > 1
            else ""
        )
        controls_block = f"""  <nav class="controls-bar" aria-label="Filter dispatches">
    <div class="container controls-inner">
      <div class="search-field">
        <label class="sr-only" for="dispatch-search">Search astronomy dispatches</label>
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.3-4.3"></path>
        </svg>
        <input
          id="dispatch-search"
          class="search-input"
          type="search"
          placeholder="Search headlines, facts, missions, telescopes..."
          autocomplete="off"
        />
      </div>
      <div class="filter-groups">
        <div class="filter-pills" role="group" aria-label="Filter by source">
          <button type="button" class="filter-btn is-active" data-filter-source="all">All Sources</button>
{source_buttons_html}
        </div>
{category_group}
      </div>
    </div>
  </nav>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AstroNewsFEED — Daily Astronomy &amp; Space Exploration Dispatches</title>
  <meta
    name="description"
    content="Automated daily astronomy and space exploration news briefing powered by Google Gemini with Google Search grounding."
  />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap"
    rel="stylesheet"
  />
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header class="masthead">
    <div class="container">
      <div class="masthead-top">
        <div class="masthead-edition">
          <span>Daily Space &amp; Astronomy Briefing</span>
          <span class="masthead-separator" aria-hidden="true">/</span>
          <span>Updated: <strong id="last-updated-value">{last_updated_text}</strong></span>
        </div>
        <div class="masthead-actions">
          <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle color theme">
            <span id="theme-toggle-label">Night Mode</span>
          </button>
        </div>
      </div>

      <div class="masthead-main">
        <div class="masthead-title-block">
          <h1 class="masthead-title">AstroNews<span>FEED</span></h1>
          <p class="masthead-subtitle">
            Daily curated dispatches from astronomy and space exploration—distilled into concise
            summaries, core observational facts, and research implications.
          </p>
        </div>

        <div class="masthead-metrics" aria-label="Archive statistics">
          <div class="metric-item">
            <span class="metric-value" id="metric-dispatches">{total_entries}</span>
            <span class="metric-label">Dispatches</span>
          </div>
          <div class="metric-item">
            <span class="metric-value" id="metric-sources">{total_sources}</span>
            <span class="metric-label">Tracked Sites</span>
          </div>
        </div>
      </div>
    </div>
  </header>

{controls_block}

  <main class="feed-main" id="feed-main">
    <div class="container">
      <p class="results-summary is-hidden" id="results-summary" role="status" aria-live="polite"></p>
{feed_body_html}
      <section class="empty-state is-hidden" id="empty-state-filtered">
        <p class="empty-state-eyebrow">No Matches</p>
        <h2>No dispatches match your current filter</h2>
        <p>Try clearing your search query or switching back to All Sources and All Topics.</p>
      </section>
    </div>
  </main>

  <footer class="site-footer">
    <div class="container">
      <div class="footer-grid">
        <div>
          <h2 class="footer-heading">Tracked Websites (sites.md)</h2>
          <ul class="sources-list" id="tracked-sources-list">
{tracked_sites_html}
          </ul>
        </div>
        <div>
          <h2 class="footer-heading">Add More Sources</h2>
          <p class="footer-note">
            Want to follow another observatory, journal, or space agency? Add its URL to
            <a href="{sites_md_url}" target="_blank" rel="noopener noreferrer"><code>sites.md</code></a>
            and the next weekly run will scan it automatically.
          </p>
        </div>
      </div>
      <div class="footer-meta">
        <span>Generated automatically by AstroNewsFEED · Gemini ({model_name}) + Google Search grounding</span>
        <a href="{escape(repo_url)}" target="_blank" rel="noopener noreferrer">View repository on GitHub &rarr;</a>
      </div>
    </div>
  </footer>

  <script>
    (function () {{
      // Theme persistence (light observatory daylight default, dark observatory night option)
      var root = document.documentElement;
      var toggleBtn = document.getElementById("theme-toggle");
      var toggleLabel = document.getElementById("theme-toggle-label");

      function applyTheme(theme) {{
        if (theme === "dark") {{
          root.setAttribute("data-theme", "dark");
          if (toggleLabel) toggleLabel.textContent = "Daylight Mode";
        }} else {{
          root.removeAttribute("data-theme");
          if (toggleLabel) toggleLabel.textContent = "Night Mode";
        }}
      }}

      var savedTheme = null;
      try {{
        savedTheme = localStorage.getItem("astronews-theme");
      }} catch (e) {{}}

      if (savedTheme) {{
        applyTheme(savedTheme);
      }} else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {{
        applyTheme("dark");
      }}

      if (toggleBtn) {{
        toggleBtn.addEventListener("click", function () {{
          var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
          applyTheme(next);
          try {{
            localStorage.setItem("astronews-theme", next);
          }} catch (e) {{}}
        }});
      }}

      // Client-side search & filtering
      var searchInput = document.getElementById("dispatch-search");
      var sourceBtns = Array.prototype.slice.call(document.querySelectorAll("[data-filter-source]"));
      var categoryBtns = Array.prototype.slice.call(document.querySelectorAll("[data-filter-category]"));
      var cards = Array.prototype.slice.call(document.querySelectorAll(".dispatch-card"));
      var dateSections = Array.prototype.slice.call(document.querySelectorAll(".date-section"));
      var summaryEl = document.getElementById("results-summary");
      var emptyFilteredEl = document.getElementById("empty-state-filtered");

      if (!cards.length) return;

      var activeSource = "all";
      var activeCategory = "all";

      function applyFilters() {{
        var query = searchInput ? searchInput.value.trim().toLowerCase() : "";
        var visibleTotal = 0;

        cards.forEach(function (card) {{
          var src = card.getAttribute("data-source") || "";
          var cat = card.getAttribute("data-category") || "";
          var haystack = card.getAttribute("data-search") || "";

          var matchSource = activeSource === "all" || src === activeSource;
          var matchCategory = activeCategory === "all" || cat === activeCategory;
          var matchQuery = !query || haystack.indexOf(query) !== -1;

          var visible = matchSource && matchCategory && matchQuery;
          card.classList.toggle("is-hidden", !visible);
          if (visible) visibleTotal += 1;
        }});

        dateSections.forEach(function (section) {{
          var visibleInSection = section.querySelectorAll(".dispatch-card:not(.is-hidden)").length;
          section.classList.toggle("is-hidden", visibleInSection === 0);
        }});

        var isFiltering = activeSource !== "all" || activeCategory !== "all" || query.length > 0;
        if (summaryEl) {{
          summaryEl.classList.toggle("is-hidden", !isFiltering);
          summaryEl.textContent = "Showing " + visibleTotal + " of " + cards.length + " dispatches";
        }}
        if (emptyFilteredEl) {{
          emptyFilteredEl.classList.toggle("is-hidden", visibleTotal > 0);
        }}
      }}

      if (searchInput) {{
        searchInput.addEventListener("input", applyFilters);
      }}

      sourceBtns.forEach(function (btn) {{
        btn.addEventListener("click", function () {{
          activeSource = btn.getAttribute("data-filter-source") || "all";
          sourceBtns.forEach(function (b) {{ b.classList.toggle("is-active", b === btn); }});
          applyFilters();
        }});
      }});

      categoryBtns.forEach(function (btn) {{
        btn.addEventListener("click", function () {{
          activeCategory = btn.getAttribute("data-filter-category") || "all";
          categoryBtns.forEach(function (b) {{ b.classList.toggle("is-active", b === btn); }});
          applyFilters();
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def write_landing_page(
    data_path: Path,
    sites: list[str],
    output_path: Path,
    repo_url: str = "https://github.com/guidoyak/AstroNewsFEED",
) -> None:
    """Load data/news.json and write the rendered index.html file."""
    if data_path.exists():
        news_data = json.loads(data_path.read_text(encoding="utf-8"))
    else:
        news_data = {"last_updated": None, "entries": []}

    html_doc = render_html(news_data, tracked_sites=sites, repo_url=repo_url)
    output_path.write_text(html_doc, encoding="utf-8")
