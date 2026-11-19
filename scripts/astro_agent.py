#!/usr/bin/env python3
"""Weekly Astronomy & Space Exploration News Agent powered by Google Gemini.

Reads websites from sites.md, scans each site for notable recent astronomy and
space exploration news using the google-genai SDK with Google Search grounding
(with automatic fallback to direct RSS/HTML fetching + Gemini synthesis on the
Gemini Free Tier where Google Search grounding requires billing), appends new
deduplicated entries to data/news.json, and renders index.html.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field

# Allow running both as `python scripts/astro_agent.py` and importing `scripts.astro_agent`
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_html import domain_label, sanitize_url, write_landing_page  # noqa: E402
from web_fetcher import fetch_site_articles  # noqa: E402

DEFAULT_MODEL = "gemini-4-hydrogen"
FALLBACK_FREE_TIER_MODELS = (
    "gemini-4-hydrogen",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MAX_PER_SITE = 5
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("astro_agent")


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------


class NewsEntrySchema(BaseModel):
    """Schema for a single astronomy/space exploration news dispatch."""

    website_source: str = Field(
        description="Human-readable publication name, e.g. 'Astronomy.com' or 'Space.com'."
    )
    article_url: str = Field(
        description="Direct https URL to the specific news article on the target website."
    )
    date: str = Field(
        description="Publication date of the article in YYYY-MM-DD format."
    )
    title: str = Field(description="Clear, accurate headline of the news article.")
    category: str = Field(
        description=(
            "Short topic tag such as 'Astrophysics', 'Exoplanets', 'Spaceflight', "
            "'Solar System', 'Cosmology', 'Observatories', or 'Planetary Science'."
        )
    )
    resume: str = Field(
        description=(
            "A crisp, engaging summary of approximately 100 words (target 90-115 words) "
            "explaining what happened, who conducted the work or mission, and why it stands out."
        )
    )
    main_facts: list[str] = Field(
        description="3 to 5 concise, verifiable factual bullet points with concrete figures, instruments, or milestones."
    )
    implications_and_development: str = Field(
        description=(
            "2 to 4 sentences explaining the broader scientific or technological implications "
            "and the expected next steps or future developments of the research or mission."
        )
    )


class SiteNewsResponse(BaseModel):
    """Top-level structured response returned by Gemini for one website."""

    articles: list[NewsEntrySchema] = Field(
        default_factory=list,
        description="List of notable recent astronomy and space exploration news articles found on the website. Empty if none qualify.",
    )


class GroundingQuotaError(RuntimeError):
    """Raised when Google Search grounding is unavailable due to Free Tier quota/billing."""


# ---------------------------------------------------------------------------
# sites.md Parsing
# ---------------------------------------------------------------------------

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")


def parse_sites_md(sites_path: Path) -> list[str]:
    """Extract unique http/https website URLs from sites.md in order."""
    if not sites_path.exists():
        raise FileNotFoundError(f"Sites file not found: {sites_path}")

    raw_text = sites_path.read_text(encoding="utf-8")
    cleaned = _HTML_COMMENT_RE.sub("", raw_text)
    cleaned = _FENCED_CODE_RE.sub("", cleaned)

    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(cleaned):
        candidate = match.group(0).rstrip(".,;:!?*`")
        safe_url = sanitize_url(candidate, fallback="")
        if not safe_url:
            continue
        key = normalize_url(safe_url)
        if key not in seen:
            seen.add(key)
            urls.append(safe_url)
    return urls


# ---------------------------------------------------------------------------
# Normalization & Deduplication Helpers
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Normalize a URL for reliable deduplication."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_QUERY_PARAMS
    ]
    query = urlencode(sorted(filtered_query))
    return urlunparse((scheme, netloc, path, "", query, ""))


def normalize_title_tokens(title: str) -> set[str]:
    """Extract lowercase alphanumeric tokens from a headline for similarity checks."""
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "as",
        "is",
        "are",
        "new",
    }
    tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    meaningful = tokens - stop_words
    return meaningful or tokens


def titles_are_similar(title_a: str, title_b: str, threshold: float = 0.78) -> bool:
    """Return True if two headlines refer to the same story."""
    norm_a = re.sub(r"[^a-z0-9]+", " ", title_a.lower()).strip()
    norm_b = re.sub(r"[^a-z0-9]+", " ", title_b.lower()).strip()
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    tokens_a = normalize_title_tokens(title_a)
    tokens_b = normalize_title_tokens(title_b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return (overlap / union) >= threshold


def is_specific_article_url(article_url: str, site_url: str) -> bool:
    """Return True when article_url points deeper than the bare source site URL."""
    norm_article = normalize_url(article_url)
    norm_site = normalize_url(site_url)
    if norm_article == norm_site:
        return False
    parsed = urlparse(article_url)
    return parsed.path.strip("/") != ""


def make_entry_id(title: str, date: str, source_url: str) -> str:
    """Create a stable, HTML-safe deterministic ID for a news entry."""
    digest = hashlib.sha256(
        f"{date}|{normalize_url(source_url)}|{title.strip().lower()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"astro-{digest}"


def sanitize_date(date_str: str, fallback_date: str) -> str:
    """Ensure publication date is valid YYYY-MM-DD, falling back safely."""
    candidate = (date_str or "").strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", candidate)
    if match:
        iso = match.group(0)
        try:
            datetime.strptime(iso, "%Y-%m-%d")
            return iso
        except ValueError:
            pass
    return fallback_date


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Parse JSON from Gemini output, tolerating markdown code fences or prose wrapping."""
    text = (raw_text or "").strip()
    if not text:
        return {"articles": []}

    fence_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start != -1 and obj_end > obj_start:
            parsed = json.loads(text[obj_start : obj_end + 1])
        else:
            raise

    if isinstance(parsed, list):
        return {"articles": parsed}
    if isinstance(parsed, dict):
        return parsed
    return {"articles": []}


# ---------------------------------------------------------------------------
# Gemini Prompting, Google Search Grounding & Free-Tier RSS/HTML Fallback
# ---------------------------------------------------------------------------


def build_scan_prompt(
    site_url: str,
    today_utc: datetime,
    lookback_days: int,
    max_articles: int,
    recent_titles: list[str],
) -> str:
    """Construct the prompt instructing Gemini to search a website for top astronomy news."""
    domain = domain_label(site_url)
    start_date = (today_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = today_utc.strftime("%Y-%m-%d")

    existing_block = ""
    if recent_titles:
        titles_list = "\n".join(f"- {t}" for t in recent_titles[:25])
        existing_block = (
            "\nAlready archived stories (DO NOT repeat or rewrite these):\n"
            f"{titles_list}\n"
        )

    return f"""Today's date is {end_date} (UTC).

Use Google Search to scan {site_url} (domain: {domain}) for notable, genuinely interesting astronomy and space exploration news published between {start_date} and {end_date}.

Selection rules:
1. Search specifically on {domain} (for example using site:{domain}) for the latest news articles from {start_date} through {end_date}.
2. Select up to {max_articles} of the most compelling, scientifically or technologically significant news stories about astronomy, astrophysics, exoplanets, telescopes, planetary science, cosmology, or space exploration/missions.
3. EXCLUDE shopping deals, telescope buying guides, horoscopes, astrological content, quizzes, sponsored promotions, evergreen explainers, or minor entertainment/sci-fi movie pieces.
4. If no genuinely notable news articles were published on {domain} in this date window, return an empty list: {{"articles": []}}. Never invent stories or reuse older articles outside the window.
{existing_block}
For each selected article, provide:
- website_source: Clean publication name (e.g. "{domain}" formatted nicely like "Astronomy.com" or "Space.com").
- article_url: The exact https URL of the news article on {domain}.
- date: Publication date in YYYY-MM-DD format.
- title: Accurate headline of the story.
- category: One short tag (e.g. "Astrophysics", "Exoplanets", "Spaceflight", "Solar System", "Cosmology", "Observatories", "Planetary Science").
- resume: A clear, well-written summary of approximately 100 words (90 to 115 words) capturing the core discovery or mission event.
- main_facts: 3 to 5 concise, factual bullet points citing concrete numbers, instruments, spacecraft, institutions, or observations from the report.
- implications_and_development: A focused paragraph (2 to 4 sentences) explaining the broader scientific or exploration implications and the expected next steps or future developments of this research/mission.

Return ONLY valid JSON matching the schema with top-level key "articles"."""


def build_direct_synthesis_prompt(
    site_url: str,
    fetched_articles: list[dict[str, str]],
    today_utc: datetime,
    max_articles: int,
    recent_titles: list[str],
) -> str:
    """Build a prompt for Gemini to curate and summarize live-fetched RSS/HTML articles."""
    domain = domain_label(site_url)
    end_date = today_utc.strftime("%Y-%m-%d")

    existing_block = ""
    if recent_titles:
        titles_list = "\n".join(f"- {t}" for t in recent_titles[:25])
        existing_block = (
            "\nAlready archived stories (DO NOT select or repeat these):\n"
            f"{titles_list}\n"
        )

    articles_text_blocks: list[str] = []
    for idx, art in enumerate(fetched_articles, start=1):
        articles_text_blocks.append(
            f"--- CANDIDATE {idx} ---\n"
            f"TITLE: {art.get('title', '').strip()}\n"
            f"URL: {art.get('url', '').strip()}\n"
            f"DATE: {art.get('date', '') or end_date}\n"
            f"CONTENT:\n{art.get('content', art.get('excerpt', '')).strip()}\n"
        )
    joined_articles = "\n".join(articles_text_blocks)

    return f"""Today's date is {end_date} (UTC).

Below are live articles fetched directly from {site_url} (domain: {domain}).
Review ALL candidates carefully and select up to {max_articles} of the most compelling, scientifically or technologically significant news stories about astronomy, astrophysics, exoplanets, telescopes, planetary science, cosmology, or space exploration/missions.

Rules:
1. ONLY use facts, URLs, and dates present in the candidate articles below. Never invent facts, numbers, or URLs.
2. Copy each selected article's exact URL into article_url and its exact publication date into date (YYYY-MM-DD).
3. EXCLUDE shopping deals, buying guides, product promos, entertainment/sci-fi movie reviews, horoscopes, or trivial pieces.
4. Select up to {max_articles} distinct, high-signal astronomy & space exploration stories (if fewer qualify, return only the ones that qualify).
{existing_block}
For each selected article, provide:
- website_source: Clean publication name (e.g. "Astronomy.com" or "Space.com").
- article_url: Exact URL copied from the candidate.
- date: Publication date in YYYY-MM-DD format.
- title: Clear, accurate headline of the story.
- category: One short tag ("Astrophysics", "Exoplanets", "Spaceflight", "Solar System", "Cosmology", "Observatories", or "Planetary Science").
- resume: A crisp, engaging summary of approximately 100 words (90 to 115 words) explaining what happened, who led the research/mission, and why it matters.
- main_facts: 3 to 5 concise, verifiable factual bullet points with concrete numbers, instruments, spacecraft, or observations from the article.
- implications_and_development: 2 to 4 sentences explaining the broader scientific or exploration implications and expected next steps or future developments of the research/mission.

CANDIDATE ARTICLES FROM {domain}:
{joined_articles}"""


def _is_quota_or_billing_error(exc: Exception) -> bool:
    """Return True when Gemini rejects a call with 429 quota/billing exhaustion."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status_code == 429 or "429" in msg or "resource_exhausted" in msg:
        return True
    return False


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Return True when a model is not accessible (404/403) on this project."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    return status_code in {403, 404} or "not found" in msg or "no longer available" in msg


def _is_transient_server_error(exc: Exception) -> bool:
    """Return True for transient 5xx/timeout errors worth retrying."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in {500, 502, 503, 504}:
        return True
    msg = str(exc).lower()
    return any(token in msg for token in ("503", "502", "504", "timeout", "temporarily unavailable"))


def _is_schema_grounding_conflict(exc: Exception) -> bool:
    """Detect when a model/endpoint refuses combining response_schema with google_search."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status_code == 400 and any(
        hint in msg
        for hint in ("response_schema", "response_mime_type", "json", "tool", "grounding", "search")
    ):
        return True
    return False


def _call_with_retry(fn, max_attempts: int = 3, base_delay: float = 2.0):
    """Execute a Gemini API call with exponential backoff on transient server failures."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _is_transient_server_error(exc):
                raise
            sleep_for = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Transient API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected retry exit")


def _extract_grounding_urls(response: Any) -> list[str]:
    """Extract cited URLs from Gemini grounding metadata for logging/verification."""
    urls: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return urls
    grounding = getattr(candidates[0], "grounding_metadata", None)
    if not grounding:
        return urls
    queries = getattr(grounding, "web_search_queries", None) or []
    if queries:
        logger.info("  Google Search queries executed: %s", queries)
    chunks = getattr(grounding, "grounding_chunks", None) or []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None) if web else None
        if uri:
            urls.append(str(uri))
    return urls


def _response_to_entries(
    raw_text: str,
    site_url: str,
    today_utc: datetime,
    max_articles: int,
) -> list[dict[str, Any]]:
    """Validate JSON text through SiteNewsResponse and normalize into archive dicts."""
    if not (raw_text or "").strip():
        logger.warning("Empty response returned for %s", site_url)
        return []

    payload = extract_json_payload(raw_text)
    validated = SiteNewsResponse.model_validate(payload)

    today_str = today_utc.strftime("%Y-%m-%d")
    now_iso = today_utc.isoformat().replace("+00:00", "Z")
    results: list[dict[str, Any]] = []
    for item in validated.articles[:max_articles]:
        clean_title = item.title.strip()
        clean_resume = item.resume.strip()
        if not clean_title or not clean_resume:
            continue

        safe_article_url = sanitize_url(item.article_url, fallback=site_url)
        clean_date = sanitize_date(item.date, fallback_date=today_str)
        clean_facts = [fact.strip() for fact in item.main_facts if fact and fact.strip()]
        clean_implications = item.implications_and_development.strip()

        results.append(
            {
                "id": make_entry_id(clean_title, clean_date, site_url),
                "website_source": item.website_source.strip() or domain_label(site_url),
                "source_url": site_url,
                "article_url": safe_article_url,
                "date": clean_date,
                "added_at": now_iso,
                "title": clean_title,
                "category": item.category.strip() or "Astronomy & Space",
                "resume": clean_resume,
                "main_facts": clean_facts,
                "implications_and_development": clean_implications,
            }
        )
    return results


def scan_site_with_gemini(
    client: Any,
    model: str,
    site_url: str,
    today_utc: datetime,
    lookback_days: int,
    max_articles: int,
    recent_titles: list[str],
) -> list[dict[str, Any]]:
    """Use google-genai with Google Search grounding to scan one website for news."""
    from google.genai import types

    prompt = build_scan_prompt(
        site_url=site_url,
        today_utc=today_utc,
        lookback_days=lookback_days,
        max_articles=max_articles,
        recent_titles=recent_titles,
    )
    system_instruction = (
        "You are an expert astronomy and space exploration science journalist and research analyst. "
        "Always ground every fact in live Google Search results from the requested website. "
        "Never hallucinate dates, URLs, or discoveries."
    )
    search_tool = types.Tool(google_search=types.GoogleSearch())

    def _structured_call():
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_tool],
                response_mime_type="application/json",
                response_schema=SiteNewsResponse,
            ),
        )

    def _fallback_json_call():
        fallback_prompt = (
            prompt
            + "\n\nRespond ONLY with raw JSON (no prose before or after) matching:\n"
            '{"articles": [{"website_source": "...", "article_url": "https://...", '
            '"date": "YYYY-MM-DD", "title": "...", "category": "...", "resume": "...", '
            '"main_facts": ["...", "..."], "implications_and_development": "..."}]}'
        )
        return client.models.generate_content(
            model=model,
            contents=fallback_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[search_tool],
            ),
        )

    try:
        response = _call_with_retry(_structured_call)
    except Exception as exc:  # noqa: BLE001
        if _is_quota_or_billing_error(exc):
            raise GroundingQuotaError(str(exc)) from exc
        if _is_schema_grounding_conflict(exc):
            logger.info(
                "Model '%s' does not allow response_schema + google_search together; "
                "falling back to grounded JSON prompting.",
                model,
            )
            try:
                response = _call_with_retry(_fallback_json_call)
            except Exception as inner_exc:  # noqa: BLE001
                if _is_quota_or_billing_error(inner_exc):
                    raise GroundingQuotaError(str(inner_exc)) from inner_exc
                raise
        else:
            raise

    _extract_grounding_urls(response)
    raw_text = getattr(response, "text", None) or ""
    return _response_to_entries(raw_text, site_url, today_utc, max_articles)


def _model_fallback_chain(preferred_model: str) -> list[str]:
    """Return deduplicated list of models to try on Free-Tier synthesis."""
    chain: list[str] = []
    for candidate in (preferred_model, *FALLBACK_FREE_TIER_MODELS):
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def scan_site_via_direct_fetch(
    client: Any,
    model: str,
    site_url: str,
    today_utc: datetime,
    lookback_days: int,
    max_articles: int,
    recent_titles: list[str],
    fetched_articles: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch live RSS/HTML articles from site_url and synthesize summaries with Gemini."""
    from google.genai import types

    candidates = (
        fetched_articles
        if fetched_articles is not None
        else fetch_site_articles(
            site_url=site_url,
            today_utc=today_utc,
            lookback_days=lookback_days,
            max_candidates=max(max_articles * 3, 12),
        )
    )
    if not candidates:
        logger.warning("  No recent articles fetched from %s", site_url)
        return [], model

    logger.info("  Synthesizing top %d stories from %d fetched article(s)...", max_articles, len(candidates))
    prompt = build_direct_synthesis_prompt(
        site_url=site_url,
        fetched_articles=candidates,
        today_utc=today_utc,
        max_articles=max_articles,
        recent_titles=recent_titles,
    )
    system_instruction = (
        "You are an expert astronomy and space exploration science journalist and research analyst. "
        "Synthesize accurate, insightful summaries grounded strictly in the provided article texts."
    )

    last_exc: Exception | None = None
    for candidate_model in _model_fallback_chain(model):
        try:
            response = _call_with_retry(
                lambda m=candidate_model: client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=SiteNewsResponse,
                    ),
                )
            )
            raw_text = getattr(response, "text", None) or ""
            return (
                _response_to_entries(raw_text, site_url, today_utc, max_articles),
                candidate_model,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_quota_or_billing_error(exc) or _is_model_unavailable_error(exc):
                logger.warning(
                    "  Model '%s' unavailable/rate-limited (%s); trying next fallback model...",
                    candidate_model,
                    getattr(exc, "code", type(exc).__name__),
                )
                continue
            raise

    if last_exc:
        raise last_exc
    return [], model


# ---------------------------------------------------------------------------
# Archive Management
# ---------------------------------------------------------------------------


def load_archive(data_path: Path) -> dict[str, Any]:
    """Load data/news.json or return an empty initialized archive."""
    if not data_path.exists():
        return {
            "last_updated": None,
            "last_run_status": "initialized",
            "model": DEFAULT_MODEL,
            "entries": [],
        }
    return json.loads(data_path.read_text(encoding="utf-8"))


def deduplicate_and_merge(
    existing_entries: list[dict[str, Any]],
    candidate_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append only genuinely new entries, avoiding duplicate URLs or headlines."""
    known_urls: set[str] = set()
    known_titles: list[str] = []

    for entry in existing_entries:
        src_url = str(entry.get("source_url", ""))
        art_url = str(entry.get("article_url", ""))
        if art_url and is_specific_article_url(art_url, src_url):
            known_urls.add(normalize_url(art_url))
        if entry.get("title"):
            known_titles.append(str(entry["title"]))

    added: list[dict[str, Any]] = []
    merged = list(existing_entries)

    for candidate in candidate_entries:
        cand_title = str(candidate.get("title", "")).strip()
        cand_src = str(candidate.get("source_url", ""))
        cand_url = str(candidate.get("article_url", ""))

        if not cand_title:
            continue

        norm_cand_url = normalize_url(cand_url) if cand_url else ""
        if (
            cand_url
            and is_specific_article_url(cand_url, cand_src)
            and norm_cand_url in known_urls
        ):
            logger.info("  Skipping duplicate URL: %s", cand_title)
            continue

        if any(titles_are_similar(cand_title, old) for old in known_titles):
            logger.info("  Skipping duplicate title: %s", cand_title)
            continue

        if cand_url and is_specific_article_url(cand_url, cand_src):
            known_urls.add(norm_cand_url)
        known_titles.append(cand_title)
        added.append(candidate)
        merged.append(candidate)

    merged.sort(
        key=lambda item: (str(item.get("date", "")), str(item.get("added_at", ""))),
        reverse=True,
    )
    return merged, added


def save_archive(data_path: Path, archive: dict[str, Any]) -> None:
    """Write the structured JSON archive cleanly with UTF-8 formatting."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(archive, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan astronomy websites listed in sites.md with Gemini + Google Search grounding and update index.html."
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=REPO_ROOT / "sites.md",
        help="Path to sites.md containing website URLs to scan (default: sites.md).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "news.json",
        help="Path to the structured JSON archive (default: data/news.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "index.html",
        help="Path to the generated HTML landing page (default: index.html).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        help=f"Gemini model ID (default: env GEMINI_MODEL or '{DEFAULT_MODEL}').",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Days to look back for recent news (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--max-per-site",
        type=int,
        default=DEFAULT_MAX_PER_SITE,
        help=f"Maximum notable stories to keep per website per run (default: {DEFAULT_MAX_PER_SITE}).",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "grounding", "direct"),
        default="auto",
        help=(
            "Scanning strategy: 'auto' tries Google Search grounding first and falls back "
            "to direct RSS/HTML fetching + Gemini on Free-Tier quota 429 (default: auto)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan websites and print discovered articles without modifying data/news.json or index.html.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip Gemini API calls and regenerate index.html directly from data/news.json and sites.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sites = parse_sites_md(args.sites)
    logger.info("Loaded %d tracked website(s) from %s", len(sites), args.sites)

    if args.render_only:
        write_landing_page(args.data, sites=sites, output_path=args.output)
        logger.info("Rendered landing page to %s (render-only mode).", args.output)
        return 0

    if not sites:
        logger.warning("No URLs found in %s. Nothing to scan.", args.sites)
        write_landing_page(args.data, sites=sites, output_path=args.output)
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "Missing GEMINI_API_KEY environment variable. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
            "or run with --render-only to rebuild HTML without scanning."
        )
        return 1

    from google import genai

    client = genai.Client(api_key=api_key)
    archive = load_archive(args.data)
    existing_entries: list[dict[str, Any]] = list(archive.get("entries") or [])
    today_utc = datetime.now(timezone.utc)

    discovered: list[dict[str, Any]] = []
    failed_sites: list[str] = []
    grounding_enabled = args.mode in {"auto", "grounding"}
    active_model = args.model

    for site_url in sites:
        logger.info("Scanning %s using model '%s'...", site_url, active_model)
        recent_for_site = [
            str(e.get("title", ""))
            for e in existing_entries[:40]
            if e.get("title")
        ]
        try:
            site_articles: list[dict[str, Any]]
            if grounding_enabled:
                try:
                    site_articles = scan_site_with_gemini(
                        client=client,
                        model=active_model,
                        site_url=site_url,
                        today_utc=today_utc,
                        lookback_days=args.lookback_days,
                        max_articles=args.max_per_site,
                        recent_titles=recent_for_site,
                    )
                except GroundingQuotaError as exc:
                    if args.mode == "grounding":
                        raise
                    logger.warning(
                        "Google Search grounding returned 429 quota/billing error (%s). "
                        "Switching to Free-Tier compatible Direct RSS/HTML Fetch + Gemini synthesis.",
                        exc,
                    )
                    grounding_enabled = False
                    site_articles, active_model = scan_site_via_direct_fetch(
                        client=client,
                        model=active_model,
                        site_url=site_url,
                        today_utc=today_utc,
                        lookback_days=args.lookback_days,
                        max_articles=args.max_per_site,
                        recent_titles=recent_for_site,
                    )
            else:
                site_articles, active_model = scan_site_via_direct_fetch(
                    client=client,
                    model=active_model,
                    site_url=site_url,
                    today_utc=today_utc,
                    lookback_days=args.lookback_days,
                    max_articles=args.max_per_site,
                    recent_titles=recent_for_site,
                )

            logger.info("  Found %d candidate article(s) on %s", len(site_articles), site_url)
            discovered.extend(site_articles)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed scanning %s: %s", site_url, exc)
            failed_sites.append(site_url)

    merged_entries, newly_added = deduplicate_and_merge(existing_entries, discovered)
    logger.info(
        "Scan complete: %d new article(s) added (%d total in archive).",
        len(newly_added),
        len(merged_entries),
    )

    if args.dry_run:
        print(json.dumps(newly_added, indent=2, ensure_ascii=False))
        return 1 if len(failed_sites) == len(sites) else 0

    if len(failed_sites) == len(sites):
        logger.error("All %d site scans failed; aborting without overwriting archive.", len(sites))
        return 1

    archive["last_updated"] = today_utc.isoformat().replace("+00:00", "Z")
    archive["model"] = active_model
    archive["last_run_status"] = (
        "partial_success" if failed_sites else "success"
    )
    archive["entries"] = merged_entries

    save_archive(args.data, archive)
    write_landing_page(args.data, sites=sites, output_path=args.output)
    logger.info("Updated %s and rendered %s.", args.data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
