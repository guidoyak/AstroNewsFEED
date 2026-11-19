"""Lightweight RSS/Atom and HTML article fetcher for astronomy news websites.

Used as an automatic fallback when Gemini API Google Search grounding is unavailable
on the Free Tier (HTTP 429 RESOURCE_EXHAUSTED without billing), or when running in
direct-fetch mode.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests

logger = logging.getLogger("astro_agent")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

EXCLUDED_URL_PATTERNS = (
    "/deals",
    "/buying-guide",
    "/best-",
    "/shop",
    "/store",
    "/coupons",
    "/horoscope",
    "/astrology",
    "/crossword",
    "/quizzes",
    "/author/",
    "/tag/",
    "/category/",
    "/about",
    "/contact",
    "/privacy",
    "/terms",
    "/subscribe",
    "/newsletter",
    "/video/",
)

EXCLUDED_TITLE_PATTERNS = (
    "black friday",
    "cyber monday",
    "prime day",
    "deal:",
    "deals:",
    "save $",
    "save up to",
    "% off",
    "gift guide",
    "buying guide",
    "best telescopes to buy",
    "best binoculars to buy",
    "horoscope",
    "wordle",
    "crossword",
    "lego ",
)


def _strip_html(raw: str) -> str:
    """Convert an HTML fragment into clean plain text."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _parse_date_to_iso(raw_date: str | None) -> str:
    """Parse RFC 2822 or ISO 8601 date strings into YYYY-MM-DD, or '' if unknown."""
    if not raw_date:
        return ""
    candidate = raw_date.strip()
    if not candidate:
        return ""

    # Try ISO 8601 first
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", candidate)
    if iso_match:
        try:
            datetime.strptime(iso_match.group(0), "%Y-%m-%d")
            return iso_match.group(0)
        except ValueError:
            pass

    # Try RFC 2822 (standard RSS pubDate)
    try:
        dt = parsedate_to_datetime(candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, IndexError):
        return ""


def _is_non_news_candidate(title: str, url: str) -> bool:
    """Filter out shopping deals, promo pages, and non-news links before calling Gemini."""
    url_lower = url.lower()
    title_lower = title.lower()
    if any(pat in url_lower for pat in EXCLUDED_URL_PATTERNS):
        return True
    if any(pat in title_lower for pat in EXCLUDED_TITLE_PATTERNS):
        return True
    return False


def discover_feed_urls(site_url: str, html_text: str | None = None) -> list[str]:
    """Discover RSS/Atom feed URLs from page <link> tags plus standard feed paths."""
    parsed = urlparse(site_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    candidates: list[str] = []

    if html_text:
        soup = BeautifulSoup(html_text, "html.parser")
        for link in soup.find_all("link"):
            rel = " ".join(link.get("rel") or []).lower()
            link_type = str(link.get("type") or "").lower()
            href = link.get("href")
            if not href:
                continue
            if "alternate" in rel and any(
                token in link_type for token in ("rss", "atom", "xml")
            ):
                resolved = urljoin(site_url, str(href).strip())
                if resolved not in candidates:
                    candidates.append(resolved)

    # Common feed endpoints (WordPress, Future/Space.com, NASA, ESA, etc.)
    section_path = parsed.path.rstrip("/")
    fallback_paths = []
    if section_path:
        fallback_paths.append(f"{root}{section_path}/feed/")
        fallback_paths.append(f"{root}{section_path}/feed")
    fallback_paths.extend(
        [
            f"{root}/feeds/all",
            f"{root}/feeds.xml",
            f"{root}/feed/",
            f"{root}/feed",
            f"{root}/rss.xml",
            f"{root}/rss",
        ]
    )
    for url in fallback_paths:
        if url not in candidates:
            candidates.append(url)
    return candidates


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from an ElementTree tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1].lower()
    return tag.lower()


def parse_feed_xml(xml_text: str, base_url: str) -> list[dict[str, str]]:
    """Parse an RSS 2.0 or Atom XML feed into normalized article dictionaries."""
    if not xml_text or "<" not in xml_text:
        return []
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError:
        return []

    items: list[dict[str, str]] = []
    root_name = _local_tag(root.tag)

    if root_name in {"rss", "rdf"}:
        for elem in root.iter():
            if _local_tag(elem.tag) != "item":
                continue
            title = ""
            link = ""
            date_str = ""
            excerpt = ""
            encoded = ""
            for child in elem:
                name = _local_tag(child.tag)
                text = (child.text or "").strip()
                if name == "title" and not title:
                    title = _strip_html(text)
                elif name == "link" and not link:
                    link = urljoin(base_url, text)
                elif name in {"pubdate", "date", "published", "updated"} and not date_str:
                    date_str = _parse_date_to_iso(text)
                elif name == "description" and not excerpt:
                    excerpt = _strip_html(text)
                elif name == "encoded" and not encoded:
                    encoded = _strip_html(text)
            if title and link:
                items.append(
                    {
                        "title": title,
                        "url": link,
                        "date": date_str,
                        "excerpt": encoded if len(encoded) > len(excerpt) else excerpt,
                    }
                )
    elif root_name == "feed":
        for elem in root:
            if _local_tag(elem.tag) != "entry":
                continue
            title = ""
            link = ""
            date_str = ""
            excerpt = ""
            for child in elem:
                name = _local_tag(child.tag)
                text = (child.text or "").strip()
                if name == "title" and not title:
                    title = _strip_html(text)
                elif name == "link":
                    rel = (child.attrib.get("rel") or "alternate").lower()
                    href = child.attrib.get("href") or text
                    if href and (not link or rel == "alternate"):
                        link = urljoin(base_url, href.strip())
                elif name in {"published", "updated"} and not date_str:
                    date_str = _parse_date_to_iso(text)
                elif name in {"content", "summary"} and len(_strip_html(text)) > len(excerpt):
                    excerpt = _strip_html(text)
            if title and link:
                items.append(
                    {
                        "title": title,
                        "url": link,
                        "date": date_str,
                        "excerpt": excerpt,
                    }
                )

    return items


def extract_links_from_html(html_text: str, site_url: str) -> list[dict[str, str]]:
    """Fallback HTML link extractor when no RSS/Atom feed is available."""
    soup = BeautifulSoup(html_text, "html.parser")
    site_host = urlparse(site_url).netloc.lower().lstrip("www.")
    seen: set[str] = set()
    items: list[dict[str, str]] = []

    containers = soup.select("article a[href], main a[href], h2 a[href], h3 a[href]")
    if not containers:
        containers = soup.find_all("a", href=True)

    for anchor in containers:
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        full_url = urljoin(site_url, href)
        parsed = urlparse(full_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        host = parsed.netloc.lower().lstrip("www.")
        if host != site_host:
            continue
        if len(parsed.path.strip("/")) < 10:
            continue

        title = anchor.get_text(" ", strip=True)
        if len(title) < 20:
            continue
        if full_url in seen or _is_non_news_candidate(title, full_url):
            continue
        seen.add(full_url)
        items.append({"title": title, "url": full_url, "date": "", "excerpt": ""})
        if len(items) >= 25:
            break

    return items


def _extract_article_metadata_and_body(html_text: str, max_chars: int = 3500) -> dict[str, str]:
    """Extract publication date and clean body paragraphs from an article HTML page."""
    soup = BeautifulSoup(html_text, "html.parser")
    pub_date = ""

    # 1. Meta tags
    for attr_name, attr_val in (
        ("property", "article:published_time"),
        ("name", "pubdate"),
        ("name", "publish-date"),
        ("name", "date"),
        ("itemprop", "datePublished"),
    ):
        tag = soup.find("meta", attrs={attr_name: attr_val})
        if tag and tag.get("content"):
            pub_date = _parse_date_to_iso(str(tag["content"]))
            if pub_date:
                break

    # 2. JSON-LD datePublished
    if not pub_date:
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                data: Any = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue
            nodes = data if isinstance(data, list) else [data]
            for node in nodes:
                if isinstance(node, dict):
                    raw = node.get("datePublished") or node.get("dateCreated")
                    if raw:
                        pub_date = _parse_date_to_iso(str(raw))
                        if pub_date:
                            break
            if pub_date:
                break

    # 3. <time datetime="...">
    if not pub_date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            pub_date = _parse_date_to_iso(str(time_tag["datetime"]))

    # Remove non-article noise before reading paragraphs
    for noisy in soup.select("script, style, nav, footer, header, aside, form, noscript"):
        noisy.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs: list[str] = []
    total_len = 0
    for p in root.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(text) < 35:
            continue
        paragraphs.append(text)
        total_len += len(text) + 1
        if total_len >= max_chars:
            break

    body = "\n".join(paragraphs)[:max_chars]
    return {"date": pub_date, "body": body}


def fetch_site_articles(
    site_url: str,
    today_utc: datetime,
    lookback_days: int,
    max_candidates: int = 12,
    timeout: int = 15,
) -> list[dict[str, str]]:
    """Fetch recent astronomy/space news candidates directly from a site's RSS/HTML."""
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    page_html: str | None = None
    try:
        resp = session.get(site_url, timeout=timeout)
        if resp.ok:
            page_html = resp.text
    except requests.RequestException as exc:
        logger.warning("  Could not fetch landing HTML %s: %s", site_url, exc)

    feed_urls = discover_feed_urls(site_url, page_html)
    raw_items: list[dict[str, str]] = []

    for feed_url in feed_urls:
        try:
            resp = session.get(feed_url, timeout=timeout)
            if not resp.ok:
                continue
            parsed = parse_feed_xml(resp.text, base_url=feed_url)
            if parsed:
                logger.info(
                    "  Fetched %d item(s) from RSS/Atom feed: %s",
                    len(parsed),
                    feed_url,
                )
                raw_items = parsed
                break
        except requests.RequestException:
            continue

    if not raw_items and page_html:
        raw_items = extract_links_from_html(page_html, site_url)
        logger.info(
            "  Extracted %d candidate link(s) from HTML page: %s",
            len(raw_items),
            site_url,
        )

    if not raw_items:
        return []

    cutoff_str = (today_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    filtered: list[dict[str, str]] = []
    undated_fallback: list[dict[str, str]] = []

    for item in raw_items:
        if _is_non_news_candidate(item["title"], item["url"]):
            continue
        item_date = item.get("date", "")
        if item_date:
            if item_date >= cutoff_str:
                filtered.append(item)
        else:
            undated_fallback.append(item)

    candidates = filtered if filtered else undated_fallback
    # If every dated item was slightly older than lookback_days, keep empty rather than stale news
    if not candidates and not filtered and not undated_fallback:
        return []

    selected = candidates[:max_candidates]
    enriched: list[dict[str, str]] = []
    for item in selected:
        article = dict(item)
        body_text = article.get("excerpt", "")
        # Fetch full article page when excerpt is short or date is missing
        if len(body_text) < 700 or not article.get("date"):
            try:
                art_resp = session.get(article["url"], timeout=12)
                if art_resp.ok:
                    meta = _extract_article_metadata_and_body(art_resp.text)
                    if meta["date"] and not article.get("date"):
                        article["date"] = meta["date"]
                    if len(meta["body"]) > len(body_text):
                        body_text = meta["body"]
            except requests.RequestException:
                pass
        article["content"] = body_text[:3500]
        # Re-check date cutoff if date was discovered on the article page
        if article.get("date") and article["date"] < cutoff_str:
            continue
        enriched.append(article)

    return enriched
