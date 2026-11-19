"""Unit tests for AstroNewsFEED agent and HTML renderer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import astro_agent  # noqa: E402
import render_html  # noqa: E402


def test_parse_sites_md_handles_bullets_links_comments_and_dedup(tmp_path: Path) -> None:
    sites_file = tmp_path / "sites.md"
    sites_file.write_text(
        """# Sources
<!-- https://ignored.example.com/comment -->
- https://www.astronomy.com/news/
- [Space](https://www.space.com/)
https://astronomy.com/news/
```
https://ignored-code-block.example.com/
```
- https://science.nasa.gov/universe/.
""",
        encoding="utf-8",
    )

    urls = astro_agent.parse_sites_md(sites_file)
    assert urls == [
        "https://www.astronomy.com/news/",
        "https://www.space.com/",
        "https://science.nasa.gov/universe/",
    ]


def test_extract_json_payload_handles_code_fences_and_prose() -> None:
    fenced = 'Here is the result:\n```json\n{"articles": [{"title": "JWST Spots Galaxy"}]}\n```'
    assert astro_agent.extract_json_payload(fenced) == {
        "articles": [{"title": "JWST Spots Galaxy"}]
    }


def test_deduplicate_and_merge_blocks_duplicate_urls_and_similar_titles() -> None:
    existing = [
        {
            "id": "astro-1",
            "website_source": "Space.com",
            "source_url": "https://www.space.com/",
            "article_url": "https://www.space.com/jwst-exoplanet-water-vapor",
            "date": "2026-11-18",
            "added_at": "2026-11-18T07:00:00Z",
            "title": "James Webb Telescope Detects Water Vapor on Temperate Exoplanet",
        }
    ]
    candidates = [
        # Duplicate URL with tracking query param
        {
            "id": "astro-2",
            "website_source": "Space.com",
            "source_url": "https://www.space.com/",
            "article_url": "https://space.com/jwst-exoplanet-water-vapor?utm_source=rss",
            "date": "2026-11-19",
            "added_at": "2026-11-19T07:00:00Z",
            "title": "Completely Different Headline For Same Link",
        },
        # Very similar title from another run
        {
            "id": "astro-3",
            "website_source": "Astronomy.com",
            "source_url": "https://www.astronomy.com/news/",
            "article_url": "https://www.astronomy.com/news/webb-water-vapor",
            "date": "2026-11-19",
            "added_at": "2026-11-19T07:00:00Z",
            "title": "James Webb Telescope detects water vapor on a temperate exoplanet!",
        },
        # Genuinely new story
        {
            "id": "astro-4",
            "website_source": "Astronomy.com",
            "source_url": "https://www.astronomy.com/news/",
            "article_url": "https://www.astronomy.com/news/europa-plume-telemetry",
            "date": "2026-11-19",
            "added_at": "2026-11-19T07:00:00Z",
            "title": "Europa Clipper Measures Subsurface Plume Chemistry During Flyby",
        },
    ]

    merged, added = astro_agent.deduplicate_and_merge(existing, candidates)
    assert len(added) == 1
    assert added[0]["id"] == "astro-4"
    assert len(merged) == 2
    assert merged[0]["date"] == "2026-11-19"


def test_scan_site_with_gemini_uses_google_search_grounding() -> None:
    captured_calls: list[dict] = []

    sample_json = json.dumps(
        {
            "articles": [
                {
                    "website_source": "Astronomy.com",
                    "article_url": "https://www.astronomy.com/news/sample-pulsar-timing",
                    "date": "2026-11-19",
                    "title": "Pulsar Timing Array Pinpoints Supermassive Black Hole Binary",
                    "category": "Astrophysics",
                    "resume": (
                        "An international collaboration of radio observatories has identified "
                        "a strong gravitational-wave signature linked to a supermassive black "
                        "hole binary roughly two billion light-years away. By monitoring dozens "
                        "of millisecond pulsars over fifteen years, researchers isolated a subtle "
                        "nanohertz oscillation standing above the cosmic background. The result "
                        "marks a major transition from detecting an aggregate stochastic hum to "
                        "resolving individual merging galactic cores, giving astrophysicists a "
                        "direct probe of how the heaviest black holes overcome the final-parsec "
                        "barrier and shape host galaxy evolution across cosmic time."
                    ),
                    "main_facts": [
                        "Data combines 15+ years of timing measurements across 68 millisecond pulsars.",
                        "Candidate binary lies approximately 2 billion light-years from Earth.",
                        "Signal peaks in the nanohertz gravitational-wave frequency band.",
                    ],
                    "implications_and_development": (
                        "Resolving an individual supermassive binary tests general relativity in "
                        "an unexplored mass regime and constrains galaxy merger rates. Next-generation "
                        "radio facilities such as the Square Kilometre Array will sharpen sky "
                        "localization enough for optical follow-up campaigns."
                    ),
                }
            ]
        }
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            captured_calls.append(kwargs)
            return SimpleNamespace(
                text=sample_json,
                candidates=[
                    SimpleNamespace(
                        grounding_metadata=SimpleNamespace(
                            web_search_queries=["site:astronomy.com latest astronomy news"],
                            grounding_chunks=[
                                SimpleNamespace(
                                    web=SimpleNamespace(
                                        uri="https://www.astronomy.com/news/sample-pulsar-timing"
                                    )
                                )
                            ],
                        )
                    )
                ],
            )

    fake_client = SimpleNamespace(models=FakeModels())
    today = datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc)

    results = astro_agent.scan_site_with_gemini(
        client=fake_client,
        model="gemini-4-hydrogen",
        site_url="https://www.astronomy.com/news/",
        today_utc=today,
        lookback_days=2,
        max_articles=3,
        recent_titles=[],
    )

    assert len(captured_calls) == 1
    cfg = captured_calls[0]["config"]
    assert cfg.tools and cfg.tools[0].google_search is not None
    assert cfg.response_mime_type == "application/json"

    assert len(results) == 1
    entry = results[0]
    assert entry["website_source"] == "Astronomy.com"
    assert entry["date"] == "2026-11-19"
    assert len(entry["main_facts"]) == 3
    assert "Square Kilometre Array" in entry["implications_and_development"]


def test_scan_site_falls_back_when_schema_conflicts_with_grounding() -> None:
    call_count = 0

    class SchemaConflictError(Exception):
        code = 400

    class FakeModels:
        def generate_content(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SchemaConflictError(
                    "Invalid argument: response_schema cannot be combined with google_search tool"
                )
            return SimpleNamespace(
                text='```json\n{"articles": []}\n```',
                candidates=[],
            )

    fake_client = SimpleNamespace(models=FakeModels())
    results = astro_agent.scan_site_with_gemini(
        client=fake_client,
        model="legacy-model",
        site_url="https://www.space.com/",
        today_utc=datetime(2026, 11, 19, 7, 0, tzinfo=timezone.utc),
        lookback_days=2,
        max_articles=3,
        recent_titles=[],
    )
    assert call_count == 2
    assert results == []


def test_render_html_includes_all_required_fields_and_escapes_xss() -> None:
    news_data = {
        "last_updated": "2026-11-19T07:05:00Z",
        "model": "gemini-4-hydrogen",
        "entries": [
            {
                "id": "astro-test01",
                "website_source": "Space.com",
                "source_url": "https://www.space.com/",
                "article_url": "javascript:alert(1)",
                "date": "2026-11-19",
                "added_at": "2026-11-19T07:05:00Z",
                "title": "Lunar South Pole Ice Core <script>alert(1)</script>",
                "category": "Planetary Science",
                "resume": "Scientists analyzed new orbital radar scans showing concentrated water-ice layers beneath permanently shadowed craters.",
                "main_facts": [
                    "Radar reflections span depths down to 3 meters.",
                    "Temperature stays below 40 Kelvin year-round.",
                ],
                "implications_and_development": "Provides a prime target for upcoming crewed Artemis sampling drills and in-situ propellant production tests.",
            }
        ],
    }

    html_output = render_html.render_html(
        news_data,
        tracked_sites=["https://www.astronomy.com/news/", "https://www.space.com/"],
    )

    assert "Space.com" in html_output
    assert "2026-11-19" in html_output
    assert "Thursday, November 19, 2026" in html_output
    assert "Resume" in html_output
    assert "Main Facts" in html_output
    assert "Implications &amp; Research Development" in html_output
    assert "Radar reflections span depths down to 3 meters." in html_output
    assert "Artemis sampling drills" in html_output

    # Ensure unsafe HTML and javascript: URLs are neutralized
    assert "<script>alert(1)</script>" not in html_output
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_output
    assert "javascript:alert(1)" not in html_output
