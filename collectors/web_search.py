"""
collectors/web_search.py — General web search collector.
Uses DuckDuckGo's unofficial JSON API (no key required) as a primary source,
with optional SerpAPI fallback if SERP_API_KEY is configured.
Fetches snippets for queries targeting Google Photos retrieval complaints.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from collectors.base import BaseCollector
from models.schemas import RawItem, SourcePlatform
import config


SEARCH_QUERIES = [
    "google photos can't find photo complaint",
    "google photos search not working reddit",
    "site:reddit.com google photos search failed",
    "site:quora.com google photos search retrieval",
    '"google photos" "can\'t find" photo',
    '"google photos" search "not working" 2023 OR 2024',
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class WebSearchCollector(BaseCollector):
    source_name = "web"

    def collect(self, limit: int = 100) -> List[RawItem]:
        """
        Collect search result snippets for retrieval-failure queries.
        Each snippet is treated as a lightweight feedback signal.
        """
        logger.info(f"[web] Running web search collection (limit={limit})...")
        collected_at = datetime.now(timezone.utc).isoformat()
        items: List[RawItem] = []

        for query in SEARCH_QUERIES:
            if len(items) >= limit:
                break
            try:
                results = (
                    self._search_serpapi(query)
                    if config.SERP_API_KEY
                    else self._search_ddg(query)
                )
                for result in results:
                    snippet = (result.get("snippet") or result.get("body") or "").strip()
                    url     = result.get("link") or result.get("href") or ""
                    title   = result.get("title") or ""

                    # Combine title + snippet for richer context
                    combined = f"{title}. {snippet}".strip()[:2000]
                    if not combined or len(combined) < 40:
                        continue

                    items.append(
                        RawItem(
                            id=str(uuid.uuid4()),
                            source=SourcePlatform.WEB,
                            platform_item_id=url,
                            text=combined,
                            date=None,
                            upvotes=0,
                            url=url,
                            collected_at=collected_at,
                        )
                    )
                self._sleep()

            except Exception as exc:
                logger.warning(f"[web] Error for query '{query}': {exc}")

        items = self.validate(items)
        self._log_collected(len(items))
        return items[:limit]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _search_ddg(self, query: str) -> List[dict]:
        """DuckDuckGo Instant Answer API (no key required, rate-limited)."""
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[dict] = []

        # RelatedTopics contain the most relevant snippets
        for topic in data.get("RelatedTopics", [])[:10]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "snippet": topic.get("Text", ""),
                    "link":    topic.get("FirstURL", ""),
                    "title":   "",
                })

        # Abstract text if available
        if data.get("AbstractText"):
            results.append({
                "snippet": data["AbstractText"],
                "link":    data.get("AbstractURL", ""),
                "title":   data.get("Heading", ""),
            })

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _search_serpapi(self, query: str) -> List[dict]:
        """SerpAPI Google search (requires SERP_API_KEY)."""
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": config.SERP_API_KEY,
                "engine": "google",
                "num": 10,
                "hl": "en",
            },
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("organic_results", [])
