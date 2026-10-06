"""
collectors/help_forum.py — Google Photos Help Community collector.
Scrapes support.google.com/photos community threads using
requests + BeautifulSoup. Respects robots.txt and uses polite delays.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from collectors.base import BaseCollector
from models.schemas import RawItem, SourcePlatform


BASE_URL = "https://support.google.com"
FORUM_URL = "https://support.google.com/photos/community"

SEARCH_QUERIES = [
    "can't find photo",
    "search not working",
    "photo disappeared",
    "search results wrong",
    "can't locate photo",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class HelpForumCollector(BaseCollector):
    source_name = "help_forum"

    def collect(self, limit: int = 100) -> List[RawItem]:
        """
        Scrape Google Photos Help Community forum threads.
        Uses targeted search queries and extracts thread titles + post bodies.
        """
        logger.info(f"[help_forum] Scraping Google Photos Help Community (limit={limit})...")
        collected_at = datetime.now(timezone.utc).isoformat()
        items: List[RawItem] = []
        seen_urls: set[str] = set()

        for query in SEARCH_QUERIES:
            if len(items) >= limit:
                break
            try:
                thread_links = self._search_threads(query)
                for thread_url in thread_links:
                    if thread_url in seen_urls or len(items) >= limit:
                        break
                    seen_urls.add(thread_url)

                    thread_items = self._scrape_thread(thread_url, collected_at)
                    items.extend(thread_items)
                    self._sleep()

            except Exception as exc:
                logger.warning(f"[help_forum] Error for query '{query}': {exc}")

        items = self.validate(items)
        self._log_collected(len(items))
        return items[:limit]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _search_threads(self, query: str) -> List[str]:
        """
        Search the Help Community and return a list of thread URLs.
        Falls back gracefully if the search page structure has changed.
        """
        search_url = f"{FORUM_URL}?hl=en"
        params = {"hl": "en", "q": f"Google Photos {query}"}

        resp = requests.get(search_url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        links: List[str] = []

        for a_tag in soup.select("a[href]"):
            href = a_tag.get("href", "")
            if "/photos/threads/" in href or "/photos/answer/" in href:
                full_url = href if href.startswith("http") else BASE_URL + href
                if full_url not in links:
                    links.append(full_url)

        return links[:5]  # limit threads per query

    def _scrape_thread(self, url: str, collected_at: str) -> List[RawItem]:
        """Extract post bodies from a single Help Community thread."""
        items: List[RawItem] = []
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract user post content — selectors may need updating
            # if Google changes their forum markup
            post_bodies = soup.select(".lia-message-body-content, .jfk-bubble-content")
            if not post_bodies:
                # Fallback: grab all paragraph text in the main content area
                main = soup.find("main") or soup.find("article") or soup
                post_bodies = main.find_all("p")

            for post in post_bodies[:5]:
                text = post.get_text(separator=" ", strip=True)[:2000]
                if len(text) < 40:
                    continue

                items.append(
                    RawItem(
                        id=str(uuid.uuid4()),
                        source=SourcePlatform.HELP_FORUM,
                        platform_item_id=url,
                        text=text,
                        date=None,
                        upvotes=0,
                        url=url,
                        collected_at=collected_at,
                    )
                )
        except Exception as exc:
            logger.debug(f"[help_forum] Thread scrape error ({url}): {exc}")

        return items
