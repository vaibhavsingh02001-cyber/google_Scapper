"""
collectors/play_store.py — Google Play Store review collector.
Uses the `google-play-scraper` library (no auth required).
Target app: com.google.android.apps.photos
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from collectors.base import BaseCollector
from models.schemas import RawItem, SourcePlatform


class PlayStoreCollector(BaseCollector):
    source_name = "play_store"
    APP_ID      = "com.google.android.apps.photos"

    def collect(self, limit: int = 300) -> List[RawItem]:
        """
        Fetch up to `limit` reviews from the Google Play Store,
        sorted by newest first.
        """
        try:
            from google_play_scraper import reviews, Sort
        except ImportError:
            logger.error("google-play-scraper not installed. Run: pip install google-play-scraper")
            return []

        logger.info(f"[play_store] Fetching up to {limit} reviews for {self.APP_ID}...")

        raw_reviews, _ = self._fetch_with_retry(limit)
        items: List[RawItem] = []

        collected_at = datetime.now(timezone.utc).isoformat()

        for r in raw_reviews:
            text = (r.get("content") or "").strip()
            if not text:
                continue

            # Truncate to max length
            text = text[:2000]

            date_val = r.get("at")
            date_str = date_val.isoformat() if date_val else None

            items.append(
                RawItem(
                    id=str(uuid.uuid4()),
                    source=SourcePlatform.PLAY_STORE,
                    platform_item_id=r.get("reviewId"),
                    text=text,
                    date=date_str,
                    upvotes=int(r.get("thumbsUpCount") or 0),
                    url=f"https://play.google.com/store/apps/details?id={self.APP_ID}",
                    collected_at=collected_at,
                )
            )

        items = self.validate(items)
        self._log_collected(len(items))
        return items

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _fetch_with_retry(self, limit: int):
        from google_play_scraper import reviews, Sort
        return reviews(
            self.APP_ID,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=limit,
            filter_score_with=None,
        )
