"""
collectors/app_store.py — Apple App Store review collector.
Uses the `app-store-scraper` library (no auth required).
Target app: Google Photos (App ID 962194608)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from collectors.base import BaseCollector
from models.schemas import RawItem, SourcePlatform


class AppStoreCollector(BaseCollector):
    source_name = "app_store"
    APP_ID      = "962194608"     # Google Photos on App Store
    APP_NAME    = "google-photos"

    def collect(self, limit: int = 200) -> List[RawItem]:
        """Fetch up to `limit` App Store reviews (US store, English)."""
        try:
            from app_store_scraper import AppStore
        except ImportError:
            logger.error("app-store-scraper not installed. Run: pip install app-store-scraper")
            return []

        logger.info(f"[app_store] Fetching up to {limit} reviews (App ID: {self.APP_ID})...")

        app = AppStore(country="us", app_name=self.APP_NAME, app_id=self.APP_ID)
        try:
            app.review(how_many=limit)
        except Exception as exc:
            logger.error(f"[app_store] Fetch failed: {exc}")
            return []

        collected_at = datetime.now(timezone.utc).isoformat()
        items: List[RawItem] = []

        for r in (app.reviews or []):
            text = (r.get("review") or "").strip()
            if not text:
                continue
            text = text[:2000]

            date_val = r.get("date")
            date_str = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val) if date_val else None

            items.append(
                RawItem(
                    id=str(uuid.uuid4()),
                    source=SourcePlatform.APP_STORE,
                    platform_item_id=str(r.get("isEdited", "")),
                    text=text,
                    date=date_str,
                    upvotes=0,   # App Store does not expose vote counts via scraper
                    url=f"https://apps.apple.com/us/app/google-photos/id{self.APP_ID}",
                    collected_at=collected_at,
                )
            )

        items = self.validate(items)
        self._log_collected(len(items))
        return items
