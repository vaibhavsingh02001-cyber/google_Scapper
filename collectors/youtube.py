"""
collectors/youtube.py — YouTube comment collector via YouTube Data API v3.
Searches for Google Photos tutorial/review videos and collects comments.
Strictly budget-aware: max ~550 API units (10k daily quota).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from collectors.base import BaseCollector
from models.schemas import RawItem, SourcePlatform
import config


VIDEO_QUERIES = [
    "google photos search how to find photos",
    "google photos search not working tips",
    "google photos review search features",
]

MAX_VIDEOS   = 5    # keep YouTube quota usage low (~500 search units)
MAX_COMMENTS = 100  # per video


class YouTubeCollector(BaseCollector):
    source_name = "youtube"

    def collect(self, limit: int = 150) -> List[RawItem]:
        """
        Search YouTube for relevant videos and collect top-level comments.
        Budget: MAX_VIDEOS × MAX_COMMENTS items maximum.
        """
        if not config.YOUTUBE_API_KEY:
            logger.warning("[youtube] No API key configured — skipping YouTube collection.")
            return []

        try:
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError:
            logger.error("google-api-python-client not installed. Run: pip install google-api-python-client")
            return []

        youtube = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)
        collected_at = datetime.now(timezone.utc).isoformat()
        items: List[RawItem] = []
        seen_video_ids: set[str] = set()

        for query in VIDEO_QUERIES:
            if len(seen_video_ids) >= MAX_VIDEOS:
                break
            try:
                # Search for videos (~100 quota units per call)
                search_response = youtube.search().list(
                    q=query,
                    part="id,snippet",
                    type="video",
                    maxResults=3,
                    relevanceLanguage="en",
                ).execute()

                for search_result in search_response.get("items", []):
                    video_id = search_result["id"].get("videoId")
                    if not video_id or video_id in seen_video_ids:
                        continue
                    if len(seen_video_ids) >= MAX_VIDEOS:
                        break
                    seen_video_ids.add(video_id)

                    video_title = search_result["snippet"].get("title", "")
                    video_url   = f"https://www.youtube.com/watch?v={video_id}"

                    logger.debug(f"[youtube] Fetching comments for: {video_title[:60]}")

                    # Fetch comments for this video (~1 quota unit per page)
                    comment_items = self._fetch_comments(
                        youtube, video_id, video_url, video_title,
                        collected_at, MAX_COMMENTS
                    )
                    items.extend(comment_items)
                    self._sleep()

            except Exception as exc:
                logger.warning(f"[youtube] Error for query '{query}': {exc}")

        items = self.validate(items)
        self._log_collected(len(items))
        return items[:limit]

    def _fetch_comments(
        self,
        youtube,
        video_id: str,
        video_url: str,
        video_title: str,
        collected_at: str,
        max_comments: int,
    ) -> List[RawItem]:
        """Fetch top-level comments for a single video."""
        items: List[RawItem] = []
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_comments, 100),
                textFormat="plainText",
                order="relevance",
            ).execute()

            for thread in response.get("items", []):
                comment = thread["snippet"]["topLevelComment"]["snippet"]
                text = (comment.get("textDisplay") or "").strip()[:2000]
                if not text:
                    continue

                published_at = comment.get("publishedAt")

                items.append(
                    RawItem(
                        id=str(uuid.uuid4()),
                        source=SourcePlatform.YOUTUBE,
                        platform_item_id=thread.get("id"),
                        text=text,
                        date=published_at,
                        upvotes=int(comment.get("likeCount") or 0),
                        url=video_url,
                        collected_at=collected_at,
                    )
                )
        except Exception as exc:
            logger.warning(f"[youtube] Comment fetch failed for video {video_id}: {exc}")

        return items
