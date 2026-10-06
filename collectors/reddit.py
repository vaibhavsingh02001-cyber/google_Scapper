"""
collectors/reddit.py — Reddit collector via PRAW (OAuth2).
Collects posts + top-level comments from targeted subreddits
using retrieval-failure search queries.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from collectors.base import BaseCollector
from models.schemas import RawItem, SourcePlatform
import config


SUBREDDITS = ["googlephotos", "android", "photography", "AndroidQuestions", "google"]

SEARCH_QUERIES = [
    "google photos search",
    "can't find photo google photos",
    "photos search broken",
    "google photos can't find",
    "google photos search not working",
    "lost photo google photos",
    "google photos search retrieval",
    "google photos can't search",
]


class RedditCollector(BaseCollector):
    source_name = "reddit"

    def collect(self, limit: int = 300) -> List[RawItem]:
        """
        Collect posts and top-level comments from Reddit using PRAW.
        Searches across configured subreddits with retrieval-failure queries.
        """
        if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
            logger.warning("[reddit] No credentials configured — skipping Reddit collection.")
            return []

        try:
            import praw
        except ImportError:
            logger.error("praw not installed. Run: pip install praw")
            return []

        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
        )

        items: List[RawItem] = []
        collected_at = datetime.now(timezone.utc).isoformat()
        seen_ids: set[str] = set()

        # Items per query per subreddit (distribute budget)
        per_query_limit = max(5, limit // (len(SUBREDDITS) * len(SEARCH_QUERIES)))

        for subreddit_name in SUBREDDITS:
            for query in SEARCH_QUERIES:
                if len(items) >= limit:
                    break
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(
                        query, sort="relevance", time_filter="all", limit=per_query_limit
                    ):
                        if submission.id in seen_ids:
                            continue
                        seen_ids.add(submission.id)

                        # ── Collect post body ──────────────────────────────
                        post_text = " ".join(
                            filter(None, [submission.title, submission.selftext])
                        ).strip()[:2000]

                        if post_text:
                            items.append(
                                RawItem(
                                    id=str(uuid.uuid4()),
                                    source=SourcePlatform.REDDIT,
                                    platform_item_id=submission.id,
                                    text=post_text,
                                    date=datetime.fromtimestamp(
                                        submission.created_utc, tz=timezone.utc
                                    ).isoformat(),
                                    upvotes=max(0, submission.score),
                                    url=f"https://www.reddit.com{submission.permalink}",
                                    collected_at=collected_at,
                                )
                            )

                        # ── Collect top-level comments ─────────────────────
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:5]:
                            if not comment.body or comment.body in ("[deleted]", "[removed]"):
                                continue
                            comment_key = f"c_{comment.id}"
                            if comment_key in seen_ids:
                                continue
                            seen_ids.add(comment_key)

                            items.append(
                                RawItem(
                                    id=str(uuid.uuid4()),
                                    source=SourcePlatform.REDDIT,
                                    platform_item_id=comment.id,
                                    text=comment.body.strip()[:2000],
                                    date=datetime.fromtimestamp(
                                        comment.created_utc, tz=timezone.utc
                                    ).isoformat(),
                                    upvotes=max(0, comment.score),
                                    url=f"https://www.reddit.com{submission.permalink}",
                                    collected_at=collected_at,
                                )
                            )

                    self._sleep()

                except Exception as exc:
                    logger.warning(
                        f"[reddit] Error searching r/{subreddit_name} for '{query}': {exc}"
                    )

        items = self.validate(items)
        self._log_collected(len(items))
        return items
