"""
collectors/base.py — Abstract base class for all data collectors.
Every collector must implement `collect()` and return List[RawItem].
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import List

from loguru import logger
from models.schemas import RawItem
import config


class BaseCollector(ABC):
    """
    Abstract base collector. Subclasses implement `collect()`.
    Shared helpers: rate-limit delay, quality validation, logging.
    """
    source_name: str = "base"

    @abstractmethod
    def collect(self, limit: int = 200) -> List[RawItem]:
        """
        Fetch items from the source.
        Must respect platform ToS and rate limits.
        Returns a list of RawItem objects (not yet deduplicated).
        """
        ...

    def validate(self, items: List[RawItem]) -> List[RawItem]:
        """
        Quality filter — remove items that are too short, empty,
        mostly non-alphabetic (emoji-only/spam), or pure links.
        """
        filtered: List[RawItem] = []
        discarded = 0

        for item in items:
            text = item.text.strip()

            # Minimum length guard
            if len(text) < config.MIN_TEXT_LENGTH:
                discarded += 1
                continue

            # Alpha-character density guard (catches emoji-spam and URLs)
            alpha_chars = sum(1 for c in text if c.isalpha())
            if len(text) > 0 and (alpha_chars / len(text)) < 0.4:
                discarded += 1
                continue

            filtered.append(item)

        if discarded:
            logger.debug(
                f"[{self.source_name}] Discarded {discarded} low-quality items "
                f"({len(filtered)} kept)"
            )
        return filtered

    @staticmethod
    def _sleep() -> None:
        """Randomised polite delay between requests."""
        delay = random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX)
        time.sleep(delay)

    def _log_collected(self, n: int) -> None:
        logger.info(f"[{self.source_name}] Collected {n} raw items")
