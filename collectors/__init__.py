"""collectors/__init__.py — Collector registry."""
from collectors.play_store  import PlayStoreCollector
from collectors.app_store   import AppStoreCollector
from collectors.reddit      import RedditCollector
from collectors.youtube     import YouTubeCollector
from collectors.help_forum  import HelpForumCollector
from collectors.web_search  import WebSearchCollector

ALL_COLLECTORS = {
    "play_store":  PlayStoreCollector,
    "app_store":   AppStoreCollector,
    "reddit":      RedditCollector,
    "youtube":     YouTubeCollector,
    "help_forum":  HelpForumCollector,
    "web":         WebSearchCollector,
}

__all__ = [
    "PlayStoreCollector",
    "AppStoreCollector",
    "RedditCollector",
    "YouTubeCollector",
    "HelpForumCollector",
    "WebSearchCollector",
    "ALL_COLLECTORS",
]
