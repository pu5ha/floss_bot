"""Generic RSS/Atom puller — covers ~all seven source layers (spec §4).

One feed failing must never abort the run: ``fetch_all_feeds`` catches, logs, and
continues. Poll only the last ~``feed_window_days`` per feed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import feedparser

from ..config import Config, FeedConfig
from ..models import Item
from ..pipeline.normalize import entry_to_item
from .base import get_with_retries

log = logging.getLogger("floss-bot")


def parse_feed(text: str, feed: FeedConfig, cfg: Config) -> list[Item]:
    """Parse feed XML into Items (pure; no network)."""
    parsed = feedparser.parse(text)
    return [entry_to_item(e, feed, cfg) for e in parsed.entries]


def fetch_feed(cfg: Config, feed: FeedConfig, days: int | None = None) -> list[Item]:
    """Fetch and normalize one feed, keeping only entries within the window."""
    window = days if days is not None else cfg.feed_window_days
    resp = get_with_retries(feed.url, cfg)
    items = parse_feed(resp.text, feed, cfg)
    cutoff = datetime.now(timezone.utc) - timedelta(days=window)
    return [it for it in items if it.published >= cutoff]


def fetch_all_feeds(cfg: Config, days: int | None = None) -> list[Item]:
    """Fetch every configured feed with per-feed failure isolation."""
    out: list[Item] = []
    for feed in cfg.feeds:
        try:
            items = fetch_feed(cfg, feed, days=days)
            log.info("feed %s: fetched %d item(s)", feed.name, len(items))
            out.extend(items)
        except Exception as exc:  # noqa: BLE001 — resilience is the whole point
            log.warning("feed %s failed: %s", feed.name, exc)
    return out
