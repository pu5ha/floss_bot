"""Cluster Items describing the SAME story across feeds (spec §5).

Two merge rules, cheapest first:
  1. same outbound URL, or same project+version -> same story (strong, cheap);
  2. else title-embedding cosine > ``cluster_sim`` within a 48h window.

``corroboration`` = count of distinct sources in a cluster; ``top_item`` = the
most authoritative item (advisory > official release/blog > news > aggregator/commit).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from ..models import Item, Story
from .normalize import parse_version

log = logging.getLogger("floss-bot")

_WINDOW = timedelta(hours=48)


def normalize_url(url: str) -> str:
    """Canonicalize a URL for equality: drop scheme, query, fragment, trailing slash."""
    if not url:
        return ""
    parts = urlsplit(url.strip().lower())
    netloc = parts.netloc[4:] if parts.netloc.startswith("www.") else parts.netloc
    path = parts.path.rstrip("/")
    return urlunsplit(("", netloc, path, "", ""))


def _hard_key(item: Item) -> str | None:
    """A strong identity key: project+version if available, else the normalized URL."""
    if item.project:
        version = parse_version(item.title)
        if version:
            return f"pv:{item.project}:{version[0]}.{version[1]}.{version[2]}"
    url = normalize_url(item.url)
    return f"url:{url}" if url else None


def authority(item: Item) -> int:
    """Rank an item's authority as a canonical source (higher = more authoritative)."""
    if item.event_type == "advisory":
        return 100
    if item.event_type in ("major_release", "minor_release"):
        return 80
    if item.event_type == "launch":
        return 70
    if item.event_type == "paper":
        return 55
    if item.event_type == "grant":
        return 50
    if item.event_type == "commit":
        return 10
    # a plain post: an official project blog outranks an aggregator/news feed.
    return 60 if item.project else 40


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _pick_top(items: list[Item]) -> Item:
    """Most authoritative item; tie-break to the earliest published (the origin)."""
    return max(
        items,
        key=lambda it: (authority(it), 1 if it.project else 0, -it.published.timestamp()),
    )


def cluster_items(
    items: list[Item], embeddings: np.ndarray, cluster_sim: float = 0.80
) -> list[Story]:
    """Group items into Stories. ``embeddings`` is row-aligned to ``items`` and
    L2-normalized (so cosine == dot product).
    """
    n = len(items)
    if n == 0:
        return []

    uf = _UnionFind(n)

    # Rule 1: strong identity keys (project+version or exact URL).
    by_key: dict[str, int] = {}
    for i, it in enumerate(items):
        key = _hard_key(it)
        if key is None:
            continue
        if key in by_key:
            uf.union(by_key[key], i)
        else:
            by_key[key] = i

    # Rule 2: title-embedding cosine within a 48h window.
    if embeddings.size:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(items[i].published - items[j].published) > _WINDOW:
                    continue
                if float(embeddings[i] @ embeddings[j]) > cluster_sim:
                    uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    stories: list[Story] = []
    for members in groups.values():
        member_items = [items[i] for i in members]
        top = _pick_top(member_items)
        corroboration = len({it.source for it in member_items})
        stories.append(
            Story(
                key=top.uid,  # stable while the canonical item persists
                items=member_items,
                corroboration=corroboration,
                top_item=top,
            )
        )
    log.info("clustered %d item(s) into %d stor(y/ies)", n, len(stories))
    return stories
