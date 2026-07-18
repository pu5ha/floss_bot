"""Core dataclasses: an ``Item`` (one feed entry) and a ``Story`` (a cluster)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Item:
    """One normalized feed entry from one source."""

    uid: str  # stable: hash(source + entry id/link)
    source: str  # feed name
    layer: int  # 1..7 (release, community-news, rights, research, standards, conf, funding)
    project: str | None  # mapped project if known (e.g. "signal")
    title: str
    url: str  # outbound link (used for clustering)
    summary: str
    published: datetime  # UTC
    event_type: str  # advisory|major_release|launch|minor_release|paper|grant|post|commit
    magnitude: float  # type-specific 0..1 (CVSS, semver delta, grant size, venue rank...)
    tier: int = 3  # project importance tier 1/2/3 (3 = unknown/low)


@dataclass
class Story:
    """A cluster of Items describing the same real-world event."""

    key: str
    items: list[Item] = field(default_factory=list)
    corroboration: int = 1  # distinct sources covering it
    score: float = 0.0  # relevance to the taste profile
    nearest: str = ""  # label of the closest taste seed (shown for transparency)
    top_item: Item | None = None  # the canonical/most-authoritative item
    headline: str | None = None  # filled by the editor for the chosen story

    @property
    def tier(self) -> int:
        """Best (lowest-numbered) tier across the clustered items."""
        return min((it.tier for it in self.items), default=3)
