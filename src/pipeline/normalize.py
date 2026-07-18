"""Feed entry -> Item, inferring event_type and magnitude (spec §4).

The inference helpers are pure and keyword/semver-driven so they can be unit
tested offline against fixtures.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from ..config import Config, FeedConfig
from ..models import Item

# Keyword sets (matched case-insensitively against title + summary).
_ADVISORY_KW = (
    "cve-",
    "advisory",
    "security release",
    "security update",
    "vulnerability",
    "security fix",
    "patched a",
    "0-day",
    "zero-day",
)
# Matched in the TITLE only (summaries of reviews/press trip loose triggers like
# "now available"). Kept to strong, intentional launch language.
_LAUNCH_KW = (
    "announcing",
    "introducing",
    "we're releasing",
    "we are releasing",
    "unveil",
    "debut",
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# A semver-ish version anywhere in the title, e.g. "Signal 7.12.1", "v2.0", "1.0".
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)(?:\.(\d+))?\b")
# CVSS *version* token (e.g. "CVSS:3.1") — stripped before reading the base score
# so it isn't mistaken for the score itself.
_CVSS_VERSION_RE = re.compile(r"cvss:\s*v?[234]\.\d", re.IGNORECASE)
# Prefer an explicit "(base) score N", else fall back to a number near "CVSS".
_CVSS_SCORE_RE = re.compile(
    r"(?:base\s+)?score[^0-9]{0,12}(\d{1,2}(?:\.\d)?)", re.IGNORECASE
)
_CVSS_NEAR_RE = re.compile(r"cvss[^0-9]{0,24}?(\d{1,2}(?:\.\d)?)", re.IGNORECASE)
# A monetary amount, e.g. "€1,200,000", "$500k", "1.5M".
_MONEY_RE = re.compile(
    r"[€$£]\s?([\d,]+(?:\.\d+)?)\s?([kmb])?|\b([\d,]+(?:\.\d+)?)\s?([kmb])\b",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


def _has(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def parse_version(title: str) -> tuple[int, int, int] | None:
    """Extract the first semver-ish version from a title as (major, minor, patch)."""
    m = _VERSION_RE.search(title)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3)) if m.group(3) else 0
    return (major, minor, patch)


def classify_release(version: tuple[int, int, int]) -> tuple[str, float]:
    """Classify a version into (event_type, magnitude).

    Without a prior version to diff against, use the standard heuristic: x.0.0 is
    a major, x.y.0 is a minor, x.y.z is a patch. Patches keep the ``minor_release``
    event type (the enum has no patch type) but a low magnitude.
    """
    major, minor, patch = version
    if minor == 0 and patch == 0:
        return ("major_release", 0.9)
    if patch == 0:
        return ("minor_release", 0.5)
    return ("minor_release", 0.2)


def _is_commit(url: str) -> bool:
    return "/commit/" in url or "/commits/" in url


def parse_cvss(text: str) -> float | None:
    """Return a 0..1 magnitude from a CVSS base score if one is present.

    The CVSS version token (e.g. ``CVSS:3.1``) is stripped first so it isn't read
    as the score; an explicit "score N" wins over a bare number near "CVSS".
    """
    stripped = _CVSS_VERSION_RE.sub(" CVSS ", text)
    m = _CVSS_SCORE_RE.search(stripped) or _CVSS_NEAR_RE.search(stripped)
    if not m:
        return None
    try:
        score = float(m.group(1))
    except ValueError:
        return None
    if 0.0 <= score <= 10.0:
        return round(score / 10.0, 3)
    return None


def parse_grant_magnitude(text: str) -> float:
    """Scale a grant/funding amount to 0..1 (default 0.5 if none parseable)."""
    m = _MONEY_RE.search(text)
    if not m:
        return 0.5
    num = m.group(1) or m.group(3) or ""
    unit = (m.group(2) or m.group(4) or "").lower()
    try:
        amount = float(num.replace(",", ""))
    except ValueError:
        return 0.5
    mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get(unit, 1.0)
    amount *= mult
    if amount >= 1_000_000:
        return 0.9
    if amount >= 100_000:
        return 0.6
    if amount >= 10_000:
        return 0.4
    return 0.3


def infer_event_type(title: str, summary: str, layer: int, url: str = "") -> str:
    """Infer the event type from feed layer + keywords (spec §4)."""
    blob = f"{title} {summary}".lower()
    title_l = title.lower()  # launch language must be in the title, not the body
    if _is_commit(url):
        return "commit"
    if _has(blob, _ADVISORY_KW):
        return "advisory"
    if layer == 1:  # release layer
        if parse_version(title):
            return classify_release(parse_version(title))[0]  # type: ignore[arg-type]
        if _has(title_l, _LAUNCH_KW):
            return "launch"
        return "minor_release"
    if layer == 4:  # research
        return "paper"
    if layer == 7:  # funding
        return "grant"
    if _has(title_l, _LAUNCH_KW):
        return "launch"
    return "post"  # layers 2/3/5/6 default


def infer_magnitude(event_type: str, title: str, summary: str) -> float:
    """Type-specific 0..1 magnitude (spec §4)."""
    blob = f"{title} {summary}"
    if event_type == "advisory":
        cvss = parse_cvss(blob)
        return cvss if cvss is not None else 0.8
    if event_type == "major_release":
        return 0.9
    if event_type == "minor_release":
        # Patch vs minor is encoded in the version; recover it if possible.
        version = parse_version(title)
        if version:
            return classify_release(version)[1]
        return 0.5
    if event_type == "launch":
        return 0.8
    if event_type == "grant":
        return parse_grant_magnitude(blob)
    if event_type == "paper":
        return 0.6
    if event_type == "commit":
        return 0.1
    return 0.4  # post default


def _uid(feed_name: str, entry_id: str) -> str:
    h = hashlib.sha1(f"{feed_name}|{entry_id}".encode()).hexdigest()[:16]
    return f"feed:{h}"


def _resolve_tier(cfg: Config, feed: FeedConfig) -> int:
    if feed.tier is not None:
        return feed.tier
    return cfg.project_tier(feed.project)


def entry_to_item(entry: dict, feed: FeedConfig, cfg: Config) -> Item:
    """Normalize a single feedparser entry (dict-like) into an Item."""
    title = clean(entry.get("title", ""))
    summary = clean(entry.get("summary", ""))
    url = entry.get("link", "") or ""

    published = None
    for key in ("published_parsed", "updated_parsed"):
        tm = entry.get(key)
        if tm:
            published = datetime(*tm[:6], tzinfo=timezone.utc)
            break
    if published is None:
        published = datetime.now(timezone.utc)

    event_type = infer_event_type(title, summary, feed.layer, url)
    magnitude = infer_magnitude(event_type, title, summary)
    entry_id = entry.get("id") or url or title

    return Item(
        uid=_uid(feed.name, entry_id),
        source=feed.name,
        layer=feed.layer,
        project=feed.project,
        title=title,
        url=url,
        summary=summary[:1000],
        published=published,
        event_type=event_type,
        magnitude=magnitude,
        tier=_resolve_tier(cfg, feed),
    )
