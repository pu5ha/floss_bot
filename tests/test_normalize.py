"""M2: event-type + magnitude inference (pure, offline)."""

from __future__ import annotations

from src.config import Config, EditorConfig, FeedConfig, Secrets
from src.pipeline.normalize import (
    classify_release,
    entry_to_item,
    infer_event_type,
    infer_magnitude,
    parse_cvss,
    parse_grant_magnitude,
    parse_version,
)


def _cfg(tiers=None) -> Config:
    return Config(
        embed_model="all-MiniLM-L6-v2",
        max_per_day=1,
        max_age_hours=36,
        floor=0.35,
        lambda_neg=0.5,
        cluster_sim=0.80,
        weights={},
        type_rank={},
        tier_weight={1: 1.0, 2: 0.6, 3: 0.3},
        feed_window_days=2,
        feeds=[],
        tiers=tiers or {1: ["signal"], 2: ["fdroid"]},
        anitya=[],
        editor=EditorConfig(),
        secrets=Secrets(),
        db_path="/tmp/x.db",
    )


def test_parse_version() -> None:
    assert parse_version("Signal 7.12.1") == (7, 12, 1)
    assert parse_version("Release v2.0") == (2, 0, 0)
    assert parse_version("no version here") is None


def test_classify_release() -> None:
    assert classify_release((2, 0, 0)) == ("major_release", 0.9)
    assert classify_release((2, 4, 0)) == ("minor_release", 0.5)
    assert classify_release((2, 4, 3)) == ("minor_release", 0.2)


def test_advisory_beats_layer() -> None:
    # A security release on the release layer is an advisory, not a plain release.
    et = infer_event_type("Signal 7.1.2 security release CVE-2026-1234", "", 1)
    assert et == "advisory"


def test_event_type_by_layer() -> None:
    assert infer_event_type("Foo 3.0.0", "", 1) == "major_release"
    assert infer_event_type("Foo 3.2.1", "", 1) == "minor_release"
    assert infer_event_type("Announcing Foo", "a new thing", 2) == "launch"
    assert infer_event_type("Some blog post", "musings", 3) == "post"
    assert infer_event_type("A new attack on X", "abstract", 4) == "paper"
    assert infer_event_type("New RFC published", "", 5) == "post"
    assert infer_event_type("Grant awarded", "we funded X", 7) == "grant"
    assert infer_event_type("Fix typo", "", 2, url="https://x/commit/abc") == "commit"


def test_parse_cvss() -> None:
    assert parse_cvss("rated CVSS 9.8 critical") == 0.98
    assert parse_cvss("CVSS:3.1/AV:N/AC:L score 7.5") == 0.75
    assert parse_cvss("no score") is None


def test_infer_magnitude() -> None:
    assert infer_magnitude("advisory", "CVSS 9.8", "") == 0.98
    assert infer_magnitude("advisory", "security fix", "") == 0.8
    assert infer_magnitude("major_release", "v3.0.0", "") == 0.9
    assert infer_magnitude("minor_release", "v3.4.2", "") == 0.2
    assert infer_magnitude("post", "hello", "") == 0.4


def test_parse_grant_magnitude() -> None:
    assert parse_grant_magnitude("awarded €1,200,000 to project") == 0.9
    assert parse_grant_magnitude("a $50k grant") == 0.4
    assert parse_grant_magnitude("no amount stated") == 0.5


def test_entry_to_item_end_to_end() -> None:
    feed = FeedConfig(name="Signal Releases", url="u", layer=1, project="signal")
    entry = {
        "title": "Signal 7.0.0",
        "summary": "<p>A <b>major</b> update.</p>",
        "link": "https://github.com/signalapp/x/releases/tag/v7.0.0",
        "id": "tag:github,2026:Repository/1/v7.0.0",
        "published_parsed": (2026, 7, 18, 0, 0, 0, 0, 0, 0),
    }
    item = entry_to_item(entry, feed, _cfg())
    assert item.event_type == "major_release"
    assert item.magnitude == 0.9
    assert item.tier == 1  # signal is tier 1 in the test config
    assert item.summary == "A major update."  # HTML stripped
    assert item.uid.startswith("feed:")
    assert item.project == "signal"
