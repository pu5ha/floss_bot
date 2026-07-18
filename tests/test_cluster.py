"""M4: clustering + corroboration (offline, hand-built embeddings)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.models import Item
from src.pipeline.cluster import authority, cluster_items, normalize_url

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _item(uid, source, title, *, project=None, event_type="post", url="", when=NOW) -> Item:
    return Item(
        uid=uid,
        source=source,
        layer=2,
        project=project,
        title=title,
        url=url,
        summary="",
        published=when,
        event_type=event_type,
        magnitude=0.4,
        tier=1 if project else 3,
    )


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_normalize_url() -> None:
    assert normalize_url("https://www.Example.org/a/") == "//example.org/a"
    assert normalize_url("http://example.org/a?x=1#f") == "//example.org/a"
    assert normalize_url("") == ""


def test_authority_order() -> None:
    adv = _item("1", "s", "x", event_type="advisory")
    rel = _item("2", "s", "x", event_type="major_release", project="p")
    blog = _item("3", "s", "x", event_type="post", project="p")
    news = _item("4", "s", "x", event_type="post")
    commit = _item("5", "s", "x", event_type="commit")
    assert authority(adv) > authority(rel) > authority(blog) > authority(news) > authority(commit)


def test_cluster_by_project_version() -> None:
    # GitHub release + Anitya release of the same project+version -> one story.
    a = _item("gh:1", "GitHub", "Signal 7.0.0", project="signal", event_type="major_release")
    b = _item("anitya:1", "Anitya", "signal 7.0.0 released", project="signal",
              event_type="major_release")
    # Distinct embeddings so ONLY the hard key merges them.
    emb = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    stories = cluster_items([a, b], emb, cluster_sim=0.8)
    assert len(stories) == 1
    assert stories[0].corroboration == 2


def test_cluster_by_embedding_within_window() -> None:
    a = _item("n1", "EFF", "Court strikes down surveillance law")
    b = _item("n2", "Access Now", "Surveillance law struck down by court")
    emb = np.vstack([_unit([1, 0.05, 0]), _unit([0.98, 0.06, 0])])  # cosine ~0.999
    stories = cluster_items([a, b], emb, cluster_sim=0.8)
    assert len(stories) == 1
    assert stories[0].corroboration == 2


def test_no_merge_outside_window() -> None:
    a = _item("n1", "EFF", "Same headline", when=NOW)
    b = _item("n2", "Access Now", "Same headline", when=NOW - timedelta(hours=72))
    emb = np.vstack([_unit([1, 0, 0]), _unit([1, 0, 0])])  # identical vectors
    stories = cluster_items([a, b], emb, cluster_sim=0.8)
    assert len(stories) == 2  # 72h apart -> not merged


def test_top_item_is_most_authoritative() -> None:
    news = _item("n1", "Phoronix", "OpenSSL 4.0 out", event_type="post")
    adv = _item("a1", "OpenSSL", "OpenSSL 4.0 security advisory", project="openssl",
                event_type="advisory")
    emb = np.vstack([_unit([1, 0, 0]), _unit([0.99, 0.02, 0])])
    stories = cluster_items([news, adv], emb, cluster_sim=0.8)
    assert len(stories) == 1
    assert stories[0].top_item.uid == "a1"
    assert stories[0].key == "a1"
