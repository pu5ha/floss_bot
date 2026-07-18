"""P2: relevance scoring + thin rule layer (pure, offline)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.models import Item, Story
from src.pipeline.score import (
    build_shortlist,
    is_excluded,
    is_forced,
    score_matrix,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def _story(key, event_type="post", *, tier=3, score=0.0) -> Story:
    top = Item(
        uid=key, source="s", layer=2, project=None, title=key, url="u", summary="",
        published=NOW, event_type=event_type, magnitude=0.4, tier=tier,
    )
    st = Story(key=key, items=[top], corroboration=1, top_item=top)
    st.score = score
    return st


def test_score_matrix_relevance() -> None:
    # Story A aligns with a positive seed; B aligns with a negative seed.
    pos = np.vstack([_unit([1, 0, 0])])
    neg = np.vstack([_unit([0, 1, 0])])
    emb = np.vstack([_unit([1, 0.1, 0]), _unit([0.1, 1, 0])])
    out = score_matrix(emb, pos, ["privacy win"], neg, lambda_neg=0.5)
    assert out[0].score > out[1].score  # A more relevant than B
    assert out[0].nearest == "privacy win"


def test_score_matrix_lambda_penalizes_disliked() -> None:
    pos = np.vstack([_unit([1, 0, 0])])
    neg = np.vstack([_unit([1, 0, 0])])  # same direction as pos
    emb = np.vstack([_unit([1, 0, 0])])
    hi = score_matrix(emb, pos, ["x"], neg, lambda_neg=0.0)[0].score
    lo = score_matrix(emb, pos, ["x"], neg, lambda_neg=1.0)[0].score
    assert hi > lo  # dislike penalty lowers the score


def test_empty_taste_returns_zeros() -> None:
    emb = np.vstack([_unit([1, 0, 0])])
    empty = np.empty((0, 0), dtype=np.float32)
    out = score_matrix(emb, empty, [], empty, lambda_neg=0.5)
    assert out[0].score == 0.0


def test_is_excluded_drops_version_bumps() -> None:
    assert is_excluded(_story("a", "minor_release"))
    assert is_excluded(_story("b", "commit"))
    assert not is_excluded(_story("c", "post"))
    assert not is_excluded(_story("d", "advisory"))
    assert not is_excluded(_story("e", "launch"))


def test_is_forced_only_flagship_advisories() -> None:
    assert is_forced(_story("a", "advisory", tier=1))
    assert is_forced(_story("b", "advisory", tier=2))
    assert not is_forced(_story("c", "advisory", tier=3))
    assert not is_forced(_story("d", "post", tier=1))


def test_build_shortlist_floor_plus_forced() -> None:
    stories = [
        _story("hi", "post", score=0.60),
        _story("mid", "post", score=0.40),
        _story("lo", "post", score=0.10),
        _story("adv", "advisory", tier=1, score=0.05),  # forced despite low score
    ]
    stories.sort(key=lambda s: s.score, reverse=True)
    out = build_shortlist(stories, floor=0.35, size=5)
    keys = {s.key for s in out}
    assert keys == {"hi", "mid", "adv"}  # lo dropped (below floor), adv forced
