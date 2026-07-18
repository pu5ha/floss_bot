"""P2: taste profile — seed parsing (incl. [NEGATIVE]) and vote folding (offline)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src import taste
from src.config import Config, EditorConfig, Secrets
from src.db import connect, init_schema, record_vote, upsert_item
from src.models import Item
from src.taste import fold_votes, load_seed_lines, load_taste_vectors, parse_seed_line


def _cfg() -> Config:
    return Config(
        embed_model="all-MiniLM-L6-v2", max_per_day=1, max_age_hours=36, floor=0.35,
        lambda_neg=0.5, cluster_sim=0.80, weights={}, type_rank={}, tier_weight={},
        feed_window_days=2, feeds=[], tiers={}, anitya=[], editor=EditorConfig(),
        secrets=Secrets(), db_path="/tmp/x.db",
    )


def test_parse_seed_line() -> None:
    assert parse_seed_line("A win — the gist here") == ("A win", "A win. the gist here")
    assert parse_seed_line("# comment") is None
    assert parse_seed_line("   ") is None
    assert parse_seed_line("no dash") == ("no dash", "no dash")


def test_load_seed_lines_splits_on_negative(tmp_path) -> None:
    p = tmp_path / "seeds.txt"
    p.write_text(
        "# header\n"
        "Privacy win — a court ruling\n"
        "Open hardware launch — a new board\n"
        "[NEGATIVE]\n"
        "Minor patch release — routine bump\n"
    )
    positives, negatives = load_seed_lines(p)
    assert [lbl for lbl, _ in positives] == ["Privacy win", "Open hardware launch"]
    assert [lbl for lbl, _ in negatives] == ["Minor patch release"]


def test_fold_votes_uses_item_embedding(tmp_path) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        # A sent item lives in `items` with an embedding; its uid is the story key.
        item = Item(
            uid="feed:abc", source="EFF", layer=3, project=None, title="A privacy win",
            url="u", summary="s", published=datetime(2026, 7, 18, tzinfo=timezone.utc),
            event_type="post", magnitude=0.4, tier=3,
        )
        vec = np.ones(4, dtype=np.float32) / 2.0
        upsert_item(conn, item, vec)
        record_vote(conn, "feed:abc", 1)  # 👍
        conn.commit()

        pos_n, neg_n = fold_votes(conn)
        assert (pos_n, neg_n) == (1, 0)
        pos, labels, neg = load_taste_vectors(conn)
        assert pos.shape[0] == 1  # the 👍 became a positive exemplar
        assert "A privacy win" in labels

        # Flipping to 👎 moves it to negatives (idempotent, no dup).
        record_vote(conn, "feed:abc", -1)
        conn.commit()
        fold_votes(conn)
        pos, _, neg = load_taste_vectors(conn)
        assert pos.shape[0] == 0 and neg.shape[0] == 1
    finally:
        conn.close()


def test_negative_seed_blob_roundtrip(tmp_path, monkeypatch) -> None:
    # rebuild_seeds without hitting a real model: stub embed to a fixed vector.
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        monkeypatch.setattr(
            taste, "embed",
            lambda texts, model=None: np.ones((len(texts), 4), dtype=np.float32) / 2.0,
        )
        seeds = tmp_path / "s.txt"
        seeds.write_text("Good thing — yes\n[NEGATIVE]\nBad thing — no\n")
        pos_n, neg_n = taste.rebuild_seeds(_cfg(), conn, seeds_path=seeds)
        assert (pos_n, neg_n) == (1, 1)
        pos, labels, neg = load_taste_vectors(conn)
        assert pos.shape[0] == 1 and neg.shape[0] == 1
        assert labels == ["Good thing"]
    finally:
        conn.close()
