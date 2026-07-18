"""P3: calibrate replay is read-only, groups by day, shows relevance picks (offline)."""

from __future__ import annotations

from datetime import datetime, timezone

from src import calibrate
from src.config import Config, EditorConfig, Secrets
from src.db import connect, init_schema
from src.models import Item, Story


def _cfg() -> Config:
    return Config(
        embed_model="all-MiniLM-L6-v2", max_per_day=1, max_age_hours=36, floor=0.35,
        lambda_neg=0.5, cluster_sim=0.80, weights={}, type_rank={}, tier_weight={},
        feed_window_days=2, feeds=[], tiers={}, anitya=[], editor=EditorConfig(),
        secrets=Secrets(), db_path="/tmp/x.db",
    )


def _story(key, when, score) -> Story:
    top = Item(
        uid=key, source="EFF", layer=3, project=None, title=key, url="u", summary="",
        published=when, event_type="post", magnitude=0.4, tier=3,
    )
    st = Story(key=key, items=[top], corroboration=1, top_item=top)
    st.score = score
    st.nearest = "privacy win"
    return st


def test_calibrate_is_read_only(tmp_path, monkeypatch, capsys) -> None:
    cfg = _cfg()
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        d1 = datetime(2026, 7, 18, tzinfo=timezone.utc)
        d2 = datetime(2026, 7, 17, tzinfo=timezone.utc)
        hot = _story("hot", d1, 0.60)   # clears floor -> would send on d1
        cold = _story("cold", d2, 0.10)  # below floor -> quiet on d2

        monkeypatch.setattr(calibrate, "fetch_items", lambda cfg, days=None: [object()])
        monkeypatch.setattr(calibrate, "embed_items", lambda cfg, items: None)
        monkeypatch.setattr(calibrate, "build_stories", lambda cfg, items: [hot, cold])
        # score_relevance normally sets .score; here scores are pre-set, so passthrough.
        monkeypatch.setattr(
            calibrate, "score_relevance", lambda cfg, conn, items, emb, stories: stories
        )
        monkeypatch.setattr(calibrate, "choose_headline",
                            lambda cfg, sl: (sl[0].__setattr__("headline", "H") or sl[0]))
        monkeypatch.setattr(calibrate, "summarize_story", lambda cfg, story: "a summary")

        calibrate.run_calibrate(cfg, conn, days=14)

        for table in ("items", "stories", "sent"):
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            assert n == 0, f"{table} should be untouched"
        out = capsys.readouterr().out
        assert "2026-07-18" in out and "2026-07-17" in out
        assert "WOULD SEND" in out and "QUIET" in out
        assert "floor sweep" in out
    finally:
        conn.close()
