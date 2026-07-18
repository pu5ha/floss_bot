"""M8: quiet-day floor, idempotent claim-before-send, Telegram formatting (offline)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import Config, EditorConfig, Secrets
from src.db import connect, init_schema, sent_today, today_utc
from src.models import Item, Story
from src.pipeline import orchestrate
from src.pipeline.orchestrate import Plan, send_plan
from src.telegram import build_keyboard, build_message

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _cfg() -> Config:
    return Config(
        embed_model="all-MiniLM-L6-v2", max_per_day=1, max_age_hours=36, floor=0.35,
        lambda_neg=0.5, cluster_sim=0.80, weights={}, type_rank={}, tier_weight={},
        feed_window_days=2, feeds=[], tiers={}, anitya=[], editor=EditorConfig(enabled=False),
        secrets=Secrets(telegram_bot_token="t", telegram_chat_id="c"), db_path="/tmp/x.db",
    )


def _story(key="k1", score=0.7, headline="A headline", corrob=1) -> Story:
    top = Item(
        uid=key, source="Anitya", layer=1, project="signal", title="Signal 7.0.0",
        url="https://x/y", summary="s", published=NOW, event_type="major_release",
        magnitude=0.9, tier=1,
    )
    st = Story(key=key, items=[top], corroboration=corrob, top_item=top, headline=headline)
    st.score = score
    return st


def _plan(story, quiet=False) -> Plan:
    return Plan(
        shortlist=[story], sendable=[] if quiet else [story],
        best_score=story.score, quiet=quiet, reason="test",
    )


def test_build_message_and_keyboard() -> None:
    st = _story(corrob=3)
    msg = build_message(st, "A three sentence summary.")
    assert "📰 <b>A headline</b>" in msg
    assert "<i>Anitya +2 more</i>" in msg
    assert "A three sentence summary." in msg
    assert "https://x/y" in msg
    kb = build_keyboard("k1")
    assert kb["inline_keyboard"][0][0]["callback_data"] == "up:k1"
    assert kb["inline_keyboard"][0][1]["callback_data"] == "down:k1"


def test_send_plan_idempotent(tmp_path, monkeypatch) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        cfg = _cfg()
        sends: list[str] = []
        monkeypatch.setattr(orchestrate, "summarize_story", lambda cfg, story: "sum")
        monkeypatch.setattr(
            orchestrate, "send_headline",
            lambda cfg, story, summary: (sends.append(story.key) or 111),
        )

        story = _story()
        first = send_plan(cfg, conn, _plan(story))
        assert first is not None and first.key == "k1"
        assert sends == ["k1"]
        assert sent_today(conn, today_utc()) == 1

        # Second send of the same key must NOT re-send (claim already exists).
        second = send_plan(cfg, conn, _plan(_story()))
        assert second is None
        assert sends == ["k1"]  # unchanged

        # message id was recorded.
        row = conn.execute("SELECT telegram_message_id FROM sent WHERE key='k1'").fetchone()
        assert row["telegram_message_id"] == 111
    finally:
        conn.close()


def test_send_plan_quiet_sends_nothing(tmp_path, monkeypatch) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        monkeypatch.setattr(
            orchestrate, "send_headline",
            lambda cfg, story: (_ for _ in ()).throw(AssertionError("must not send")),
        )
        assert send_plan(_cfg(), conn, _plan(_story(), quiet=True)) is None
        n = conn.execute("SELECT COUNT(*) AS n FROM sent").fetchone()["n"]
        assert n == 0
    finally:
        conn.close()
