"""P4: vote callback handling + offset advance (offline; Telegram mocked)."""

from __future__ import annotations

from src import votes
from src.config import Config, EditorConfig, Secrets
from src.db import connect, init_schema
from src.votes import drain_once


def _cfg() -> Config:
    return Config(
        embed_model="all-MiniLM-L6-v2", max_per_day=1, max_age_hours=36, floor=0.35,
        lambda_neg=0.5, cluster_sim=0.80, weights={}, type_rank={}, tier_weight={},
        feed_window_days=2, feeds=[], tiers={}, anitya=[], editor=EditorConfig(),
        secrets=Secrets(telegram_bot_token="t", telegram_chat_id="c"), db_path="/tmp/x.db",
    )


def _mute_telegram(monkeypatch) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(votes.telegram, "answer_callback",
                        lambda cfg, cqid, text="": events.append(("answer", cqid)))
    monkeypatch.setattr(votes.telegram, "mark_message_voted",
                        lambda cfg, chat, mid, label: events.append(("mark", label)))
    return events


def _update(update_id, key, action):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cq{update_id}", "data": f"{action}:{key}",
            "message": {"chat": {"id": 1}, "message_id": update_id},
        },
    }


def test_drain_records_votes_and_advances_offset(tmp_path, monkeypatch) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        cfg = _cfg()
        events = _mute_telegram(monkeypatch)
        monkeypatch.setattr(
            votes.telegram, "get_updates",
            lambda cfg, offset, timeout=0: [
                _update(10, "feed:aaa", "up"),
                _update(11, "feed:bbb", "down"),
            ],
        )
        n = drain_once(cfg, conn, timeout=0)
        assert n == 2
        rows = {r["key"]: r["vote"] for r in conn.execute("SELECT key, vote FROM votes")}
        assert rows == {"feed:aaa": 1, "feed:bbb": -1}
        # offset advanced past the last update_id
        off = conn.execute(
            "SELECT last_cursor FROM source_state WHERE source='telegram'"
        ).fetchone()["last_cursor"]
        assert int(off) == 12
        assert ("answer", "cq10") in events and ("mark", "👍 logged") in events
    finally:
        conn.close()


def test_noop_callback_records_nothing(tmp_path, monkeypatch) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        cfg = _cfg()
        _mute_telegram(monkeypatch)
        monkeypatch.setattr(
            votes.telegram, "get_updates",
            lambda cfg, offset, timeout=0: [{
                "update_id": 5,
                "callback_query": {"id": "cq5", "data": "noop",
                                   "message": {"chat": {"id": 1}, "message_id": 5}},
            }],
        )
        assert drain_once(cfg, conn, timeout=0) == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM votes").fetchone()["n"] == 0
    finally:
        conn.close()
