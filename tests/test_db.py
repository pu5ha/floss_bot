"""M1: DB schema + core helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.db import (
    connect,
    get_offset,
    init_schema,
    record_vote,
    set_offset,
    upsert_item,
)
from src.models import Item


def _item(uid: str = "feed:abc") -> Item:
    return Item(
        uid=uid,
        source="Example",
        layer=1,
        project="signal",
        title="Signal 7.0 released",
        url="https://example.org/signal-7",
        summary="A major release.",
        published=datetime(2026, 7, 18, tzinfo=timezone.utc),
        event_type="major_release",
        magnitude=0.9,
        tier=1,
    )


def test_schema_and_upsert_idempotent(tmp_path) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"items", "stories", "sent", "votes", "weights", "source_state"} <= tables

        vec = np.ones(4, dtype=np.float32)
        upsert_item(conn, _item(), vec)
        upsert_item(conn, _item(), vec)  # INSERT OR IGNORE -> still one row
        conn.commit()
        count = conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
        assert count == 1
    finally:
        conn.close()


def test_vote_and_offset(tmp_path) -> None:
    conn = connect(tmp_path / "floss.db")
    try:
        init_schema(conn)
        assert get_offset(conn) == 0
        set_offset(conn, 42)
        assert get_offset(conn) == 42

        record_vote(conn, "story:1", 1)
        record_vote(conn, "story:1", -1)  # re-vote overwrites
        conn.commit()
        row = conn.execute("SELECT vote FROM votes WHERE key='story:1'").fetchone()
        assert row["vote"] == -1
    finally:
        conn.close()
