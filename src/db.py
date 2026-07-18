"""SQLite schema init, connection, and shared storage helpers (spec §3)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import Item

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  uid TEXT PRIMARY KEY, source TEXT, layer INT, project TEXT, title TEXT, url TEXT,
  summary TEXT, published TEXT, event_type TEXT, magnitude REAL, tier INT,
  embedding BLOB, first_seen TEXT
);
CREATE TABLE IF NOT EXISTS stories (
  key TEXT PRIMARY KEY, day TEXT, score REAL, corroboration INT, top_uid TEXT, item_uids TEXT
);
CREATE TABLE IF NOT EXISTS sent (
  key TEXT PRIMARY KEY, day TEXT, headline TEXT, telegram_message_id INTEGER,
  sent_at TEXT, score REAL
);
CREATE TABLE IF NOT EXISTS votes (
  key TEXT PRIMARY KEY, vote INTEGER, voted_at TEXT
);
CREATE TABLE IF NOT EXISTS weights (
  name TEXT PRIMARY KEY, value REAL
);
CREATE TABLE IF NOT EXISTS taste (
  id TEXT PRIMARY KEY, kind TEXT, label TEXT, embedding BLOB
);
CREATE TABLE IF NOT EXISTS source_state (
  source TEXT PRIMARY KEY, last_run TEXT, last_cursor TEXT
);
"""


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today_utc() -> str:
    """Current UTC date as YYYY-MM-DD (the 'day' key for one-headline-per-day)."""
    return datetime.now(timezone.utc).date().isoformat()


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating parent dir if needed) and return a configured connection."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # run-once (cron) and poll-votes (service) write concurrently; wait for the
    # lock instead of failing with "database is locked".
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def embedding_to_blob(vec: np.ndarray) -> bytes:
    """Serialize an embedding to a float32 blob for BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a float32 blob back into a numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


def load_weights(
    conn: sqlite3.Connection, defaults: dict[str, float]
) -> dict[str, float]:
    """Return the tunable signal weights, seeding the table from ``defaults`` once.

    Stored values override defaults; any default missing from the table is written
    back so the vote-tuning loop (M9) has a row to nudge.
    """
    stored = {
        r["name"]: float(r["value"]) for r in conn.execute("SELECT name, value FROM weights")
    }
    merged = dict(defaults)
    merged.update(stored)
    missing = {k: v for k, v in merged.items() if k not in stored}
    if missing:
        save_weights(conn, missing)
        conn.commit()
    return merged


def save_weights(conn: sqlite3.Connection, weights: dict[str, float]) -> None:
    """Upsert signal weights. Caller commits."""
    conn.executemany(
        "INSERT INTO weights (name, value) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
        [(k, float(v)) for k, v in weights.items()],
    )


def known_projects(conn: sqlite3.Connection) -> set[str]:
    """Distinct project slugs already stored (for the novelty signal)."""
    return {
        r["project"]
        for r in conn.execute("SELECT DISTINCT project FROM items WHERE project IS NOT NULL")
    }


def sent_keys(conn: sqlite3.Connection) -> set[str]:
    """Story keys already sent (never re-send)."""
    return {r["key"] for r in conn.execute("SELECT key FROM sent")}


def sent_today(conn: sqlite3.Connection, day: str) -> int:
    """How many headlines have already gone out today (one-per-day guard)."""
    row = conn.execute("SELECT COUNT(*) AS n FROM sent WHERE day = ?", (day,)).fetchone()
    return int(row["n"])


def claim_sent(
    conn: sqlite3.Connection, key: str, day: str, headline: str, score: float
) -> bool:
    """Reserve a story key in ``sent`` BEFORE the Telegram call (idempotency).

    Returns True if this call claimed it (safe to send), False if already claimed
    (a crash/retry never double-pings). Caller commits.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO sent (key, day, headline, telegram_message_id, sent_at, score) "
        "VALUES (?, ?, ?, NULL, ?, ?)",
        (key, day, headline, now_utc_iso(), score),
    )
    return cur.rowcount > 0


def set_sent_message_id(conn: sqlite3.Connection, key: str, message_id: int) -> None:
    """Record the Telegram message id after a successful send. Caller commits."""
    conn.execute(
        "UPDATE sent SET telegram_message_id = ? WHERE key = ?", (message_id, key)
    )


def record_vote(conn: sqlite3.Connection, key: str, vote: int) -> None:
    """Upsert a 👍 (+1) / 👎 (-1) vote. Re-voting overwrites the prior row."""
    conn.execute(
        "INSERT OR REPLACE INTO votes (key, vote, voted_at) VALUES (?, ?, ?)",
        (key, vote, now_utc_iso()),
    )


def get_offset(conn: sqlite3.Connection) -> int:
    """Last-processed Telegram getUpdates offset (0 if none)."""
    row = conn.execute(
        "SELECT last_cursor FROM source_state WHERE source = 'telegram'"
    ).fetchone()
    return int(row["last_cursor"]) if row and row["last_cursor"] else 0


def set_offset(conn: sqlite3.Connection, offset: int) -> None:
    conn.execute(
        "INSERT INTO source_state (source, last_run, last_cursor) "
        "VALUES ('telegram', ?, ?) "
        "ON CONFLICT(source) DO UPDATE SET last_run=excluded.last_run, "
        "last_cursor=excluded.last_cursor",
        (now_utc_iso(), str(offset)),
    )


def upsert_item(conn: sqlite3.Connection, item: Item, embedding: np.ndarray) -> None:
    """Insert an item and its clustering embedding (no-op if uid exists). Caller commits."""
    conn.execute(
        "INSERT OR IGNORE INTO items "
        "(uid, source, layer, project, title, url, summary, published, "
        " event_type, magnitude, tier, embedding, first_seen) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            item.uid,
            item.source,
            item.layer,
            item.project,
            item.title,
            item.url,
            item.summary,
            item.published.isoformat() if item.published else None,
            item.event_type,
            item.magnitude,
            item.tier,
            embedding_to_blob(embedding),
            now_utc_iso(),
        ),
    )
