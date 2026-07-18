"""`poll-votes` — the 👍/👎 feedback loop. Long-polls getUpdates, records votes,
confirms them in Telegram, and refreshes the taste profile so votes take effect:
👍 = "more headlines like this", 👎 = "less like this".
"""

from __future__ import annotations

import logging
import sqlite3

import httpx

from . import telegram
from .config import Config
from .db import get_offset, record_vote, set_offset
from .taste import refresh_taste

log = logging.getLogger("floss-bot")


def _handle_callback(cfg: Config, conn: sqlite3.Connection, cq: dict) -> bool:
    """Process one callback_query. Returns True if a vote was recorded."""
    # Always stop the spinner, even for stale/noop buttons.
    telegram.answer_callback(cfg, cq.get("id", ""))

    data = cq.get("data", "")
    if ":" not in data:
        return False
    action, key = data.split(":", 1)  # story keys can contain ':' -> split once
    if action not in ("up", "down"):
        return False

    vote = 1 if action == "up" else -1
    record_vote(conn, key, vote)
    conn.commit()

    label = "👍 logged" if vote > 0 else "👎 logged"
    msg = cq.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    message_id = msg.get("message_id")
    if chat_id and message_id:
        try:
            telegram.mark_message_voted(cfg, chat_id, message_id, label)
        except httpx.HTTPError as exc:
            log.warning("could not edit message for %s: %s", key, exc)
    log.info("recorded %s for %s", label, key)
    return True


def drain_once(cfg: Config, conn: sqlite3.Connection, timeout: int = 0) -> int:
    """Fetch pending updates, process callbacks, advance the offset. Returns votes recorded."""
    offset = get_offset(conn)
    updates = telegram.get_updates(cfg, offset, timeout=timeout)
    votes = 0
    for u in updates:
        offset = u["update_id"] + 1
        cq = u.get("callback_query")
        if cq and _handle_callback(cfg, conn, cq):
            votes += 1
    if updates:
        set_offset(conn, offset)
        conn.commit()
    return votes


def poll_votes(cfg: Config, conn: sqlite3.Connection, once: bool = False) -> None:
    """Drain votes; refresh taste when any land. ``once`` drains pending and exits;
    otherwise long-polls continuously (Ctrl-C to stop).
    """
    if not telegram.is_configured(cfg):
        log.warning("Telegram not configured — set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return
    while True:
        try:
            votes = drain_once(cfg, conn, timeout=0 if once else 25)
        except httpx.HTTPError as exc:
            log.warning("getUpdates failed: %s", exc)
            votes = 0
        if votes:
            counts = refresh_taste(cfg, conn)
            log.info(
                "folded %d vote(s); taste now %d seeds / %d 👍 / %d 👎",
                votes, counts.get("seed", 0), counts.get("pos", 0), counts.get("neg", 0),
            )
        if once:
            return
