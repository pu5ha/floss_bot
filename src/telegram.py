"""Telegram send + vote-callback primitives (spec §9). Only network credential
is the bot token; all calls use httpx with follow_redirects=True.
"""

from __future__ import annotations

import html
import json
import logging

import httpx

from .config import Config
from .models import Story

log = logging.getLogger("floss-bot")

API = "https://api.telegram.org"


def is_configured(cfg: Config) -> bool:
    return bool(cfg.secrets.telegram_bot_token and cfg.secrets.telegram_chat_id)


def _api(cfg: Config, method: str) -> str:
    return f"{API}/bot{cfg.secrets.telegram_bot_token}/{method}"


def _esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def build_keyboard(key: str) -> dict:
    """Inline 👍 / 👎 keyboard. callback_data is ``up:{key}`` / ``down:{key}``."""
    return {
        "inline_keyboard": [
            [
                {"text": "👍", "callback_data": f"up:{key}"},
                {"text": "👎", "callback_data": f"down:{key}"},
            ]
        ]
    }


def build_message(story: Story, summary: str) -> str:
    """HTML body: headline / source(+N more) / 3-sentence summary / link."""
    top = story.top_item
    headline = story.headline or top.title
    more = f" +{story.corroboration - 1} more" if story.corroboration > 1 else ""
    return (
        f"📰 <b>{_esc(headline)}</b>\n"
        f"<i>{_esc(top.source)}{more}</i>\n"
        f"{_esc(summary)}\n"
        f"{_esc(top.url)}"
    )


def send_headline(cfg: Config, story: Story, summary: str) -> int | None:
    """Send one headline + summary with the 👍/👎 keyboard. Returns the message id."""
    if not is_configured(cfg):
        raise RuntimeError("Telegram is not configured (token/chat_id missing)")
    payload = {
        "chat_id": cfg.secrets.telegram_chat_id,
        "text": build_message(story, summary),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": json.dumps(build_keyboard(story.key)),
    }
    resp = httpx.post(
        _api(cfg, "sendMessage"), data=payload, timeout=30, follow_redirects=True
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        log.warning("sendMessage not ok: %s", data)
        return None
    return int(data["result"]["message_id"])


def get_updates(cfg: Config, offset: int, timeout: int = 25) -> list[dict]:
    """Long-poll getUpdates. Returns the result list (empty on error)."""
    resp = httpx.get(
        _api(cfg, "getUpdates"),
        params={"offset": offset, "timeout": timeout},
        timeout=timeout + 10,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", []) if data.get("ok") else []


def answer_callback(cfg: Config, callback_query_id: str, text: str = "") -> None:
    """Stop the button spinner (best-effort)."""
    httpx.post(
        _api(cfg, "answerCallbackQuery"),
        data={"callback_query_id": callback_query_id, "text": text},
        timeout=15,
        follow_redirects=True,
    )


def mark_message_voted(cfg: Config, chat_id: int, message_id: int, label: str) -> None:
    """Replace the keyboard with a single disabled button confirming the vote."""
    httpx.post(
        _api(cfg, "editMessageReplyMarkup"),
        data={
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": json.dumps(
                {"inline_keyboard": [[{"text": label, "callback_data": "noop"}]]}
            ),
        },
        timeout=15,
        follow_redirects=True,
    )
