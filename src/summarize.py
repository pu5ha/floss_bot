"""Local-LLM (Ollama) 3-sentence summary for the chosen story.

Only ever runs for the single story actually sent, so cost is trivial. Falls back
to a truncated feed summary if Ollama is unreachable or returns nothing.
"""

from __future__ import annotations

import logging

import httpx

from .config import Config
from .models import Story

log = logging.getLogger("floss-bot")

_PROMPT = (
    "Summarize the following {kind} in exactly 3 concise, factual sentences for a "
    "reader who follows free/open-source software, privacy, cryptography, and "
    "decentralization. Do NOT invent details that are not in the text; if the text "
    "is short, keep the summary short. Output ONLY the summary — no preamble, no "
    "lists.\n\nTitle: {title}\n\n{body}"
)


# Below this length there's nothing to summarize (e.g. a bare Anitya version
# line); use the feed text directly rather than making the LLM pad or refuse.
_MIN_BODY_FOR_LLM = 160


def _richest_body(story: Story) -> str:
    """The longest summary across all clustered items (a GitHub changelog beats a
    bare Anitya line even when Anitya is the canonical top item).
    """
    return max((it.summary for it in story.items), key=len, default="")


def _fallback(story: Story) -> str:
    text = _richest_body(story) or story.top_item.title
    return text[:300] + ("…" if len(text) > 300 else "")


def summarize_story(cfg: Config, story: Story) -> str:
    """Return a 3-sentence summary, or the feed text directly when there's too
    little to summarize / on any failure.
    """
    top = story.top_item
    body = _richest_body(story)
    if not cfg.editor.enabled or len(body) < _MIN_BODY_FOR_LLM:
        return _fallback(story)

    prompt = _PROMPT.format(
        kind=top.event_type.replace("_", " "),
        title=top.title,
        body=body[:2000],
    )
    try:
        resp = httpx.post(
            f"{cfg.editor.url}/api/generate",
            json={
                "model": cfg.editor.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=cfg.editor.timeout,
        )
        resp.raise_for_status()
        summary = " ".join((resp.json().get("response") or "").split())
        return summary or _fallback(story)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("summarize failed for %s: %s (using feed summary)", story.key, exc)
        return _fallback(story)
