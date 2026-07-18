"""Editor pass — a local LLM (Ollama) picks #1 of the top-5 and writes the
headline (spec §7).

The editor only ever sees the shortlist of 5, so it cannot silently drop a
critical item — the rules already guaranteed eligibility. JSON is parsed
defensively (fence-stripping, one retry); on any failure it falls back to the
#1-by-score story and uses its title. A model hiccup must never skip the day.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from ..config import Config
from ..models import Story

log = logging.getLogger("floss-bot")

SYSTEM_PROMPT = (
    "You edit a daily briefing for someone who cares deeply about free/open-source "
    "software, privacy, decentralization, cryptography, open hardware, and digital "
    'rights (the "d/acc" worldview). From the numbered candidate stories, choose the '
    "SINGLE most significant for this reader today, and write one punchy headline "
    "(<=120 chars) FOR THAT STORY. Prefer: security-critical news, first-of-their-kind "
    "releases, things that shift power toward users. Respond ONLY as JSON: "
    '{"choice": <the story number>, "headline": "<your headline for that story>"}.'
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_MAX_HEADLINE = 120


def _distinct_sources(story: Story) -> str:
    seen: list[str] = []
    for it in story.items:
        if it.source not in seen:
            seen.append(it.source)
    return ", ".join(seen)


def build_user_prompt(shortlist: list[Story]) -> str:
    """Render the numbered candidates: title, event_type, tier, corroboration, sources, summary."""
    blocks = []
    for i, st in enumerate(shortlist, 1):
        top = st.top_item
        blocks.append(
            f"[{i}] title: {top.title}\n"
            f"    type: {top.event_type} · tier {st.tier} · "
            f"corroboration {st.corroboration} · sources: {_distinct_sources(st)}\n"
            f"    summary: {top.summary[:300]}"
        )
    return "Candidate stories:\n\n" + "\n\n".join(blocks)


def _parse_response(content: str, n: int) -> tuple[int, str] | None:
    """Extract (choice_index_0based, headline) from a response, tolerating fences.

    ``n`` is the shortlist length; the model's 1-based choice must be in [1, n].
    """
    cleaned = _FENCE_RE.sub("", content).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    choice = data.get("choice")
    headline = data.get("headline")
    if headline is None or choice is None:
        return None
    try:
        idx = int(choice) - 1
    except (TypeError, ValueError):
        return None
    if not (0 <= idx < n):
        return None
    headline = str(headline).strip()
    if not headline:
        return None
    return idx, headline


def _call_ollama(cfg: Config, user_prompt: str) -> str:
    """POST the shortlist to Ollama /api/chat and return the raw message content."""
    resp = httpx.post(
        f"{cfg.editor.url}/api/chat",
        json={
            "model": cfg.editor.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3},
        },
        timeout=cfg.editor.timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "") or ""


def choose_headline(cfg: Config, shortlist: list[Story]) -> Story:
    """Pick one story from the shortlist and set its ``.headline`` (spec §7).

    Falls back to the #1-by-score story (title as headline) on any failure so the
    day is never skipped silently. Returns the chosen Story.
    """
    if not shortlist:
        raise ValueError("choose_headline called with an empty shortlist")

    ranked = sorted(shortlist, key=lambda s: s.score, reverse=True)
    fallback = ranked[0]

    if not cfg.editor.enabled:
        fallback.headline = fallback.top_item.title
        return fallback

    user_prompt = build_user_prompt(ranked)

    for attempt in (1, 2):  # one retry
        try:
            raw = _call_ollama(cfg, user_prompt)
            log.info("editor raw output (attempt %d): %s", attempt, raw[:500])
            parsed = _parse_response(raw, len(ranked))
        except httpx.HTTPError as exc:
            log.warning("editor call failed (attempt %d): %s", attempt, exc)
            parsed = None

        if parsed is not None:
            idx, headline = parsed
            chosen = ranked[idx]  # index guarantees headline matches the chosen story
            chosen.headline = headline[:_MAX_HEADLINE]
            return chosen

    log.warning("editor produced no usable choice; falling back to #1 by score (its title)")
    fallback.headline = fallback.top_item.title
    return fallback
