"""``calibrate`` — see the bot's judgment before it ever pings you (spec §11).

Replays the last N days and, for each day, shows the ONE headline it would have
sent (real editor pick + summary) or QUIET with the nearest near-miss. Ranking is
by RELEVANCE to the taste profile. Sends and stores NOTHING.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict

from .config import Config
from .models import Story
from .pipeline.editor import choose_headline
from .pipeline.orchestrate import (
    build_stories,
    embed_items,
    fetch_items,
    score_relevance,
)
from .pipeline.score import build_shortlist, is_forced
from .summarize import summarize_story

log = logging.getLogger("floss-bot")

_FLOOR_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _print_floor_sweep(day_scores: dict[str, list[Story]]) -> None:
    n = len(day_scores)
    print("\n--- floor sweep (how many of the replayed days would send) ---")
    for floor in _FLOOR_SWEEP:
        active = sum(
            1 for sts in day_scores.values()
            if any(s.score >= floor or is_forced(s) for s in sts)
        )
        print(f"  floor {floor:.2f}:  send {active}/{n} days  ·  quiet {n - active}")


def run_calibrate(cfg: Config, conn: sqlite3.Connection, days: int = 14) -> None:
    """Replay ``days`` of history, printing the would-send headline per day."""
    items = fetch_items(cfg, days=days)
    if not items:
        print("No items fetched — nothing to calibrate.")
        return

    embeddings = embed_items(cfg, items)
    stories = build_stories(cfg, items)
    scored = score_relevance(cfg, conn, items, embeddings, stories)

    by_day: dict[str, list[Story]] = defaultdict(list)
    for st in scored:
        by_day[st.top_item.published.date().isoformat()].append(st)

    print(
        f"\nReplaying {len(scored)} stories across {len(by_day)} day(s) "
        f"(window {days}d, relevance floor {cfg.floor:.2f}, lambda_neg {cfg.lambda_neg:.2f})."
    )

    sent_days = 0
    for day in sorted(by_day, reverse=True):
        day_stories = sorted(by_day[day], key=lambda s: s.score, reverse=True)
        shortlist = build_shortlist(day_stories, cfg.floor, size=5)
        if not shortlist:
            best = day_stories[0]
            print(f"\n=== {day} · QUIET ===")
            print(f"  best near-miss: {best.score:.3f}  {best.top_item.title[:72]}")
            print(f"     (closest to: {best.nearest})")
            continue

        sent_days += 1
        pick = choose_headline(cfg, shortlist)
        top = pick.top_item
        forced = " ⚑FORCED" if is_forced(pick) else ""
        summary = summarize_story(cfg, pick)
        print(f"\n=== {day} · WOULD SEND (relevance {pick.score:.3f}{forced}) ===")
        print(f"  📰 {pick.headline}")
        print(f"     {top.source} · closest to: {pick.nearest}")
        print(f"     {summary}")
        print(f"     {top.url}")
        for s in [s for s in shortlist if s.key != pick.key][:3]:
            print(f"       also eligible: {s.score:.3f}  {s.top_item.title[:58]}")

    _print_floor_sweep(by_day)
    print(
        f"\nAt floor {cfg.floor:.2f}: would send on {sent_days}/{len(by_day)} days, "
        f"quiet on {len(by_day) - sent_days}.  (Nothing was sent or stored.)"
    )
