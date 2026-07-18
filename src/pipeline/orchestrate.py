"""Run-cycle orchestration: fetch -> normalize (M2) -> cluster/score/select (M4+)."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from ..config import Config
from ..db import (
    claim_sent,
    sent_keys,
    sent_today,
    set_sent_message_id,
    today_utc,
    upsert_item,
)
from ..embed import embed
from ..models import Item, Story
from ..sources.feeds import fetch_all_feeds
from ..summarize import summarize_story
from ..taste import load_taste_vectors, refresh_taste
from ..telegram import send_headline
from .cluster import cluster_items
from .editor import choose_headline
from .score import build_shortlist, is_excluded, is_forced, score_matrix

log = logging.getLogger("floss-bot")


@dataclass
class Plan:
    """The read-only outcome of one cycle (no side effects yet)."""

    items: list[Item] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 0), np.float32))
    shortlist: list[Story] = field(default_factory=list)
    sendable: list[Story] = field(default_factory=list)  # floor-clearing + forced only
    best_score: float = 0.0
    quiet: bool = True
    reason: str = ""


def fetch_items(cfg: Config, days: int | None = None) -> list[Item]:
    """Fetch + normalize all sources, de-duped by uid within the batch."""
    raw = fetch_all_feeds(cfg, days=days)
    # Anitya (the cross-forge release tracker) is intentionally disabled: this bot
    # curates news, not version bumps. Re-enable by uncommenting if ever wanted.
    # try:
    #     raw.extend(anitya.fetch_recent(cfg))
    # except Exception as exc:  # noqa: BLE001
    #     log.warning("anitya source failed: %s", exc)
    seen: set[str] = set()
    items: list[Item] = []
    for it in raw:
        if it.uid in seen:
            continue
        seen.add(it.uid)
        items.append(it)
    log.info("fetched %d unique item(s) from %d feed(s)", len(items), len(cfg.feeds))
    return items


def embed_items(cfg: Config, items: list[Item]) -> np.ndarray:
    """Embed ``title. summary`` for clustering (row-aligned to ``items``)."""
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    texts = [f"{it.title}. {it.summary}"[:1500] for it in items]
    return embed(texts, model=cfg.embed_model)


def build_stories(cfg: Config, items: list[Item]) -> list[Story]:
    """Embed + cluster items into corroborated Stories."""
    embeddings = embed_items(cfg, items)
    return cluster_items(items, embeddings, cluster_sim=cfg.cluster_sim)


def _story_vectors(
    items: list[Item], embeddings: np.ndarray, stories: list[Story]
) -> np.ndarray:
    """Per-story embedding matrix (each story's top-item vector), row-aligned."""
    row = {it.uid: i for i, it in enumerate(items)}
    return np.vstack([embeddings[row[s.top_item.uid]] for s in stories])


def _taste_vectors(cfg: Config, conn: sqlite3.Connection):
    """Load the taste profile, auto-seeding from the seed file on first run."""
    pos, labels, neg = load_taste_vectors(conn)
    if pos.shape[0] == 0:
        log.info("taste profile empty — seeding from ground_truth.txt")
        refresh_taste(cfg, conn)
        pos, labels, neg = load_taste_vectors(conn)
    return pos, labels, neg


def score_relevance(
    cfg: Config,
    conn: sqlite3.Connection,
    items: list[Item],
    embeddings: np.ndarray,
    stories: list[Story],
) -> list[Story]:
    """Set each story's relevance ``.score`` + ``.nearest`` against the taste
    profile, after excluding routine version bumps. Returns the kept stories.
    """
    kept = [s for s in stories if not is_excluded(s)]
    if not kept:
        return []
    pos, labels, neg = _taste_vectors(cfg, conn)
    vecs = _story_vectors(items, embeddings, kept)
    for st, r in zip(kept, score_matrix(vecs, pos, labels, neg, cfg.lambda_neg)):
        st.score = r.score
        st.nearest = r.nearest
    return kept


def run_cycle(cfg: Config, conn: sqlite3.Connection) -> Plan:
    """Read-only cycle: fetch -> cluster -> relevance-score -> decide.

    Excludes version bumps, drops already-sent keys, applies the ``max_age_hours``
    staleness guard, builds the relevance shortlist, and sets the quiet/send
    decision. No writes, no network send.
    """
    items = fetch_items(cfg)
    embeddings = embed_items(cfg, items)
    stories = cluster_items(items, embeddings, cluster_sim=cfg.cluster_sim)
    scored = score_relevance(cfg, conn, items, embeddings, stories)

    already = sent_keys(conn)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.max_age_hours)
    fresh = [
        s for s in scored if s.key not in already and s.top_item.published >= cutoff
    ]
    fresh.sort(key=lambda s: s.score, reverse=True)

    # build_shortlist already keeps only floor-clearing + forced items, so the
    # shortlist IS the sendable set (the editor can't pick a below-floor item).
    shortlist = build_shortlist(fresh, cfg.floor, size=5)
    best = max((s.score for s in fresh), default=0.0)
    forced = any(is_forced(s) for s in shortlist)
    plan = Plan(
        items=items, embeddings=embeddings, shortlist=shortlist,
        sendable=shortlist, best_score=best,
    )

    if sent_today(conn, today_utc()) >= cfg.max_per_day:
        plan.quiet, plan.reason = True, "already sent today (one-per-day max)"
    elif not shortlist:
        plan.quiet = True
        plan.reason = f"quiet day: best relevance {best:.3f} < floor {cfg.floor:.2f}"
    else:
        plan.quiet = False
        plan.reason = "forced advisory" if best < cfg.floor and forced else "clears floor"
    return plan


def persist_items(cfg: Config, conn: sqlite3.Connection, plan: Plan) -> None:
    """Record every fetched item + embedding (novelty history + uid-dedup). Commits."""
    emb = plan.embeddings
    for i, it in enumerate(plan.items):
        vec = emb[i] if emb.size else np.zeros(1, dtype=np.float32)
        upsert_item(conn, it, vec)
    conn.commit()


def send_plan(cfg: Config, conn: sqlite3.Connection, plan: Plan) -> Story | None:
    """Editor-pick + send one headline with claim-before-send idempotency (spec §8).

    Assumes ``persist_items`` already ran. Returns the sent Story, or None if quiet
    or already claimed. A send that fails after the claim is a deliberate miss for
    the day, never a duplicate.
    """
    if plan.quiet or not plan.sendable:
        return None
    chosen = choose_headline(cfg, plan.sendable)
    headline = chosen.headline or chosen.top_item.title
    if not claim_sent(conn, chosen.key, today_utc(), headline, chosen.score):
        conn.commit()
        log.info("story %s already sent; skipping", chosen.key)
        return None
    conn.commit()
    summary = summarize_story(cfg, chosen)
    try:
        message_id = send_headline(cfg, chosen, summary)
    except httpx.HTTPError as exc:
        log.warning("send failed for %s: %s (claimed; no retry today)", chosen.key, exc)
        return chosen
    if message_id is not None:
        set_sent_message_id(conn, chosen.key, message_id)
        conn.commit()
    return chosen
