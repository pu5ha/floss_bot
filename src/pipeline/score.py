"""Relevance scoring: how close a story is to the taste profile (spec §6, pivoted).

    pos_sim = max_j  story · positive_j        (nearest liked seed/vote)
    neg_sim = max_k  story · negative_k        (nearest disliked seed/vote; 0 if none)
    score   = pos_sim - lambda_neg * neg_sim
    nearest = label of the argmax positive     (shown for transparency)

Everything is L2-normalized, so cosine == dot product. A thin rule layer still
runs on top: force critical advisories on flagship projects, and exclude routine
version bumps (this is a news bot, not a release monitor).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..models import Story

log = logging.getLogger("floss-bot")

# Routine version bumps / churn — never news, always excluded.
_EXCLUDED_TYPES = {"minor_release", "commit"}


@dataclass
class ScoreResult:
    score: float
    nearest: str


def score_matrix(
    emb: np.ndarray,
    pos: np.ndarray,
    pos_labels: list[str],
    neg: np.ndarray,
    lambda_neg: float,
) -> list[ScoreResult]:
    """Score each row of ``emb`` (an ``(n, dim)`` array of story embeddings)."""
    n = emb.shape[0]
    if n == 0:
        return []

    if pos.shape[0]:
        psim = emb @ pos.T  # (n, P)
        pos_sim = psim.max(axis=1)
        pos_arg = psim.argmax(axis=1)
    else:
        pos_sim = np.zeros(n, dtype=np.float32)
        pos_arg = None

    if neg.shape[0]:
        neg_sim = (emb @ neg.T).max(axis=1)
    else:
        neg_sim = np.zeros(n, dtype=np.float32)

    scores = pos_sim - lambda_neg * neg_sim
    return [
        ScoreResult(
            score=float(scores[i]),
            nearest=pos_labels[int(pos_arg[i])] if pos_arg is not None else "",
        )
        for i in range(n)
    ]


def is_excluded(story: Story) -> bool:
    """Routine version bumps / commits are never news — drop them entirely."""
    return story.top_item.event_type in _EXCLUDED_TYPES


def is_forced(story: Story) -> bool:
    """Force-include a critical security advisory on a flagship (tier-1/2) project,
    regardless of relevance — security news you should never miss.
    """
    top = story.top_item
    return top.event_type == "advisory" and story.tier in (1, 2)


def build_shortlist(stories: list[Story], floor: float, size: int = 5) -> list[Story]:
    """Top-``size`` stories by relevance that clear the floor, plus any forced.

    Assumes ``.score`` is set and ``stories`` is sorted by score descending.
    """
    shortlist = [s for s in stories[:size] if s.score >= floor or is_forced(s)]
    chosen = {s.key for s in shortlist}
    for st in stories:
        if is_forced(st) and st.key not in chosen:
            shortlist.append(st)
            chosen.add(st.key)
    shortlist.sort(key=lambda s: s.score, reverse=True)
    return shortlist
