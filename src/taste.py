"""Taste profile: positives (seed anchors + 👍'd stories) and negatives
(negative seed anchors + 👎'd stories).

Stored in the ``taste`` table as L2-normalized float32 blobs. ``kind`` is
``'seed'`` | ``'negseed'`` | ``'pos'`` | ``'neg'``. Positives = seed ∪ pos;
negatives = negseed ∪ neg. Forked from the research bot's taste engine.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np

from .config import ROOT, Config
from .db import blob_to_embedding, embedding_to_blob
from .embed import embed

log = logging.getLogger("floss-bot")

SEED_SEP = " — "  # em dash separates the label from the gist
NEGATIVE_MARKER = "[NEGATIVE]"  # lines after this are negative (disliked) seeds
DEFAULT_SEEDS = ROOT / "seeds" / "ground_truth.txt"


def _seed_id(label: str, kind: str) -> str:
    digest = hashlib.sha1(f"{kind}:{label}".encode()).hexdigest()[:12]
    return f"{kind}:{digest}"


def parse_seed_line(line: str) -> tuple[str, str] | None:
    """Parse one seed line into ``(label, text)``; None for blanks/comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if SEED_SEP in stripped:
        label, gist = stripped.split(SEED_SEP, 1)
        label = label.strip()
        return label, f"{label}. {gist.strip()}"
    return stripped, stripped


def load_seed_lines(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(positives, negatives)`` seed pairs, split on the [NEGATIVE] marker."""
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    negative_mode = False
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip().upper() == NEGATIVE_MARKER:
                negative_mode = True
                continue
            pair = parse_seed_line(raw)
            if pair is None:
                continue
            (negatives if negative_mode else positives).append(pair)
    return positives, negatives


def rebuild_seeds(
    cfg: Config, conn: sqlite3.Connection, seeds_path: Path | str | None = None
) -> tuple[int, int]:
    """Embed the seed file and replace all seed rows. Returns (positives, negatives)."""
    path = Path(seeds_path) if seeds_path else DEFAULT_SEEDS
    if not path.exists():
        log.warning("no seed file at %s; skipping seed rebuild", path)
        return (0, 0)

    positives, negatives = load_seed_lines(path)
    conn.execute("DELETE FROM taste WHERE kind IN ('seed', 'negseed')")

    for kind, pairs in (("seed", positives), ("negseed", negatives)):
        if not pairs:
            continue
        labels = [lbl for lbl, _ in pairs]
        vecs = embed([txt for _, txt in pairs], model=cfg.embed_model)
        conn.executemany(
            f"INSERT OR REPLACE INTO taste (id, kind, label, embedding) "
            f"VALUES (?, '{kind}', ?, ?)",
            [(_seed_id(lbl, kind), lbl, embedding_to_blob(v)) for lbl, v in zip(labels, vecs)],
        )
    conn.commit()
    log.info("loaded %d positive + %d negative seeds from %s",
             len(positives), len(negatives), path)
    return (len(positives), len(negatives))


def fold_votes(conn: sqlite3.Connection) -> tuple[int, int]:
    """Reflect every vote into the taste table using the story's stored embedding.

    👍 → ``kind='pos'``, 👎 → ``kind='neg'``. The vote key is the sent story's key,
    which is its top item's uid, so we join votes.key -> items.uid. Idempotent.
    """
    rows = conn.execute(
        "SELECT v.key AS key, v.vote AS vote, i.title AS title, i.embedding AS embedding "
        "FROM votes v JOIN items i ON i.uid = v.key"
    ).fetchall()

    pos = neg = 0
    for r in rows:
        if r["embedding"] is None:
            log.warning("vote for %s has no stored embedding; skipping", r["key"])
            continue
        kind = "pos" if r["vote"] > 0 else "neg"
        conn.execute(
            "INSERT OR REPLACE INTO taste (id, kind, label, embedding) VALUES (?, ?, ?, ?)",
            (f"vote:{r['key']}", kind, r["title"] or r["key"], r["embedding"]),
        )
        pos += kind == "pos"
        neg += kind == "neg"
    conn.commit()
    return pos, neg


def taste_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT kind, COUNT(*) AS c FROM taste GROUP BY kind").fetchall()
    return {r["kind"]: r["c"] for r in rows}


def refresh_taste(cfg: Config, conn: sqlite3.Connection) -> dict[str, int]:
    """Rebuild seeds from the file and fold in any votes. Returns per-kind counts."""
    rebuild_seeds(cfg, conn)
    fold_votes(conn)
    return taste_counts(conn)


def load_taste_vectors(
    conn: sqlite3.Connection,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load the profile for scoring: ``(positives, pos_labels, negatives)``.

    Positives = seed ∪ pos (labels aligned by row). Negatives = negseed ∪ neg.
    Empty categories come back as ``(0, dim)`` arrays.
    """
    pos_labels: list[str] = []
    pos_vecs: list[np.ndarray] = []
    neg_vecs: list[np.ndarray] = []
    for r in conn.execute("SELECT kind, label, embedding FROM taste"):
        vec = blob_to_embedding(r["embedding"])
        if r["kind"] in ("seed", "pos"):
            pos_labels.append(r["label"])
            pos_vecs.append(vec)
        else:  # negseed | neg
            neg_vecs.append(vec)

    pos = np.vstack(pos_vecs) if pos_vecs else np.empty((0, 0), dtype=np.float32)
    neg = np.vstack(neg_vecs) if neg_vecs else np.empty((0, 0), dtype=np.float32)
    return pos, pos_labels, neg
