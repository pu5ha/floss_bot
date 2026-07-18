"""M1: embedding sanity — normalization makes cosine == dot product."""

from __future__ import annotations

import numpy as np

from src.embed import cosine, embed


def test_cosine_basic() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine(a, b) == 1.0
    assert abs(cosine(a, c)) < 1e-6


def test_cosine_zero_vector() -> None:
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine(a, b) == 0.0


def test_embed_empty() -> None:
    out = embed([])
    assert out.shape == (0, 0)


def test_embed_normalized_and_dot_is_cosine() -> None:
    # Real model call (CPU, tiny). Skips gracefully if the model can't load.
    try:
        vecs = embed(["signal releases a new version", "signal ships an update"])
    except Exception as exc:  # noqa: BLE001 — offline/no-model environments
        import pytest

        pytest.skip(f"embedding model unavailable: {exc}")
    assert vecs.shape[0] == 2
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # Two near-paraphrases should be strongly similar.
    assert float(vecs[0] @ vecs[1]) > 0.5
