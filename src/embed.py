"""Local embedding primitive (clustering only). Loads one model per process and
returns L2-normalized float32 vectors so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import os

import numpy as np

# Keep model-load output quiet (progress bars, advisory warnings) before the
# heavy libs are imported. Set as defaults so callers can still override.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The model is heavy to load, so we cache it per (model name) for the process.
_MODEL_CACHE: dict[str, object] = {}


def _get_model(name: str):
    model = _MODEL_CACHE.get(name)
    if model is None:
        # Imported lazily so that non-embedding CLI paths don't pay the import cost.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name)
        _MODEL_CACHE[name] = model
    return model


def embed(texts: list[str], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed ``texts`` into an ``(n, dim)`` float32 array of unit-norm vectors.

    Normalization is guaranteed here (via ``normalize_embeddings=True``) because
    clustering treats cosine similarity as a dot product.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vecs = _get_model(model).encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (safe even if not pre-normalized)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
