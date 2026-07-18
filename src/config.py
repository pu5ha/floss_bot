"""Typed configuration: ``config.yaml`` + ``config/feeds.yaml`` + ``config/tiers.yaml``
(non-secret) plus ``.env`` (secrets).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root = parent of the ``src`` package directory.
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FeedConfig:
    """One RSS/Atom feed from ``config/feeds.yaml``."""

    name: str
    url: str
    layer: int
    project: str | None = None
    tier: int | None = None


@dataclass
class AnityaProject:
    """One upstream tracked on release-monitoring.org (queried by name, matched by id)."""

    name: str  # Anitya project name (the ?name= query)
    id: int  # Anitya project id (disambiguates name collisions)
    project: str  # our internal project slug (ties into tiers)


@dataclass
class EditorConfig:
    """The local LLM (Ollama) that picks + phrases the daily headline."""

    enabled: bool = True
    model: str = "llama3.1:8b"
    url: str = "http://localhost:11434"
    timeout: float = 60.0


@dataclass
class Secrets:
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    contact_email: str | None = None


# Default significance weights (spec §6). Overridable via config.yaml and
# tuned over time in the ``weights`` DB table by the 👍/👎 feedback loop.
# Tuned via `calibrate` (M6): tier weight raised and corroboration lowered so a
# tier-1 release clears the floor while tier-3 noise stays quiet.
DEFAULT_WEIGHTS: dict[str, float] = {
    "w_type": 0.30,
    "w_mag": 0.15,
    "w_tier": 0.30,
    "w_corrob": 0.15,
    "w_novel": 0.10,
}

# Rank of each event type (spec §6). Also tunable via config.yaml.
# "Big news only": minor/patch releases rank LOW so they stay below the floor;
# only major releases, launches, and tier-1/2 advisories (forced) surface.
DEFAULT_TYPE_RANK: dict[str, float] = {
    "advisory": 1.0,
    "major_release": 0.8,
    "launch": 0.75,
    "paper": 0.6,
    "grant": 0.55,
    "minor_release": 0.35,
    "post": 0.35,
    "commit": 0.1,
}

# Weight applied per project tier (spec §6).
DEFAULT_TIER_WEIGHT: dict[int, float] = {1: 1.0, 2: 0.6, 3: 0.3}


@dataclass
class Config:
    embed_model: str
    max_per_day: int
    max_age_hours: int
    floor: float  # relevance floor: the quiet-day bar (cosine to the taste profile)
    lambda_neg: float  # penalty weight on nearest disliked seed/vote
    cluster_sim: float
    weights: dict[str, float]
    type_rank: dict[str, float]
    tier_weight: dict[int, float]
    feed_window_days: int
    feeds: list[FeedConfig]
    tiers: dict[int, list[str]]  # tier number -> list of project slugs
    anitya: list[AnityaProject]  # upstreams to poll on release-monitoring.org
    editor: EditorConfig
    secrets: Secrets
    db_path: Path

    @property
    def user_agent(self) -> str:
        contact = self.secrets.contact_email or "unknown"
        return f"floss-bot/0.1 (contact: {contact})"

    def project_tier(self, project: str | None) -> int:
        """Return the tier (1/2/3) for a project slug; 3 if unknown."""
        if not project:
            return 3
        for tier, names in self.tiers.items():
            if project in names:
                return tier
        return 3


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_path: Path | str | None = None,
    env_path: Path | str | None = None,
    feeds_path: Path | str | None = None,
    tiers_path: Path | str | None = None,
) -> Config:
    """Load and type the config. Fails loudly if ``config.yaml`` is missing.

    ``.env`` and the feeds/tiers files are optional (empty -> no feeds/tiers).
    """
    cfg_path = Path(config_path) if config_path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config file not found at {cfg_path}. "
            "Copy config.example.yaml to config.yaml and edit it."
        )
    raw = _load_yaml(cfg_path)

    # .env is optional; load_dotenv silently no-ops if the file is absent.
    load_dotenv(Path(env_path) if env_path else ROOT / ".env")

    feeds_file = Path(feeds_path) if feeds_path else ROOT / "config" / "feeds.yaml"
    feeds_raw = _load_yaml(feeds_file)
    feeds = [
        FeedConfig(
            name=str(f["name"]),
            url=str(f["url"]),
            layer=int(f.get("layer", 2)),
            project=(str(f["project"]) if f.get("project") else None),
            tier=(int(f["tier"]) if f.get("tier") is not None else None),
        )
        for f in (feeds_raw.get("feeds") or [])
    ]

    tiers_file = Path(tiers_path) if tiers_path else ROOT / "config" / "tiers.yaml"
    tiers_raw = _load_yaml(tiers_file)
    tiers = {
        int(tier): [str(p) for p in (projs or [])]
        for tier, projs in (tiers_raw.get("tiers") or {}).items()
    }
    anitya = [
        AnityaProject(
            name=str(a["name"]),
            id=int(a["id"]),
            project=str(a.get("project", a["name"])),
        )
        for a in (tiers_raw.get("anitya") or [])
    ]

    ed = raw.get("editor") or {}
    editor = EditorConfig(
        enabled=bool(ed.get("enabled", True)),
        model=str(ed.get("model", "llama3.1:8b")),
        url=str(ed.get("url", "http://localhost:11434")),
        timeout=float(ed.get("timeout", 60.0)),
    )

    secrets = Secrets(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
        contact_email=os.getenv("CONTACT_EMAIL") or None,
    )

    weights = dict(DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in (raw.get("weights") or {}).items()})

    type_rank = dict(DEFAULT_TYPE_RANK)
    type_rank.update({k: float(v) for k, v in (raw.get("type_rank") or {}).items()})

    tier_weight = dict(DEFAULT_TIER_WEIGHT)
    tier_weight.update(
        {int(k): float(v) for k, v in (raw.get("tier_weight") or {}).items()}
    )

    return Config(
        embed_model=str(raw.get("embed_model", "all-MiniLM-L6-v2")),
        max_per_day=int(raw.get("max_per_day", 1)),
        max_age_hours=int(raw.get("max_age_hours", 36)),
        floor=float(raw.get("floor", 0.35)),
        lambda_neg=float(raw.get("lambda_neg", 0.5)),
        cluster_sim=float(raw.get("cluster_sim", 0.80)),
        weights=weights,
        type_rank=type_rank,
        tier_weight=tier_weight,
        feed_window_days=int(raw.get("feed_window_days", 2)),
        feeds=feeds,
        tiers=tiers,
        anitya=anitya,
        editor=editor,
        secrets=secrets,
        db_path=ROOT / "data" / "floss.db",
    )
