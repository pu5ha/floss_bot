"""Anitya (release-monitoring.org) poller — the cross-forge release backbone (spec §4).

Anitya tracks upstream release versions across every forge/tarball host, so it
catches releases GitHub-only feeds miss. There is no by-id GET endpoint, so we
query ``?name=`` and select the item matching the configured id (names collide,
e.g. "signal" resolves to a crates.io package). Poll politely.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from ..config import AnityaProject, Config
from ..models import Item
from ..pipeline.normalize import classify_release, parse_version
from .base import get_with_retries

log = logging.getLogger("floss-bot")

NAME = "Anitya"
API = "https://release-monitoring.org/api/v2/projects/"
_SLEEP = 1.0  # polite gap between per-project requests


def _select(items: list[dict], want_id: int) -> dict | None:
    for it in items:
        if int(it.get("id", -1)) == want_id:
            return it
    return None


def parse_project(data: dict, entry: AnityaProject) -> Item | None:
    """Build an Item from an Anitya project record (pure; no network).

    ``data`` is the ``?name=`` response ({"items": [...]}); we pick the item whose
    id matches ``entry.id``. Returns ``None`` if absent or versionless.
    """
    record = _select(data.get("items", []), entry.id)
    if record is None:
        return None
    version = record.get("version")
    if not version:
        return None

    updated = record.get("updated_on")
    published = (
        datetime.fromtimestamp(float(updated), tz=timezone.utc)
        if updated
        else datetime.now(timezone.utc)
    )
    url = record.get("version_url") or record.get("homepage") or ""
    backend = record.get("backend") or "upstream"

    parsed = parse_version(str(version))
    if parsed:
        event_type, magnitude = classify_release(parsed)
    else:
        event_type, magnitude = "minor_release", 0.5

    return Item(
        # version in the uid -> a new release is a new item; a re-check is a no-op.
        uid=f"anitya:{entry.id}:{version}",
        source=NAME,
        layer=1,
        project=entry.project,
        title=f"{entry.project} {version} released",
        url=url,
        summary=f"New {entry.project} release {version} (via release-monitoring.org, {backend}).",
        published=published,
        event_type=event_type,
        magnitude=magnitude,
        tier=1,  # only tier-1/2 upstreams are configured for Anitya
    )


def fetch_project(cfg: Config, entry: AnityaProject) -> Item | None:
    """Fetch and normalize one Anitya project by name, matched by id."""
    resp = get_with_retries(
        API,
        cfg,
        params={"name": entry.name},
        headers={"Accept": "application/json"},
    )
    return parse_project(resp.json(), entry)


def fetch_recent(cfg: Config) -> list[Item]:
    """Poll every configured Anitya upstream, isolating per-project failures."""
    out: list[Item] = []
    for i, entry in enumerate(cfg.anitya):
        if i:
            time.sleep(_SLEEP)
        try:
            item = fetch_project(cfg, entry)
            if item is not None:
                out.append(item)
        except Exception as exc:  # noqa: BLE001 — one project must not abort the run
            log.warning("anitya %s (id=%s) failed: %s", entry.name, entry.id, exc)
    log.info("anitya: fetched %d release(s) from %d project(s)", len(out), len(cfg.anitya))
    return out
