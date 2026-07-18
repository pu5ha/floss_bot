"""Shared, polite HTTP GET with retries and backoff (spec §13)."""

from __future__ import annotations

import logging
import time

import httpx

from ..config import Config

log = logging.getLogger("floss-bot")


def get_with_retries(
    url: str,
    cfg: Config,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: float = 30.0,
) -> httpx.Response:
    """GET with a descriptive User-Agent, redirect-following, and backoff.

    ``follow_redirects=True`` is mandatory — many feed hosts 301 to https and
    silently return nothing otherwise. Raises the last ``httpx.HTTPError`` if all
    attempts fail (the caller is responsible for per-source isolation).
    """
    hdrs = {"User-Agent": cfg.user_agent}
    if headers:
        hdrs.update(headers)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(
                url,
                params=params,
                headers=hdrs,
                timeout=timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc
