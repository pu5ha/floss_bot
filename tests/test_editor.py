"""M7: editor JSON parsing + robust fallback (offline; Ollama never hit)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from src.config import Config, EditorConfig, Secrets
from src.models import Item, Story
from src.pipeline import editor


def _cfg(enabled: bool = True) -> Config:
    return Config(
        embed_model="all-MiniLM-L6-v2", max_per_day=1, max_age_hours=36, floor=0.35,
        lambda_neg=0.5, cluster_sim=0.80, weights={}, type_rank={}, tier_weight={},
        feed_window_days=2,
        feeds=[], tiers={}, anitya=[], editor=EditorConfig(enabled=enabled),
        secrets=Secrets(), db_path="/tmp/x.db",
    )


def _story(key, score, title="A title") -> Story:
    top = Item(
        uid=key, source="Src", layer=1, project="p", title=title, url="u", summary="s",
        published=datetime(2026, 7, 18, tzinfo=timezone.utc),
        event_type="major_release", magnitude=0.9, tier=1,
    )
    st = Story(key=key, items=[top], corroboration=1, top_item=top)
    st.score = score
    return st


def _shortlist():
    return [_story("k1", 0.7, "Story one"), _story("k2", 0.6, "Story two")]


def test_parse_response_plain_json() -> None:
    assert editor._parse_response('{"choice":1,"headline":"Hi"}', 2) == (0, "Hi")


def test_parse_response_strips_fences() -> None:
    fenced = '```json\n{"choice":2,"headline":"Yo"}\n```'
    assert editor._parse_response(fenced, 2) == (1, "Yo")


def test_parse_response_bad() -> None:
    assert editor._parse_response("not json at all", 2) is None
    assert editor._parse_response('{"choice":1}', 2) is None  # missing headline
    assert editor._parse_response('{"choice":9,"headline":"H"}', 2) is None  # out of range
    assert editor._parse_response('{"headline":"H"}', 2) is None  # missing choice


def test_choose_uses_model_pick(monkeypatch) -> None:
    # ranked #2 (k2, score 0.6) -> choice 2. Headline must attach to k2.
    monkeypatch.setattr(
        editor, "_call_ollama", lambda cfg, prompt: '{"choice":2,"headline":"Model wrote this"}'
    )
    chosen = editor.choose_headline(_cfg(), _shortlist())
    assert chosen.key == "k2"
    assert chosen.headline == "Model wrote this"


def test_choose_falls_back_on_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(editor, "_call_ollama", lambda cfg, prompt: "garbage")
    chosen = editor.choose_headline(_cfg(), _shortlist())
    assert chosen.key == "k1"  # #1 by score
    assert chosen.headline == "Story one"  # its own title


def test_choose_falls_back_on_http_error(monkeypatch) -> None:
    def boom(cfg, prompt):
        raise httpx.ConnectError("no ollama")

    monkeypatch.setattr(editor, "_call_ollama", boom)
    chosen = editor.choose_headline(_cfg(), _shortlist())
    assert chosen.key == "k1"
    assert chosen.headline == "Story one"


def test_choose_out_of_range_falls_back(monkeypatch) -> None:
    # An out-of-range choice must NOT keep the model headline (that caused the
    # electrum-key/gnupg-headline mismatch). Fall back to #1 with ITS title.
    monkeypatch.setattr(
        editor, "_call_ollama", lambda cfg, prompt: '{"choice":9,"headline":"Mismatched"}'
    )
    chosen = editor.choose_headline(_cfg(), _shortlist())
    assert chosen.key == "k1"
    assert chosen.headline == "Story one"  # NOT "Mismatched"


def test_disabled_editor_skips_call(monkeypatch) -> None:
    def boom(cfg, prompt):  # must not be called
        raise AssertionError("Ollama should not be called when disabled")

    monkeypatch.setattr(editor, "_call_ollama", boom)
    chosen = editor.choose_headline(_cfg(enabled=False), _shortlist())
    assert chosen.key == "k1"
    assert chosen.headline == "Story one"


def test_headline_truncated(monkeypatch) -> None:
    long = "x" * 300
    monkeypatch.setattr(
        editor, "_call_ollama", lambda cfg, prompt: f'{{"choice":1,"headline":"{long}"}}'
    )
    chosen = editor.choose_headline(_cfg(), _shortlist())
    assert len(chosen.headline) == 120
