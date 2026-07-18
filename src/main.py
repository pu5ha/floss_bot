"""CLI entrypoint. Subcommands: run-once | poll-votes | calibrate | refresh-taste."""

from __future__ import annotations

import argparse
import logging
import sys

from .calibrate import run_calibrate
from .config import load_config
from .db import connect, init_schema
from .pipeline.editor import choose_headline
from .pipeline.orchestrate import persist_items, run_cycle, send_plan
from .pipeline.score import is_forced
from .summarize import summarize_story
from .taste import refresh_taste
from .telegram import is_configured
from .votes import poll_votes

log = logging.getLogger("floss-bot")


def _print_shortlist(shortlist: list, floor: float) -> None:
    print(f"\nshortlist ({len(shortlist)} stories by relevance, floor {floor:.2f}):\n")
    for rank, st in enumerate(shortlist, 1):
        top = st.top_item
        more = f" +{st.corroboration - 1} more" if st.corroboration > 1 else ""
        flags = " ⚑FORCED" if is_forced(st) else ""
        floor_flag = "" if st.score >= floor else " (below floor)"
        print(f"  #{rank}  relevance {st.score:.3f}{floor_flag}{flags}")
        print(f"      {top.title}")
        print(f"      {top.source}{more} · closest to: {st.nearest}")
        print(f"      {top.url}")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence chatty third-party libs (HF cache-metadata HEADs, device notices).
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_pick(chosen, summary: str, prefix: str = "editor's pick") -> None:
    print(f"\n{prefix}:")
    print(f"  📰 {chosen.headline}")
    print(f"     {chosen.top_item.source} · score {chosen.score:.3f} · {chosen.top_item.event_type}")
    print(f"     {summary}")
    print(f"     {chosen.top_item.url}")


def cmd_run_once(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        plan = run_cycle(cfg, conn)
        _print_shortlist(plan.shortlist, cfg.floor)

        if args.dry:
            if plan.quiet:
                print(f"\n{plan.reason} — nothing to pick.")
            else:
                chosen = choose_headline(cfg, plan.sendable)
                _print_pick(chosen, summarize_story(cfg, chosen))
            print("\n(--dry: nothing sent or persisted.)")
            return 0

        # Non-dry: record history (novelty + dedup) even on quiet days.
        persist_items(cfg, conn, plan)

        if plan.quiet:
            print(f"\n{plan.reason} — sent nothing (this is correct behavior).")
        elif not is_configured(cfg):
            chosen = choose_headline(cfg, plan.sendable)
            _print_pick(chosen, summarize_story(cfg, chosen), "would send")
            print("\n(Telegram not configured — set TELEGRAM_BOT_TOKEN / "
                  "TELEGRAM_CHAT_ID in .env. Recorded history, sent nothing.)")
        else:
            chosen = send_plan(cfg, conn, plan)
            if chosen is not None:
                print(f"\nsent: 📰 {chosen.headline}  ({chosen.top_item.source})")
            else:
                print("\nnothing sent (already claimed today).")
    finally:
        conn.close()
    return 0


def cmd_poll_votes(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        poll_votes(cfg, conn, once=args.once)
    finally:
        conn.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        run_calibrate(cfg, conn, days=args.days)
    finally:
        conn.close()
    return 0


def cmd_refresh_taste(args: argparse.Namespace) -> int:
    cfg = load_config()
    conn = connect(cfg.db_path)
    try:
        init_schema(conn)
        counts = refresh_taste(cfg, conn)
    finally:
        conn.close()
    log.info(
        "taste profile: %d seeds, %d neg-seeds, %d 👍, %d 👎",
        counts.get("seed", 0), counts.get("negseed", 0),
        counts.get("pos", 0), counts.get("neg", 0),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floss-bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-once", help="one cycle: pick + send one headline or nothing")
    p_run.add_argument(
        "--dry", action="store_true", help="preview the pick without sending (M7+)"
    )
    p_run.set_defaults(func=cmd_run_once)

    p_poll = sub.add_parser("poll-votes", help="Telegram 👍/👎 callback loop")
    p_poll.add_argument(
        "--once", action="store_true", help="drain pending votes and exit (for cron)"
    )
    p_poll.set_defaults(func=cmd_poll_votes)

    p_cal = sub.add_parser("calibrate", help="replay N days: shortlists + picks, send nothing")
    p_cal.add_argument("--days", type=int, default=14)
    p_cal.set_defaults(func=cmd_calibrate)

    p_ref = sub.add_parser(
        "refresh-taste", help="rebuild the taste profile from seeds + 👍/👎 votes"
    )
    p_ref.set_defaults(func=cmd_refresh_taste)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
