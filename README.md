# FLOSS / d-acc "Headline of the Day" Bot

Watches the free/libre + privacy + decentralization + open-hardware world across a
curated set of **news** feeds, figures out the single most interesting thing that
happened each day, and sends **one headline** via Telegram — or **nothing**, on a
genuinely quiet day.

It curates **news**, not software version bumps. Stories are ranked by **relevance**
to a taste profile (embedding similarity to example headlines you care about),
across four domains:

- **Privacy & digital rights** (EFF, Access Now, EDRi, Privacy International, The Markup, 404 Media)
- **Open hardware & new products** (Hackaday, CNX Software, SparkFun, Adafruit, Olimex, Framework, PINE64)
- **FLOSS world news & analysis** (LWN, It's FOSS, The Register, Phoronix)
- **Cryptography & security research** (Schneier, Trail of Bits, NIST, Quanta)
- plus decentralization/self-hosting (Matrix, Mastodon, Nextcloud, selfh.st) and flagship project blogs.

A **local LLM via Ollama** picks the single most significant story from a shortlist,
writes the headline, and writes a 3-sentence summary. Every 👍/👎 sharpens the taste
profile over time. No cloud LLM, no API keys — the only network credential is the
Telegram bot token.

## How ranking works

1. **Fetch** all feeds (generic RSS/Atom puller; one feed failing never aborts the run).
2. **Cluster** items describing the same story (corroboration across sources).
3. **Score by relevance**: `relevance = max cosine(story, liked-seed) − λ·max cosine(story, disliked-seed)`,
   using `all-MiniLM-L6-v2` embeddings and your `seeds/ground_truth.txt`.
4. **Rules layer**: routine version bumps are excluded; a critical security advisory
   on a flagship project is force-included.
5. **Quiet-day floor**: nothing goes out unless the day's best story clears `floor`.
6. **Editor**: the local LLM picks one story from the shortlist and writes the headline + summary.
7. **Send** one headline (idempotent; one per day max).

## Install

```bash
cd floss-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml    # edit as needed
cp .env.example .env                  # fill TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
ollama pull llama3.1:8b               # or any 7B–14B; set it in config.yaml editor.model
```

## Usage

```bash
python -m src.main run-once           # send one interesting headline (or nothing)
python -m src.main run-once --dry     # preview the pick + summary, no send/writes
python -m src.main calibrate --days 14   # replay N days: would-send headlines + floor sweep
python -m src.main refresh-taste      # rebuild the taste profile from seeds + votes
python -m src.main poll-votes --once  # drain 👍/👎 votes and fold them into taste
```

## Tuning what it finds

- **`seeds/ground_truth.txt`** — the heart of relevance. Positive lines are headlines
  you'd love; lines under `[NEGATIVE]` are things you don't want (version bumps,
  benchmarks, off-topic posts). Edit freely, then `refresh-taste`.
- **`config.yaml`** — `floor` (quiet-day bar; lower = more sends), `lambda_neg`
  (dislike penalty), `cluster_sim`, `max_age_hours`, `max_per_day`, the Ollama `editor`.
- **`config/feeds.yaml`** — the source list. Add/remove feeds (each: name, url, layer).
- Run `calibrate` after any change to see the effect before going live.
- Your **👍/👎** on sent headlines fold into the taste profile automatically via `poll-votes`.

## Architecture

```
src/
  config.py         config.yaml + feeds.yaml + .env -> typed Config
  models.py         Item, Story dataclasses
  db.py             SQLite (data/floss.db): items, stories, sent, votes, taste, source_state
  embed.py          MiniLM; normalized vectors (clustering + relevance)
  taste.py          seed/vote taste profile (relevance anchors)
  sources/          base.py (polite HTTP), feeds.py (generic RSS/Atom); anitya.py retained but unused
  pipeline/         normalize.py, cluster.py, score.py (relevance), editor.py (Ollama)
  summarize.py      Ollama 3-sentence summary
  telegram.py       send_headline(), vote helpers
  votes.py          poll-votes loop -> folds votes into taste
  calibrate.py      replay N days, show would-send headlines, send nothing
  main.py           CLI: run-once | poll-votes | calibrate | refresh-taste
```

## Deployment

See `deploy/`: `crontab.example` (daily `run-once` + periodic `poll-votes --once`) and
`floss-votes.service` (systemd user unit for the continuous vote loop). **Ollama must be
running locally**, so a home / always-on box fits better than an ephemeral CI runner.

## Known limitations

- Relevance is only as good as your seeds — the first weeks lean on `ground_truth.txt`;
  the 👍/👎 loop sharpens it toward your taste.
- Clustering is heuristic; it occasionally splits/merges stories imperfectly.
- Open hardware and research are the thinnest domains for clean feeds; some days are
  legitimately quiet.
- The local LLM's judgment is weaker than a frontier model's, but it only picks among a
  handful of pre-vetted, relevance-ranked stories and writes a line — a job small models
  handle well. Swap `editor.model` up if your hardware allows.
- Software releases are intentionally excluded (this is a news bot). Re-enable the Anitya
  poller in `src/pipeline/orchestrate.py` if you ever want version tracking back.
