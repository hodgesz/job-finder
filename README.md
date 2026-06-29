# job-finder

A multi-agent Python system for surfacing **hidden-market job opportunities** by
tracking the leading signals that precede a hire — often weeks or months before a
role is ever posted publicly.

The premise: roughly 70–80% of senior roles (VP, SVP, C-suite) are never advertised.
They're filled through internal networks and retained search before a job board ever
sees them. But organizational change leaves a data trail — hiring surges, capital
raises, executive departures — across public and regulatory sources. `job-finder`
collects those signals, correlates them, scores intent, and turns them into a
prioritized, actionable target list.

> **Status:** early scaffolding. The architecture below is the roadmap; see
> [Roadmap](#roadmap) for what exists today versus what's planned.

## How it works

`job-finder` is organized as a set of cooperating agents, each owning one stage of
the pipeline:

```
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Collectors │ →  │  Normalizer  │ →  │  Correlator  │ →  │   Scorer     │
  │ (per signal)│    │ (dedup/clean)│    │ (join by org)│    │ (intent rank)│
  └─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    ↓
                                                              ┌──────────────┐
                                                              │   Reporter   │
                                                              │ (digest/feed)│
                                                              └──────────────┘
```

- **Collector agents** pull raw signals from a single source each (one per pillar below).
- **Normalizer** deduplicates and cleans records into a common schema keyed by company.
- **Correlator** joins signals for the same organization across sources and time.
- **Scorer** computes a weighted composite intent score (multiple concurrent signals
  rank highest).
- **Reporter** emits a prioritized digest of high-intent targets.

## The three signal pillars

### Pillar I — Hiring-pattern analysis (job boards & ATS)

Lower-level requisitions are a real-time map of a company's strategic priorities. A
surge of SDR/BDR listings with no VP of Sales in place mathematically implies an
imminent executive search. Phrases like *"first hire"* or *"greenfield"* signal a new
division forming. Sources: public ATS feeds (Greenhouse, Lever, Ashby, Workable) and
aggregated job-posting datasets. Cross-run diffing yields per-department headcount
velocity.

### Pillar II — Private capital deployment (SEC Form D)

Series B/C funding is a definitive signal that a company must build go-to-market
infrastructure — and has the budget to hire it. SEC **Form D** filings (via EDGAR)
capture raised capital within 15 days of first sale, well ahead of the curated press
release. We parse `totalAmountSold`, `totalRemaining`, `industryGroup`, and the
related-persons list to flag freshly funded organizations inside the optimal outreach
window.

### Pillar III — Executive transitions (SEC Form 8-K, Item 5.02)

Public companies must file an **8-K Item 5.02** within four business days of an
executive departure or appointment. A 5.02(b) departure filed *without* a matching
5.02(c) appointment is a leadership vacuum with no succession plan — maximum urgency.
Correlating 5.02 with items like 4.01/4.02 (auditor changes, restatements) diagnoses
*why* the seat opened, which shapes the kind of candidate that fits.

### Composite intent scoring

| Signal | Source | Significance |
| --- | --- | --- |
| New funding round | SEC Form D / Crunchbase | Unlocked budget; mandate to scale |
| Entry-level hiring surge | ATS scrapers | Operational strain needing leadership |
| Tech-stack overhaul | Job-description NLP | Strategic pivot; new leadership restructuring |
| Executive departure | SEC 8-K Item 5.02 | Immediate vacuum; high leverage |
| Recent executive hire | Job-change signals | First-90-days decision window |

One signal is a lead. Three or more concurrently is the highest tier of predictive
hiring intent.

## Architecture: modular monolith, with the first A2A seam

The pipeline runs in-process today (collectors → scorer → CLI), which keeps it
simple to test and reason about. The target architecture, though, is a Google
ADK + Gemini orchestrator that delegates to specialist services over the
[A2A protocol](https://google.github.io/adk-docs/a2a/). The 8-K specialist is
the first module to cross that seam:

```
  ADK + Gemini orchestrator  ──A2A──▶  8-K signal specialist
  (RemoteA2aAgent sub-agent)           (LangGraph StateGraph behind to_a2a())
                                        └─ reuses the in-process extractor,
                                           speaks the Signal schema on the wire
```

`jobfinder/a2a/` holds this extraction: a LangGraph graph wraps the existing
8-K extractor, a thin ADK `BaseAgent` exposes it via `to_a2a()` (serving an
agent card at `/.well-known/agent-card.json`), and an `LlmAgent` orchestrator
consumes it as a `RemoteA2aAgent`. The wire contract (`jobfinder/a2a/contract.py`)
is built on the framework-free `Signal`/`Evidence` schema — the domain object
*is* the contract. The in-process path and CLI are unchanged; A2A is an
additional consumption path, not a replacement.

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) and targets the Python version
pinned in `.python-version`.

```bash
# Install dependencies into a local .venv
uv sync

# Run the entry point
uv run python main.py

# Rank companies where a senior finance role may be forming.
# `demo` uses a built-in offline dataset (no network, no API key):
uv run python -m jobfinder.cli demo
# `live` fetches real EDGAR filings (SEC requires a contact User-Agent):
uv run python -m jobfinder.cli live --cik 320193 --user-agent "job-finder you@example.com"
# Add a candidate profile to derive a real company_fit: target sectors are
# matched against each filer's SIC sector (from the same submissions fetch).
# (Stage/headcount flags exist but score neutral until richer enrichment lands,
# since SEC filings disclose neither.)
uv run python -m jobfinder.cli live --cik 320193 \
  --user-agent "job-finder you@example.com" \
  --target-sector "electronic" --target-sector "software"

# Persist a run so history accumulates (--db takes a SQLite path or any
# SQLAlchemy URL; re-runs upsert by id rather than duplicating):
uv run python -m jobfinder.cli demo --db runs.db
uv run python -m jobfinder.cli live --cik 320193 \
  --user-agent "job-finder you@example.com" \
  --db "postgresql+psycopg://user:pw@localhost/jobfinder"

# Report a cross-run digest from a persisted store (no network). Without
# --since it prints the current ranked standings; with --since it diffs against
# that cutoff, flagging new vs recurring opportunities, score movement, and the
# signals that newly appeared:
uv run python -m jobfinder.cli report --db runs.db
uv run python -m jobfinder.cli report --db runs.db --since 2026-06-01

# Run the tests
uv run pytest
```

## Development

```bash
uv run pytest            # tests
uvx ruff check .         # lint
uvx ruff format .        # format
```

CI runs lint, format-check, and tests on every push and pull request to `main`.
`main` is protected — contribute via a pull request that passes CI.

## Roadmap

- [x] Project scaffolding, CI, and tooling
- [x] Common signal schema (`Signal` / `Opportunity` / `Evidence` / `Company`)
- [x] Pillar III: SEC 8-K Item 5.02 collector (EDGAR) + LLM extraction pass
- [x] Pillar II: SEC Form D collector (EDGAR)
- [x] Composite intent scorer + ranked-opportunity CLI demo
- [x] Persistence layer (SQLite/Postgres via SQLAlchemy) — runs accumulate for cross-run diffing
- [x] First A2A extraction: 8-K specialist as a LangGraph service behind `to_a2a()`, consumed by an ADK + Gemini `RemoteA2aAgent` orchestrator
- [x] Pillar I: ATS collectors (Greenhouse / Lever / Ashby) — hiring-velocity / department-surge / greenfield-team signals, activating the `hiring_velocity` + `strategic_language` scorer components
- [x] Reporter (cross-run digest) — `report --db [--since]` turns accumulated runs into a prioritized "what changed since last week" view (new vs recurring, score movement, newly-appeared signals)
- [x] Firmographic `company_fit`: a candidate-vs-company fit model (`jobfinder.fit`), wired into live runs by deriving each filer's sector from its SEC SIC classification (`live --target-sector`)
- [x] Listed-roles corroboration (`jobfinder.listings`): surfaces the public ATS reqs already fetched next to each opportunity, flagging the ones *in the same function* as the target persona — the hidden seat corroborated by live, listed roles
- [ ] Enrichment integrations (contacts, richer firmographics — funding stage, headcount)

## Legal & ethical use

This is a research and personal-productivity tool. Use it responsibly:

- **Respect terms of service and robots.txt.** Some platforms (notably LinkedIn)
  prohibit automated scraping; prefer official APIs and licensed data providers, and
  do not use this project to violate a site's terms.
- **SEC EDGAR data is public** and intended for programmatic access — but follow the
  SEC's fair-access guidelines (rate limits and a descriptive User-Agent).
- **Handle personal data lawfully.** Enrichment and outreach features must comply with
  applicable privacy and anti-spam laws (e.g. GDPR, CAN-SPAM).

You are responsible for how you use this software.

## License

Licensed under the [Apache License 2.0](LICENSE).
