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
- [ ] Pillar I: ATS collectors (Greenhouse / Lever / Ashby)
- [ ] Persistence (Postgres) + reporter (digest output)
- [ ] Enrichment integrations (contacts, firmographics)

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
