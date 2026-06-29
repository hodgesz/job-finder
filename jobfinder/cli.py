"""job-finder CLI — Slice 2 demo (plan section 12).

Produces the headline deliverable: *"top N companies where a senior role may be
forming, with why-now + evidence"*, by wiring 8-K leadership-vacuum signals,
Form D funding signals and ATS hiring-pattern signals through the weighted
scorer into ranked `Opportunity` objects. Each opportunity's target persona is
derived from its own signals (a CFO departure → a finance leader; an Engineering
surge → an engineering leader), not fixed system-wide.

Two modes:

    job-finder demo                          run the built-in offline dataset
    job-finder live --cik 320193             fetch real SEC filings for CIK(s)
    job-finder live --ats greenhouse:stripe  read a public ATS job board
    job-finder live --cik 320193 --ats lever:netflix   both, per company

The demo mode embeds its filings and job boards inline so it runs anywhere with
zero network and zero API key — the regex fallback handles 8-K classification.
`live` mode requires at least one source (--cik and/or --ats) and a contact
User-Agent (see --user-agent); SEC mandates a contact email when fetching
filings.

Built on argparse (stdlib) — no new dependency.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone

from jobfinder.fit import CandidateProfile, Firmographics
from jobfinder.pipeline import CompanyInputs, PipelineResult, run_pipeline_detailed
from jobfinder.reporter import render_digest
from jobfinder.schemas import Opportunity
from jobfinder.signals.extraction import RegexExtractor
from jobfinder.signals.form_d import FORM_D_RECENCY_HORIZON_DAYS
from jobfinder.sources.ats import PROVIDERS, AtsClient, JobBoard, JobPosting
from jobfinder.sources.edgar import EdgarClient, Filing, FormD, RelatedPerson
from jobfinder.sources.firmographics import firmographics_from_sec
from jobfinder.store import PersistResult, Store

# A fixed "now" so the demo's recency scores are reproducible.
_DEMO_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)

# The person we're hunting roles for, used by the demo to *derive* each
# company's fit instead of hand-setting a magic number: a finance/ops leader who
# wants an early-growth (Series A/B) robotics or industrial company of ~50-300
# people. The demo firmographics below score against this.
_DEMO_PROFILE = CandidateProfile(
    target_sectors=("robotics", "industrial", "logistics"),
    target_stages=("series_a", "series_b"),
    min_employees=50,
    max_employees=300,
)


# --------------------------------------------------------------------------- #
# Built-in offline demo dataset.
# --------------------------------------------------------------------------- #
def _demo_posting(
    pid: str,
    title: str,
    *,
    department: str | None = None,
    days_ago: int = 5,
) -> JobPosting:
    """A demo job posting dated relative to the fixed demo `now`."""
    return JobPosting(
        id=pid,
        title=title,
        department=department,
        updated_at=_DEMO_NOW - timedelta(days=days_ago),
        url=f"https://jobs.example.com/{pid}",
    )


def _demo_companies() -> list[CompanyInputs]:
    """Four synthetic companies exercising the full signal matrix.

    - Northwind: fresh $55M raise, a CFO departure with no successor named, AND
      a hiring surge on its job board -> all four pillars fire, ranks first.
    - Lumen Bio: large raise only -> funding signal, mid rank.
    - Atlas Freight: CFO vacuum only -> leadership signal, lower than the
      funding+vacuum company.
    - Helix Labs: no SEC filings at all, only a public ATS board with a
      department surge and a founding leadership req -> demonstrates the new
      Pillar I (hiring/strategic) lighting up standalone.
    """
    northwind_8k = Filing(
        cik="1950000",
        accession_number="0001950000-26-000005",
        form="8-K",
        filing_date=date(2026, 4, 20),
        report_date=date(2026, 4, 18),
        items=["5.02"],
        primary_document="nw_8k.htm",
    )
    northwind_8k_doc = (
        "Item 5.02 Departure of Directors or Certain Officers; Election of "
        "Directors; Appointment of Certain Officers; Compensatory Arrangements "
        "of Certain Officers. On April 18, 2026, Ada Marsh, the Chief Financial "
        "Officer of Northwind Robotics Inc., notified the Board of her "
        "resignation, effective April 30, 2026. The Company has commenced a "
        "search for a permanent successor."
    )
    northwind_form_d_filing = Filing(
        cik="1950000",
        accession_number="0001950000-26-000003",
        form="D",
        filing_date=date(2026, 4, 20),
        report_date=None,
        items=[],
        primary_document="primary_doc.xml",
    )
    northwind_form_d = FormD(
        issuer_cik="0001950000",
        issuer_name="Northwind Robotics Inc.",
        accession_number="0001950000-26-000003",
        total_offering_amount=60_000_000.0,
        total_amount_sold=55_000_000.0,
        total_remaining=5_000_000.0,
        industry_group="Technology",
        related_persons=[
            RelatedPerson(name="Ada Marsh", relationships=["Executive Officer"]),
        ],
    )

    lumen_form_d_filing = Filing(
        cik="1960000",
        accession_number="0001960000-26-000002",
        form="D",
        filing_date=date(2026, 5, 15),
        report_date=None,
        items=[],
        primary_document="primary_doc.xml",
    )
    lumen_form_d = FormD(
        issuer_cik="0001960000",
        issuer_name="Lumen Bio Corp.",
        accession_number="0001960000-26-000002",
        total_offering_amount=40_000_000.0,
        total_amount_sold=40_000_000.0,
        total_remaining=0.0,
        industry_group="Biotechnology",
    )

    atlas_8k = Filing(
        cik="1970000",
        accession_number="0001970000-26-000007",
        form="8-K",
        filing_date=date(2026, 3, 2),
        report_date=date(2026, 2, 28),
        items=["5.02"],
        primary_document="atlas_8k.htm",
    )
    atlas_8k_doc = (
        "Item 5.02 Departure of Directors or Certain Officers. On February 28, "
        "2026, the Chief Financial Officer of Atlas Freight Inc. stepped down "
        "from his role. The Board is evaluating candidates for a permanent "
        "replacement."
    )

    # Northwind also has an active job board: a finance build-out behind its CFO
    # departure, so all four pillars line up on one company.
    northwind_board = JobBoard(
        provider="greenhouse",
        token="northwind",
        url="https://boards.greenhouse.io/northwind",
        postings=[
            _demo_posting("nw1", "Senior Accountant", department="Finance"),
            _demo_posting("nw2", "FP&A Manager", department="Finance"),
            _demo_posting("nw3", "Revenue Operations Lead", department="Finance"),
            _demo_posting("nw4", "Controller", department="Finance"),
            _demo_posting("nw5", "Account Executive", department="Sales"),
        ],
    )

    # Helix Labs: no SEC presence, only a public board. A clear Engineering
    # surge plus an explicit founding leadership req -> Pillar I standalone.
    helix_board = JobBoard(
        provider="lever",
        token="helix",
        url="https://jobs.lever.co/helix",
        postings=[
            _demo_posting(
                "hx1", "Founding Engineer, Platform", department="Engineering"
            ),
            _demo_posting("hx2", "Senior Backend Engineer", department="Engineering"),
            _demo_posting("hx3", "ML Engineer", department="Engineering"),
            _demo_posting("hx4", "Infrastructure Engineer", department="Engineering"),
            _demo_posting("hx5", "Head of Data", department="Data"),
        ],
    )

    # Firmographics replace the old hand-set company_fit magic numbers: each
    # company's fit is now *derived* by scoring these against `_DEMO_PROFILE`
    # (early-growth robotics/industrial, ~50-300 ppl). Northwind (Series B
    # robotics, 180) is a perfect match; Atlas (Series A logistics, 240) also
    # matches on all three; Lumen (Series C biotech, 90) is off-sector with an
    # adjacent stage; Helix (seed AI, 35) is off-sector — its only saving grace a
    # near-miss stage/size — so it lands lowest.
    return [
        CompanyInputs(
            company_id="co-northwind",
            name="Northwind Robotics Inc.",
            eight_k=[(northwind_8k, northwind_8k_doc)],
            form_d=[(northwind_form_d_filing, northwind_form_d)],
            ats_boards=[northwind_board],
            firmographics=Firmographics(
                sector="Robotics", funding_stage="Series B", employee_count=180
            ),
        ),
        CompanyInputs(
            company_id="co-lumen",
            name="Lumen Bio Corp.",
            form_d=[(lumen_form_d_filing, lumen_form_d)],
            firmographics=Firmographics(
                sector="Biotechnology", funding_stage="Series C", employee_count=90
            ),
        ),
        CompanyInputs(
            company_id="co-atlas",
            name="Atlas Freight Inc.",
            eight_k=[(atlas_8k, atlas_8k_doc)],
            firmographics=Firmographics(
                sector="Logistics", funding_stage="Series A", employee_count=240
            ),
        ),
        CompanyInputs(
            company_id="co-helix",
            name="Helix Labs",
            ats_boards=[helix_board],
            firmographics=Firmographics(
                sector="Artificial Intelligence",
                funding_stage="Seed",
                employee_count=35,
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Live mode: fetch real filings for given CIKs.
# --------------------------------------------------------------------------- #
def _live_companies(
    client: EdgarClient, ciks: list[str], *, now: datetime
) -> list[CompanyInputs]:
    # One run clock drives both the Form D pre-filter cutoff and the signal-level
    # recency floor, so a slow multi-CIK fetch can't age otherwise-identical
    # filings against drifting instants. `now` is passed on to the pipeline too.
    form_d_since = (now - timedelta(days=FORM_D_RECENCY_HORIZON_DAYS)).date()
    companies: list[CompanyInputs] = []
    for cik in ciks:
        # One fetch of the submissions index serves all three needs (filer info,
        # 8-K and Form D filing lists), so a live CIK makes a single request to
        # SEC's rate-limited endpoint rather than three identical ones.
        info, filings = client.company_submissions(cik)
        eight_k = [
            (f, client.fetch_document(f))
            for f in client.recent_8k(cik, item="5.02", filings=filings)
        ]
        # Only fetch Form D documents recent enough to still carry a funding
        # signal: the signal module discards anything past its recency horizon,
        # so fetching older XML just to drop it wastes SEC requests.
        form_d = [
            (f, client.fetch_form_d(f))
            for f in client.recent_form_d(cik, filings=filings, since=form_d_since)
        ]
        # Derive firmographics from data we just fetched: the filer's SIC sector
        # (from the submissions header), falling back to a Form D industry group.
        # Funding stage/headcount aren't in SEC filings, so they stay neutral.
        firmographics = firmographics_from_sec(info, form_d=[fd for _, fd in form_d])
        companies.append(
            CompanyInputs(
                company_id=f"cik-{cik}",
                # Prefer the real entity name from the index over a "CIK <n>" stub.
                name=info.name or f"CIK {cik}",
                eight_k=eight_k,
                form_d=form_d,
                firmographics=firmographics,
            )
        )
    return companies


def _profile_from_args(args: argparse.Namespace) -> CandidateProfile | None:
    """Build a CandidateProfile from the live `--target-*`/`--*-employees` flags.

    Returns None when the user supplied none of them, so live mode keeps the
    pre-Slice-9 behaviour (the literal neutral company_fit) unless a profile is
    actually requested.
    """
    if not (
        args.target_sector
        or args.target_stage
        or args.min_employees is not None
        or args.max_employees is not None
    ):
        return None
    return CandidateProfile(
        target_sectors=tuple(args.target_sector),
        target_stages=tuple(args.target_stage),
        min_employees=args.min_employees,
        max_employees=args.max_employees,
    )


def _parse_ats_spec(spec: str) -> tuple[str, str]:
    """Parse a ``provider:token`` --ats argument, e.g. ``greenhouse:stripe``."""
    provider, _, token = spec.partition(":")
    provider = provider.strip().lower()
    token = token.strip()
    if provider not in PROVIDERS or not token:
        raise ValueError(
            f"--ats must be 'provider:token' where provider is one of "
            f"{', '.join(PROVIDERS)}; got {spec!r}."
        )
    return provider, token


def _ats_companies(client: AtsClient, specs: list[str]) -> list[CompanyInputs]:
    """Build one company per ``provider:token`` board spec."""
    companies: list[CompanyInputs] = []
    for spec in specs:
        provider, token = _parse_ats_spec(spec)
        board = client.fetch_board(provider, token)
        companies.append(
            CompanyInputs(
                company_id=f"ats-{provider}-{token}",
                name=f"{token} ({provider})",
                ats_boards=[board],
            )
        )
    return companies


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def render(
    opportunities: list[Opportunity],
    companies: list[CompanyInputs],
    *,
    top: int,
) -> str:
    """Render ranked opportunities as a human-readable report."""
    names = {c.company_id: (c.name or c.company_id) for c in companies}
    lines: list[str] = []
    # The target persona is now derived per-opportunity from its signals (an
    # Engineering surge reads as an engineering leader, a CFO departure as a
    # finance leader), so the header no longer claims a single fixed persona —
    # each row prints its own under "Target:".
    header = f"Top {min(top, len(opportunities))} companies where a senior role may be forming"
    lines.append(header)
    lines.append("=" * len(header))
    if not opportunities:
        lines.append("")
        lines.append("No qualifying opportunities found.")
        return "\n".join(lines)

    for rank, opp in enumerate(opportunities[:top], start=1):
        name = names.get(opp.company_id, opp.company_id)
        lines.append("")
        lines.append(
            f"{rank}. {name}  —  score {opp.score:.2f}  "
            f"(confidence {opp.confidence:.0%}, urgency {opp.urgency:.0%})"
        )
        lines.append(f"   Target: {opp.target_persona}")
        lines.append(f"   Why now: {opp.why_now}")
        lines.append(f"   Next: {opp.recommended_next_action}")
        lines.append(
            f"   Evidence (supporting signals): {', '.join(opp.supporting_signal_ids)}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Persistence.
# --------------------------------------------------------------------------- #
def _store_url(db: str) -> str:
    """Turn a --db argument into a SQLAlchemy URL.

    A bare path or filename (``runs.db``, ``/tmp/jf.db``) becomes a local SQLite
    URL; anything already containing a ``://`` scheme (e.g. a Postgres URL) is
    passed through untouched, so the same flag drives both backends.
    """
    return db if "://" in db else f"sqlite+pysqlite:///{db}"


def render_persistence(result: PersistResult, db: str) -> str:
    """One-line summary of what a run wrote, distinguishing new from recurring."""
    return (
        f"Persisted to {db}: "
        f"signals {result.signals_inserted} new / {result.signals_updated} updated, "
        f"opportunities {result.opportunities_inserted} new / "
        f"{result.opportunities_updated} updated."
    )


# --------------------------------------------------------------------------- #
# argparse wiring.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-finder",
        description="Rank companies where a senior finance role may be forming.",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=5, help="How many companies to show."
    )

    # Options shared by every subcommand. Kept on a parent parser so they can be
    # given after the subcommand (`demo --db runs.db`), which reads naturally.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db",
        default=None,
        help=(
            "Persist this run's signals and opportunities. Accepts a SQLite "
            "path ('runs.db') or any SQLAlchemy URL "
            "('postgresql+psycopg://user:pw@host/db'). Re-runs upsert by id so "
            "history accumulates without duplicating."
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "demo",
        parents=[common],
        help="Run the built-in offline demo dataset (no network).",
    )

    live = sub.add_parser(
        "live",
        parents=[common],
        help="Fetch real filings (--cik) and/or public ATS boards (--ats).",
    )
    live.add_argument(
        "--cik", action="append", default=[], help="SEC CIK (repeatable)."
    )
    live.add_argument(
        "--ats",
        action="append",
        default=[],
        metavar="PROVIDER:TOKEN",
        help=(
            "Public ATS board as 'provider:token' (repeatable), e.g. "
            "'greenhouse:stripe', 'lever:netflix', 'ashby:openai'."
        ),
    )
    live.add_argument(
        "--user-agent",
        required=True,
        help="Contact User-Agent, e.g. 'job-finder you@example.com' (SEC requires "
        "a contact email when fetching filings).",
    )
    # Candidate profile: drives the derived company_fit. Sectors are scored
    # against the firmographics derived from the SEC filings (the SIC sector).
    # Stage/size are accepted but cannot influence the live SCORE yet: SEC
    # filings disclose neither, so those firmographic fields are None and score
    # neutral (the fit model never penalises a dimension it can't see). They will
    # engage once a richer enrichment source populates those fields. Omit all of
    # these and fit falls back to the neutral literal (the pre-Slice-9 behaviour).
    live.add_argument(
        "--target-sector",
        action="append",
        default=[],
        metavar="SECTOR",
        help="A sector the candidate targets (repeatable), e.g. 'robotics'. "
        "Matched whole-word against each company's derived SIC sector.",
    )
    live.add_argument(
        "--target-stage",
        action="append",
        default=[],
        metavar="STAGE",
        help="A funding stage the candidate targets (repeatable), e.g. "
        "'series_b'. SEC filings carry no funding stage, so this scores neutral "
        "until stage enrichment lands.",
    )
    live.add_argument(
        "--min-employees",
        type=int,
        default=None,
        help="Lower bound of the target headcount band. SEC filings carry no "
        "headcount, so this scores neutral until headcount enrichment lands.",
    )
    live.add_argument(
        "--max-employees",
        type=int,
        default=None,
        help="Upper bound of the target headcount band. SEC filings carry no "
        "headcount, so this scores neutral until headcount enrichment lands.",
    )

    report = sub.add_parser(
        "report",
        help="Render a cross-run digest from a persisted --db (no network).",
    )
    report.add_argument(
        "--db",
        required=True,
        help=(
            "The store to read (a SQLite path like 'runs.db' or any SQLAlchemy "
            "URL). Reports over whatever past runs were persisted there."
        ),
    )
    report.add_argument(
        "--since",
        default=None,
        metavar="DATE",
        help=(
            "Diff cutoff as an ISO date/datetime (e.g. '2026-06-01'). "
            "Opportunities first seen on/after it are flagged NEW and only "
            "signals first seen on/after it are listed. Omit for a plain ranked "
            "standings digest."
        ),
    )
    return parser


def _parse_since(value: str) -> datetime:
    """Parse a ``--since`` ISO date or datetime into a tz-aware UTC datetime.

    A bare date ('2026-06-01') is read as midnight UTC; a naive datetime is
    assumed UTC, matching how the store normalises its timestamps.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "report":
        return _report(args.db, since=args.since, top=args.top)

    if args.command == "demo":
        companies = _demo_companies()
        # Force the deterministic regex extractor so the demo is reproducible
        # and offline even when a GEMINI_API_KEY is present in the environment.
        result = run_pipeline_detailed(
            companies,
            candidate_profile=_DEMO_PROFILE,
            observed_at=_DEMO_NOW,
            now=_DEMO_NOW,
            extractor=RegexExtractor(),
        )
    elif args.command == "live":
        if not args.cik and not args.ats:
            print("live: provide at least one --cik or --ats source.", file=sys.stderr)
            return 2
        # One reference instant for the whole run: the Form D pre-filter cutoff
        # and the recency floor/decay the pipeline applies share it, so they
        # can't drift across a slow multi-CIK fetch.
        now = datetime.now(timezone.utc)
        companies: list[CompanyInputs] = []
        if args.cik:
            edgar = EdgarClient.with_user_agent(args.user_agent)
            companies.extend(_live_companies(edgar, args.cik, now=now))
        if args.ats:
            ats = AtsClient.with_user_agent(args.user_agent)
            companies.extend(_ats_companies(ats, args.ats))
        # A candidate profile (if any --target-* flags were given) turns each
        # company's derived firmographics into a real company_fit; without one,
        # fit stays the neutral literal.
        result = run_pipeline_detailed(
            companies,
            candidate_profile=_profile_from_args(args),
            observed_at=now,
            now=now,
        )
    else:  # pragma: no cover - argparse enforces a valid command
        return 2

    print(render(result.opportunities, companies, top=args.top))
    if args.db:
        persisted = _persist(result, args.db)
        print()
        print(render_persistence(persisted, args.db))
    return 0


def _persist(result: PipelineResult, db: str) -> PersistResult:
    store = Store(_store_url(db))
    return store.persist_run(result.signals, result.opportunities)


def _report(db: str, *, since: str | None, top: int) -> int:
    """Render a cross-run digest from a persisted store. No network, no clock
    baked in beyond `now` (utcnow) for relative ages."""
    since_dt = _parse_since(since) if since else None
    # create=True (the default) keeps the offline contract: pointed at a fresh
    # or empty store the digest renders "No opportunities on file." rather than
    # erroring, and construction also runs the additive migration that brings a
    # store written by an earlier slice up to the current schema.
    store = Store(_store_url(db))
    diff = store.diff(since=since_dt)
    print(render_digest(diff, now=datetime.now(timezone.utc), top=top))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
