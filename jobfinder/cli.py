"""job-finder CLI — Slice 2 demo (plan section 12).

Produces the headline deliverable: *"top N companies where a CFO / VP-Finance
role may be forming, with why-now + evidence"*, by wiring 8-K leadership-vacuum
signals and Form D funding signals through the weighted scorer into ranked
`Opportunity` objects.

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

from jobfinder.pipeline import CompanyInputs, PipelineResult, run_pipeline_detailed
from jobfinder.schemas import Opportunity
from jobfinder.signals.extraction import RegexExtractor
from jobfinder.sources.ats import PROVIDERS, AtsClient, JobBoard, JobPosting
from jobfinder.sources.edgar import EdgarClient, Filing, FormD, RelatedPerson
from jobfinder.store import PersistResult, Store

# A fixed "now" so the demo's recency scores are reproducible.
_DEMO_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


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

    return [
        CompanyInputs(
            company_id="co-northwind",
            name="Northwind Robotics Inc.",
            eight_k=[(northwind_8k, northwind_8k_doc)],
            form_d=[(northwind_form_d_filing, northwind_form_d)],
            ats_boards=[northwind_board],
            company_fit=0.8,
        ),
        CompanyInputs(
            company_id="co-lumen",
            name="Lumen Bio Corp.",
            form_d=[(lumen_form_d_filing, lumen_form_d)],
            company_fit=0.5,
        ),
        CompanyInputs(
            company_id="co-atlas",
            name="Atlas Freight Inc.",
            eight_k=[(atlas_8k, atlas_8k_doc)],
            company_fit=0.6,
        ),
        CompanyInputs(
            company_id="co-helix",
            name="Helix Labs",
            ats_boards=[helix_board],
            company_fit=0.5,
        ),
    ]


# --------------------------------------------------------------------------- #
# Live mode: fetch real filings for given CIKs.
# --------------------------------------------------------------------------- #
def _live_companies(client: EdgarClient, ciks: list[str]) -> list[CompanyInputs]:
    companies: list[CompanyInputs] = []
    for cik in ciks:
        eight_k = [
            (f, client.fetch_document(f)) for f in client.recent_8k(cik, item="5.02")
        ]
        form_d = [(f, client.fetch_form_d(f)) for f in client.recent_form_d(cik)]
        companies.append(
            CompanyInputs(
                company_id=f"cik-{cik}",
                name=f"CIK {cik}",
                eight_k=eight_k,
                form_d=form_d,
            )
        )
    return companies


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
    header = f"Top {min(top, len(opportunities))} companies where a CFO / VP-Finance role may be forming"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        companies = _demo_companies()
        # Force the deterministic regex extractor so the demo is reproducible
        # and offline even when a GEMINI_API_KEY is present in the environment.
        result = run_pipeline_detailed(
            companies,
            observed_at=_DEMO_NOW,
            now=_DEMO_NOW,
            extractor=RegexExtractor(),
        )
    elif args.command == "live":
        if not args.cik and not args.ats:
            print("live: provide at least one --cik or --ats source.", file=sys.stderr)
            return 2
        companies: list[CompanyInputs] = []
        if args.cik:
            edgar = EdgarClient.with_user_agent(args.user_agent)
            companies.extend(_live_companies(edgar, args.cik))
        if args.ats:
            ats = AtsClient.with_user_agent(args.user_agent)
            companies.extend(_ats_companies(ats, args.ats))
        result = run_pipeline_detailed(companies)
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
