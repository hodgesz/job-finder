"""CLI for the personal job-search tool — its OWN entrypoint.

    python -m jobfinder.jobsearch rank --alerts-dir ~/li_alerts
    python -m jobfinder.jobsearch rank --alerts-dir ~/li_alerts \
        --ats greenhouse:databricks --user-agent "job-finder you@example.com"

The ``rank`` subcommand ingests saved LinkedIn job-alert ``.eml`` files (and,
optionally, public ATS boards), de-duplicates, scores each job against the
VP-of-AI target profile, and prints a ranked, tiered, evidence-cited list. It is
fully offline when only ``--alerts-dir`` is given; ``--ats`` adds a network fetch
of the named public boards (SEC-style descriptive User-Agent required, reusing the
core ``AtsClient``). ``--gmail-label``/``--gmail-query`` read the same LinkedIn
alert emails live from the user's mailbox (read-only OAuth) instead of exported
``.eml`` files, producing the same postings.

Deliberately separate from ``jobfinder.cli`` (the core opportunity-intelligence
CLI), so the core tool is untouched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from jobfinder.jobsearch.match import rank_jobs
from jobfinder.jobsearch.models import JobMatch, RawPosting, Tier
from jobfinder.jobsearch.normalize import canonicalize
from jobfinder.jobsearch.profile import LIVE_DIMENSIONS, VP_AI_PROFILE
from jobfinder.jobsearch.sources.eml_dir import read_eml_dir
from jobfinder.jobsearch.sources.gmail import (
    CLIENT_SECRET_FILENAME,
    DEFAULT_CRED_DIR,
    GmailSource,
)
from jobfinder.sources.ats import PROVIDERS, AtsClient, JobBoard

_TIER_ORDER = {Tier.A: 0, Tier.B: 1, Tier.C: 2}


def _parse_ats_spec(spec: str) -> tuple[str, str]:
    """Parse a ``provider:token`` --ats value (mirrors jobfinder.cli)."""
    provider, _, token = spec.partition(":")
    provider = provider.strip().lower()
    token = token.strip()
    if provider not in PROVIDERS or not token:
        raise ValueError(
            f"--ats must be 'provider:token' where provider is one of "
            f"{', '.join(PROVIDERS)}; got {spec!r}."
        )
    return provider, token


def _fetch_boards(specs: list[str], user_agent: str | None) -> list[JobBoard]:
    """Fetch each public ATS board named by an --ats spec."""
    if not specs:
        return []
    if not user_agent or not user_agent.strip():
        raise ValueError(
            "--ats requires --user-agent, e.g. 'job-finder you@example.com'."
        )
    client = AtsClient.with_user_agent(user_agent)
    boards: list[JobBoard] = []
    for spec in specs:
        provider, token = _parse_ats_spec(spec)
        boards.append(client.fetch_board(provider, token))
    return boards


def _fetch_gmail_postings(label: str | None, query: str | None) -> list[RawPosting]:
    """Read LinkedIn alert emails live from Gmail when requested.

    Builds the source lazily (and only when ``--gmail-label``/``--gmail-query``
    is given), so a run without Gmail flags never touches OAuth. Raises
    ``RuntimeError`` when Gmail was requested but no credentials are on disk, so
    the user gets a clear setup message rather than a silent empty result.

    A falsy value (``None`` or an empty string) counts as "not requested", so a
    caller passing ``--gmail-label ""`` does not trigger OAuth — matching how
    ``_run_rank``'s source-presence check treats the flags via ``not args.*``.
    """
    if not label and not query:
        return []
    source = GmailSource.from_env()
    if source is None:
        raise RuntimeError(
            "--gmail-label/--gmail-query needs Gmail credentials on disk: place "
            f"{CLIENT_SECRET_FILENAME} under {DEFAULT_CRED_DIR} and authorize "
            "once (read-only). See jobfinder/jobsearch/sources/gmail.py."
        )
    return source.fetch_postings(label=label or None, query=query or None)


def render(matches: list[JobMatch], *, top: int, min_tier: Tier) -> str:
    """Render ranked matches as a human-readable, evidence-cited report."""
    shown = [
        m
        for m in matches
        if not m.rejected and _TIER_ORDER[m.tier] <= _TIER_ORDER[min_tier]
    ][:top]

    lines: list[str] = []
    header = f"Top {len(shown)} VP-of-AI matches (tier {min_tier.value}+)"
    lines.append(header)
    lines.append("=" * len(header))
    if not shown:
        lines.append("")
        lines.append("No qualifying matches found.")
        return "\n".join(lines)

    for rank, m in enumerate(shown, start=1):
        job = m.job
        lines.append("")
        lines.append(
            f"{rank}. [{m.tier.value}] {job.title}  —  {job.company}  "
            f"(score {m.score:.0f}/100)"
        )
        if job.location:
            lines.append(f"   Location: {job.location}")
        lines.append(f"   Why: {m.reason}")
        # Per-dimension breakdown of the live (evaluated) dimensions. Driven by
        # profile.LIVE_DIMENSIONS so it can't drift from what the scorer evaluates.
        live = [
            f"{d.name} {d.raw:.2f}×{d.weight:.2f}"
            for d in m.dimensions
            if d.name in LIVE_DIMENSIONS
        ]
        if live:
            lines.append(f"   Breakdown: {', '.join(live)}")
        lines.append(f"   Sources: {', '.join(job.source_kinds)}")
        lines.append(
            f"   Apply: {job.best_apply_url or '(open LinkedIn listing manually)'}"
        )
        if m.risks:
            lines.append(f"   Risks: {'; '.join(m.risks)}")
    return "\n".join(lines)


def _run_rank(args: argparse.Namespace) -> int:
    if (
        not args.alerts_dir
        and not args.ats
        and not args.gmail_label
        and not args.gmail_query
    ):
        print(
            "rank: provide at least one source "
            "(--alerts-dir, --gmail-label/--gmail-query, and/or --ats).",
            file=sys.stderr,
        )
        return 2
    try:
        raw_postings = read_eml_dir(args.alerts_dir) if args.alerts_dir else []
        raw_postings += _fetch_gmail_postings(args.gmail_label, args.gmail_query)
        boards = _fetch_boards(args.ats, args.user_agent)
    except (ValueError, NotADirectoryError, RuntimeError) as exc:
        print(f"rank: {exc}", file=sys.stderr)
        return 2

    jobs = canonicalize(raw_postings, boards)
    now = datetime.now(timezone.utc)
    matches = rank_jobs(jobs, VP_AI_PROFILE, now=now)
    print(render(matches, top=args.top, min_tier=Tier(args.min_tier)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobfinder.jobsearch",
        description="Personal VP-of-AI job-search ingestion + ranking (isolated tool).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rank = sub.add_parser(
        "rank",
        help="Ingest LinkedIn job-alert emails (+ optional ATS boards), dedupe, "
        "and rank against the VP-of-AI profile.",
    )
    rank.add_argument(
        "--alerts-dir",
        default=None,
        metavar="DIR",
        help="Directory of saved LinkedIn job-alert .eml files (parsed offline).",
    )
    rank.add_argument(
        "--ats",
        action="append",
        default=[],
        metavar="PROVIDER:TOKEN",
        help="Public ATS board as 'provider:token' (repeatable), e.g. "
        "'greenhouse:databricks'. Requires --user-agent.",
    )
    rank.add_argument(
        "--gmail-label",
        default=None,
        metavar="LABEL",
        help="Read LinkedIn alert emails live from this Gmail label, e.g. "
        "'job-alerts' (read-only OAuth; needs credentials under "
        f"{DEFAULT_CRED_DIR}). Combinable with --gmail-query.",
    )
    rank.add_argument(
        "--gmail-query",
        default=None,
        metavar="QUERY",
        help="Read LinkedIn alert emails live matching this Gmail search, e.g. "
        "'from:jobalerts-noreply@linkedin.com'. Combinable with --gmail-label.",
    )
    rank.add_argument(
        "--user-agent",
        default=None,
        help="Contact User-Agent for ATS fetches, e.g. 'job-finder you@example.com'.",
    )
    rank.add_argument(
        "-n", "--top", type=int, default=10, help="How many matches to show."
    )
    rank.add_argument(
        "--min-tier",
        choices=[t.value for t in Tier],
        default=Tier.B.value,
        help="Lowest tier to display (A best). Default B.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rank":
        return _run_rank(args)
    return 2  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
