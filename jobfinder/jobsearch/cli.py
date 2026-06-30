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

from sqlalchemy.exc import SQLAlchemyError

from jobfinder.jobsearch.match import rank_jobs
from jobfinder.jobsearch.models import ApplicationStatus, JobMatch, RawPosting, Tier
from jobfinder.jobsearch.normalize import canonicalize, job_key
from jobfinder.jobsearch.profile import LIVE_DIMENSIONS, VP_AI_PROFILE
from jobfinder.jobsearch.rerank import (
    DEFAULT_RERANK_TOP,
    GeminiReranker,
    rerank_matches,
)
from jobfinder.jobsearch.sources.eml_dir import read_eml_dir
from jobfinder.jobsearch.sources.gmail import (
    CLIENT_SECRET_FILENAME,
    DEFAULT_CRED_DIR,
    GmailSource,
)
from jobfinder.jobsearch.store import JobStore, StoredJob
from jobfinder.sources.ats import PROVIDERS, AtsClient, JobBoard

_TIER_ORDER = {Tier.A: 0, Tier.B: 1, Tier.C: 2}


def _store_url(db: str) -> str:
    """Map a --db value to a SQLAlchemy URL (mirrors jobfinder.cli._store_url).

    A bare path becomes a local SQLite file; anything already containing ``://``
    (a full driver URL) is passed through unchanged.
    """
    return db if "://" in db else f"sqlite+pysqlite:///{db}"


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
        # Layer-2 (LLM) contribution, when an opt-in --rerank pass annotated this
        # match. Surfaced (not folded into the score) so a human sees why the LLM
        # moved a job — the Layer-1 score above stays authoritative.
        if m.llm is not None:
            lines.append(
                f"   LLM re-rank: #{m.llm.rank} ({m.llm.relevance}) — {m.llm.rationale}"
            )
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

    # Optional Layer-2 LLM re-rank over the top-N (opt-in via --rerank). Built
    # lazily and only when a key exists; with no key the re-ranker is None and
    # rerank_matches returns the Layer-1 order unchanged — a run without the flag
    # (or without a key) is pure Layer 1.
    if args.rerank:
        reranker = GeminiReranker.from_env()
        if reranker is None:
            # No key → warn and stay on Layer-1 (don't re-resolve from_env inside
            # rerank_matches, which would just return None again).
            print(
                "rank: --rerank needs GEMINI_API_KEY set; "
                "falling back to deterministic Layer-1 ranking.",
                file=sys.stderr,
            )
        else:
            matches = rerank_matches(
                matches, VP_AI_PROFILE, reranker=reranker, top_n=args.rerank_top
            )

    # Optional persistence (opt-in via --db). A run without --db is unchanged:
    # print-and-forget, no DB touched. With --db, the ranked run is saved in one
    # transaction; a re-seen job updates in place (status/first_seen_at kept).
    # Hard-rejected matches (IC roles, internships) are NOT persisted — they are
    # noise the rank display already hides, so the CRM mirrors what the user sees
    # rather than filling with disqualified rows. A store/open failure is reported
    # cleanly (rc 2) instead of crashing AFTER the ranked output is lost.
    if args.db:
        keepers = [m for m in matches if not m.rejected]
        try:
            result = JobStore(_store_url(args.db)).save_matches(keepers, now=now)
        except SQLAlchemyError as exc:
            print(f"rank: could not save to {args.db}: {exc}", file=sys.stderr)
            return 2
        print(
            f"Saved to {args.db}: {result.inserted} new, {result.updated} updated.",
            file=sys.stderr,
        )

    print(render(matches, top=args.top, min_tier=Tier(args.min_tier)))
    return 0


def render_stored(jobs: list[StoredJob]) -> str:
    """Render persisted CRM jobs as a status-annotated, id-keyed list."""
    lines: list[str] = []
    header = f"{len(jobs)} job(s) in the CRM"
    lines.append(header)
    lines.append("=" * len(header))
    if not jobs:
        lines.append("")
        lines.append("No jobs match. (Persist a run with `rank --db <path>` first.)")
        return "\n".join(lines)
    for sj in jobs:
        job = sj.match.job
        lines.append("")
        lines.append(
            f"[{sj.status.value}] [{sj.match.tier.value}] {job.title}  —  "
            f"{job.company}  (score {sj.match.score:.0f}/100)"
        )
        lines.append(f"   id: {job_key(job)}")
        if job.location:
            lines.append(f"   Location: {job.location}")
        lines.append(
            f"   Apply: {job.best_apply_url or '(open LinkedIn listing manually)'}"
        )
    return "\n".join(lines)


def _open_store(db: str) -> JobStore:
    """Open (creating if absent) the CRM store, raising SQLAlchemyError on failure.

    Callers wrap this so a bad/unwritable/corrupt --db path yields a clean rc-2
    message rather than an uncaught traceback.
    """
    return JobStore(_store_url(db))


def _run_list(args: argparse.Namespace) -> int:
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"list: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    jobs = store.list_jobs(
        status=ApplicationStatus(args.status) if args.status else None,
        min_tier=Tier(args.min_tier) if args.min_tier else None,
        include_archived=args.all,
        limit=args.top,
    )
    print(render_stored(jobs))
    return 0


def _run_status(args: argparse.Namespace) -> int:
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"status: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    # An empty fragment would "match" every row via the prefix scan; require a
    # real fragment so `status --db x '' applied` can't silently mutate the sole
    # stored job (it would slip past the >1-match ambiguity guard).
    if not args.job_id:
        print(
            "status: provide a job id (or a leading fragment); see `list`.",
            file=sys.stderr,
        )
        return 2
    # Resolve an unambiguous leading fragment to the full id, so the user need not
    # paste the whole composite key.
    matches = store.find_ids(args.job_id)
    if not matches:
        print(f"status: no job id starting with {args.job_id!r}.", file=sys.stderr)
        return 2
    if len(matches) > 1:
        print(
            f"status: {args.job_id!r} is ambiguous ({len(matches)} jobs); "
            "use a longer id fragment:",
            file=sys.stderr,
        )
        for mid in matches:
            print(f"  {mid}", file=sys.stderr)
        return 2
    job_id = matches[0]
    # set_status returns False if the row vanished between resolution and write —
    # report that rather than printing a false success.
    if not store.set_status(job_id, ApplicationStatus(args.new_status)):
        print(f"status: job {job_id!r} no longer exists.", file=sys.stderr)
        return 2
    print(f"{job_id} → {args.new_status}")
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
        "--rerank",
        action="store_true",
        help="Opt-in Layer-2 Gemini relevance re-rank over the top-N candidates "
        "(needs GEMINI_API_KEY). Degrades to deterministic Layer-1 without a key "
        "or on any LLM error; the Layer-1 score stays authoritative.",
    )
    rank.add_argument(
        "--rerank-top",
        type=int,
        default=DEFAULT_RERANK_TOP,
        metavar="N",
        help=f"How many top Layer-1 candidates to send to the LLM re-ranker "
        f"(cost control; default {DEFAULT_RERANK_TOP}). Only used with --rerank.",
    )
    rank.add_argument(
        "--min-tier",
        choices=[t.value for t in Tier],
        default=Tier.B.value,
        help="Lowest tier to display (A best). Default B.",
    )
    rank.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Persist the ranked run to this local CRM database (a SQLite file "
        "path, or a full driver URL). A re-seen job updates in place, keeping the "
        "application status you set. Omit for the offline print-and-forget run.",
    )

    list_p = sub.add_parser(
        "list",
        help="List jobs saved in the CRM database, highest score first.",
    )
    list_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database to read."
    )
    list_p.add_argument(
        "--status",
        choices=[s.value for s in ApplicationStatus],
        default=None,
        help="Show only jobs in this pipeline status.",
    )
    list_p.add_argument(
        "--min-tier",
        choices=[t.value for t in Tier],
        default=None,
        help="Show only jobs at or above this tier (A best).",
    )
    list_p.add_argument(
        "--all",
        action="store_true",
        help="Include ARCHIVED jobs (hidden by default unless --status archived).",
    )
    list_p.add_argument(
        "-n", "--top", type=int, default=None, help="Cap how many jobs to show."
    )

    status_p = sub.add_parser(
        "status",
        help="Set a saved job's application-pipeline status (free transitions).",
    )
    status_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database to update."
    )
    status_p.add_argument(
        "job_id",
        metavar="JOB_ID",
        help="The job id (or an unambiguous leading fragment) — see `list`.",
    )
    status_p.add_argument(
        "new_status",
        metavar="STATUS",
        choices=[s.value for s in ApplicationStatus],
        help="The new status: " + ", ".join(s.value for s in ApplicationStatus) + ".",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rank":
        return _run_rank(args)
    if args.command == "list":
        return _run_list(args)
    if args.command == "status":
        return _run_status(args)
    return 2  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
