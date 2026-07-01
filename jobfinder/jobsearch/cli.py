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

from jobfinder.jobsearch.contacts import render_checklist, target_contacts
from jobfinder.jobsearch.email_format import (
    guess_emails,
    is_personal_domain,
    is_valid_business_email,
    normalize_domain,
)
from jobfinder.jobsearch.match import rank_jobs
from jobfinder.jobsearch.models import (
    ApplicationStatus,
    Contact,
    ContactRole,
    ContactSource,
    DraftStatus,
    EmailGuess,
    JobMatch,
    RawPosting,
    Tier,
)
from jobfinder.jobsearch.normalize import canonicalize, job_key
from jobfinder.jobsearch.outreach import (
    GeminiTailorer,
    SenderIdentity,
    assemble_email,
    render_email_text,
)
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
from jobfinder.jobsearch.sources.gmail_send import (
    SEND_TOKEN_FILENAME,
    GmailSendError,
    GmailSender,
)
from jobfinder.jobsearch.store import JobStore, StoredContact, StoredDraft, StoredJob
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


def _resolve_id(
    fragment: str,
    finder,
    *,
    cmd: str,
    noun: str,
    see: str,
) -> str | None:
    """Resolve an unambiguous leading id fragment to a full id, or None.

    Prints a clear error to stderr (and returns None) when the fragment is empty,
    matches nothing, or is ambiguous — so the caller just returns rc 2. ``finder``
    is the store lookup (``find_ids`` for jobs, ``find_draft_ids`` for drafts);
    ``noun`` and ``see`` tailor the messages. Sharing this keeps the resolution
    rules (incl. the empty-fragment guard) identical across every subcommand.
    """
    # An empty fragment would "match" every row via the prefix scan; require a real
    # fragment so e.g. `status --db x '' applied` can't silently hit the sole row.
    if not fragment:
        print(
            f"{cmd}: provide a {noun} id (or a leading fragment); see `{see}`.",
            file=sys.stderr,
        )
        return None
    matches = finder(fragment)
    if not matches:
        print(f"{cmd}: no {noun} id starting with {fragment!r}.", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(
            f"{cmd}: {fragment!r} is ambiguous ({len(matches)} {noun}s); "
            "use a longer id fragment:",
            file=sys.stderr,
        )
        for mid in matches:
            print(f"  {mid}", file=sys.stderr)
        return None
    return matches[0]


def _resolve_job_id(store: JobStore, fragment: str, *, cmd: str) -> str | None:
    """Resolve a leading job-id fragment to a full job_key (see :func:`_resolve_id`).

    Shared by every subcommand that takes a JOB_ID (status/contacts/add-contact/
    outreach draft). (`email` takes name+domain, not a job id, so it does not use
    this.)
    """
    return _resolve_id(fragment, store.find_ids, cmd=cmd, noun="job", see="list")


def _run_status(args: argparse.Namespace) -> int:
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"status: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    job_id = _resolve_job_id(store, args.job_id, cmd="status")
    if job_id is None:
        return 2
    # set_status returns False if the row vanished between resolution and write —
    # report that rather than printing a false success.
    if not store.set_status(job_id, ApplicationStatus(args.new_status)):
        print(f"status: job {job_id!r} no longer exists.", file=sys.stderr)
        return 2
    print(f"{job_id} → {args.new_status}")
    return 0


def render_contacts(
    stored: list[StoredContact],
    *,
    store: JobStore,
    provider=None,
    top_guesses: int = 3,
) -> str:
    """Render recorded contacts with confidence-ranked business-email guesses.

    Every email guess is checked against the always-honored do-not-contact list:
    a suppressed address is never shown, and a contact whose every guess is
    suppressed (or whose domain is on the list) is marked accordingly. Personal
    email domains never produce a guess (business emails only).
    """
    lines: list[str] = []
    header = f"{len(stored)} contact(s)"
    lines.append(header)
    lines.append("=" * len(header))
    if not stored:
        lines.append("")
        lines.append(
            "No contacts recorded. Use `contacts` for a checklist, then "
            "`add-contact` to record who you find."
        )
        return "\n".join(lines)
    for sc in stored:
        c = sc.contact
        lines.append("")
        title = f" — {c.title}" if c.title else ""
        lines.append(f"[{c.role.value}] {c.name}{title}  ({c.company})")
        if c.linkedin_url:
            lines.append(f"   LinkedIn: {c.linkedin_url}")
        if c.notes:
            lines.append(f"   Notes: {c.notes}")
        if not c.email_domain:
            lines.append("   Email: (no company domain recorded — pass --domain)")
            continue
        if is_personal_domain(c.email_domain):
            lines.append(
                f"   Email: ({c.email_domain} is a personal domain — "
                "business emails only, no guess)"
            )
            continue
        # Suppression is applied INSIDE guess_emails (the single producer), so a
        # do-not-contact address is never constructed. We still detect the
        # "everything suppressed" case by comparing against an unsuppressed run so
        # the user is told why no guess appears rather than seeing a bare blank.
        guesses = guess_emails(
            c.name, c.email_domain, provider=provider, is_suppressed=store.is_suppressed
        )
        if guesses:
            lines.append("   Email guesses (business; confidence):")
            for g in guesses[:top_guesses]:
                lines.append(f"     {_format_guess(g)}")
        elif guess_emails(c.name, c.email_domain, provider=provider):
            lines.append(
                "   Email: (all guesses on the do-not-contact list — suppressed)"
            )
        else:
            lines.append("   Email: (could not construct a business email)")
    return "\n".join(lines)


def _run_contacts(args: argparse.Namespace) -> int:
    """Show the manual contact checklist for a job, or recorded contacts (--show)."""
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"contacts: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    job_id = _resolve_job_id(store, args.job_id, cmd="contacts")
    if job_id is None:
        return 2
    stored_job = store.get(job_id)
    if stored_job is None:
        print(f"contacts: job {job_id!r} no longer exists.", file=sys.stderr)
        return 2
    if args.show:
        print(render_contacts(store.list_contacts(job_id), store=store))
        return 0
    job = stored_job.match.job
    print(render_checklist(job, target_contacts(job)))
    return 0


def _run_add_contact(args: argparse.Namespace) -> int:
    """Record (paste back) a contact found via manual LinkedIn review."""
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"add-contact: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    job_id = _resolve_job_id(store, args.job_id, cmd="add-contact")
    if job_id is None:
        return 2
    stored_job = store.get(job_id)
    if stored_job is None:
        print(f"add-contact: job {job_id!r} no longer exists.", file=sys.stderr)
        return 2
    if not args.name.strip():
        print("add-contact: a contact name is required.", file=sys.stderr)
        return 2
    # Normalize the domain to its bare host up front: reject a personal domain,
    # reject a malformed one (rather than silently storing junk that yields no
    # guesses later), and store the canonical form so the contact_key matches how
    # guess_emails/is_suppressed normalize it (no duplicate rows for acme.com vs
    # https://acme.com/careers).
    email_domain: str | None = None
    if args.domain:
        if is_personal_domain(args.domain):
            print(
                f"add-contact: {args.domain!r} is a personal email domain; "
                "business emails only.",
                file=sys.stderr,
            )
            return 2
        email_domain = normalize_domain(args.domain)
        if email_domain is None:
            print(
                f"add-contact: {args.domain!r} is not a valid business domain.",
                file=sys.stderr,
            )
            return 2
    contact = Contact(
        name=args.name.strip(),
        company=stored_job.match.job.company,
        role=ContactRole(args.role),
        title=args.title,
        linkedin_url=args.linkedin_url,
        email_domain=email_domain,
        source=ContactSource.MANUAL,
        notes=args.notes,
    )
    inserted = store.save_contact(job_id, contact)
    verb = "Added" if inserted else "Updated"
    print(f"{verb} contact {contact.name!r} on {job_id}.")
    return 0


def _run_email(args: argparse.Namespace) -> int:
    """Construct confidence-ranked business-email guesses for a name + domain.

    Always consults the do-not-contact list (``--db`` is required) so the
    "always honored" guarantee can't be bypassed by omitting a flag. Personal
    domains are refused; suppressed addresses are never constructed.
    """
    if is_personal_domain(args.domain):
        print(
            f"email: {args.domain!r} is a personal email domain; business emails only.",
            file=sys.stderr,
        )
        return 2
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"email: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    guesses = guess_emails(args.name, args.domain, is_suppressed=store.is_suppressed)
    if guesses:
        print(_render_guesses(args.name, guesses))
        return 0
    # Nothing to show: distinguish "all suppressed" from "couldn't construct".
    if guess_emails(args.name, args.domain):
        print("email: all guesses are on the do-not-contact list — suppressed.")
        return 0
    print(
        f"email: could not construct a business email for {args.name!r} "
        f"@ {args.domain!r} (need a full name and a valid business domain).",
        file=sys.stderr,
    )
    return 2


def _format_guess(g: EmailGuess) -> str:
    """One-line rendering of an email guess, shared by `email` and `contacts --show`."""
    return f"{g.email}  [{g.pattern}, {g.confidence:.0%}, {g.provenance}]"


def _render_guesses(name: str, guesses: list[EmailGuess]) -> str:
    lines = [f"Business-email guesses for {name} (confidence):"]
    lines.extend(f"  {_format_guess(g)}" for g in guesses)
    return "\n".join(lines)


def _run_dnc(args: argparse.Namespace) -> int:
    """Manage the always-honored do-not-contact list (add / list)."""
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"dnc: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    if args.value:
        entry = store.add_do_not_contact(args.value, reason=args.reason)
        if entry is None:
            print(
                f"dnc: {args.value!r} is not a valid email or domain.",
                file=sys.stderr,
            )
            return 2
        print(f"Do-not-contact: added {entry.kind} {entry.value!r}.")
        return 0
    entries = store.list_do_not_contact()
    if not entries:
        print("Do-not-contact list is empty.")
        return 0
    print(
        f"Do-not-contact list ({len(entries)} entr{'y' if len(entries) == 1 else 'ies'}):"
    )
    for e in entries:
        reason = f"  — {e.reason}" if e.reason else ""
        print(f"  [{e.kind}] {e.value}{reason}")
    return 0


def _resolve_draft_id(store: JobStore, fragment: str, *, cmd: str) -> str | None:
    """Resolve a leading draft-id fragment to a full id (see :func:`_resolve_id`)."""
    return _resolve_id(
        fragment,
        store.find_draft_ids,
        cmd=cmd,
        noun="draft",
        see="outreach list-drafts",
    )


def _resolve_recipient(args, store: JobStore, *, cmd: str) -> str | None:
    """Resolve the outreach recipient: an explicit --to-email or a top email guess.

    Both paths enforce the same two structural guards: the address must be a
    BUSINESS domain (personal domains refused) and must NOT be on the
    do-not-contact list. An explicit ``--to-email`` is validated directly; without
    one, the top confidence-ranked business guess from ``guess_emails`` (which
    already applies the do-not-contact filter at the producer) is used. Returns
    ``None`` (after printing why) when no honest, permitted address is available.
    """
    if args.to_email:
        addr = args.to_email.strip()
        # Fail closed on malformed/personal addresses: is_valid_business_email
        # rejects the "@acme.com"/"a@@acme.com" shapes a domain-only check would
        # pass, and any personal/free-mail domain (business emails only).
        if not is_valid_business_email(addr):
            print(
                f"{cmd}: {addr!r} is not a valid business email (business domains "
                "only; personal domains refused).",
                file=sys.stderr,
            )
            return None
        if store.is_suppressed(addr):
            print(
                f"{cmd}: {addr!r} is on the do-not-contact list — refusing.",
                file=sys.stderr,
            )
            return None
        return addr
    if not args.domain:
        print(
            f"{cmd}: provide a recipient with --to-email, or --domain to guess one.",
            file=sys.stderr,
        )
        return None
    if is_personal_domain(args.domain):
        print(
            f"{cmd}: {args.domain!r} is a personal email domain; business emails only.",
            file=sys.stderr,
        )
        return None
    guesses = guess_emails(args.name, args.domain, is_suppressed=store.is_suppressed)
    if guesses:
        return guesses[0].email
    # Nothing usable: distinguish "all suppressed" from "couldn't construct".
    if guess_emails(args.name, args.domain):
        print(
            f"{cmd}: every business-email guess for {args.name!r} @ {args.domain!r} "
            "is on the do-not-contact list — refusing.",
            file=sys.stderr,
        )
    else:
        print(
            f"{cmd}: could not construct a business email for {args.name!r} @ "
            f"{args.domain!r} (need a full name and a valid business domain), and "
            "no --to-email was given.",
            file=sys.stderr,
        )
    return None


def render_draft(sd: StoredDraft, *, full: bool = False) -> str:
    """Render a stored draft for review. With ``full``, include the body text."""
    e = sd.email
    lines = [
        f"Draft {sd.id}  [{sd.status.value}]",
        f"   To:      {e.to_name} <{e.to_email}>",
        f"   From:    {e.from_name} <{e.from_email}>",
        f"   Subject: {e.subject}",
        f"   Role:    {e.job_title} @ {e.company}",
        f"   Tailoring: {e.tailoring}",
    ]
    if sd.sent_at is not None:
        lines.append(f"   Sent at: {sd.sent_at.isoformat()}")
    if full:
        lines.append("   ----- email text -----")
        for line in render_email_text(e).splitlines():
            lines.append(f"   {line}")
        lines.append("   ----------------------")
    return "\n".join(lines)


def _run_outreach_draft(args: argparse.Namespace) -> int:
    """Assemble a tailored outreach draft for a job + recipient. Sends NOTHING.

    The draft is stored DRAFTED; reviewing it and sending are separate steps. The
    recipient is always a permitted business address; the body is the deterministic
    template, optionally sharpened by the injected LLM seam (--tailor).
    """
    if not args.from_name.strip() or not args.from_email.strip():
        print(
            "outreach draft: --from-name and --from-email are required (a truthful "
            "sender identity).",
            file=sys.stderr,
        )
        return 2
    if is_personal_domain(args.from_email):
        # The sender's own address may legitimately be anything, but a personal
        # FROM undercuts the business-outreach posture; keep it consistent.
        print(
            "outreach draft: --from-email should be a business address.",
            file=sys.stderr,
        )
        return 2
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"outreach draft: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    job_id = _resolve_job_id(store, args.job_id, cmd="outreach draft")
    if job_id is None:
        return 2
    stored_job = store.get(job_id)
    if stored_job is None:
        print(f"outreach draft: job {job_id!r} no longer exists.", file=sys.stderr)
        return 2
    if not args.name.strip():
        print("outreach draft: a contact name is required.", file=sys.stderr)
        return 2
    to_email = _resolve_recipient(args, store, cmd="outreach draft")
    if to_email is None:
        return 2

    job = stored_job.match.job
    contact = Contact(
        name=args.name.strip(),
        company=job.company,
        role=ContactRole(args.role),
        email_domain=normalize_domain(args.domain) if args.domain else None,
        source=ContactSource.MANUAL,
    )
    # Tailoring seam: built lazily and only when --tailor is set AND a key exists.
    # With no key the tailorer is None and assemble_email uses the deterministic
    # template — a run without --tailor (or without a key) is pure template.
    tailorer = None
    if args.tailor:
        tailorer = GeminiTailorer.from_env(match_reason=stored_job.match.reason)
        if tailorer is None:
            print(
                "outreach draft: --tailor needs GEMINI_API_KEY set; "
                "using the deterministic template body.",
                file=sys.stderr,
            )
    email = assemble_email(
        job,
        contact,
        to_email,
        sender=SenderIdentity(
            name=args.from_name.strip(), email=args.from_email.strip()
        ),
        match_reason=stored_job.match.reason,
        tailorer=tailorer,
    )
    if email is None:
        print(
            f"outreach draft: could not assemble a draft for {to_email!r} "
            "(unusable or personal recipient domain).",
            file=sys.stderr,
        )
        return 2
    sd = store.save_draft(job_id, email)
    # save_draft refuses to overwrite an already-SENT draft (so the "already
    # contacted" record and the re-send guard survive). Detect that and warn
    # rather than falsely claiming a fresh draft was written.
    if sd.status is DraftStatus.SENT:
        print(
            f"outreach draft: {to_email} was already emailed for this job "
            f"({sd.sent_at.isoformat() if sd.sent_at else 'unknown time'}); "
            "keeping the sent record (nothing re-drafted).",
            file=sys.stderr,
        )
        return 2
    print(f"Drafted (nothing sent). Review, then `outreach send {sd.id} --confirm`.")
    print(render_draft(sd, full=True))
    return 0


def _run_outreach_list(args: argparse.Namespace) -> int:
    """List assembled outreach drafts (newest first), optionally by status."""
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"outreach list-drafts: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    status = DraftStatus(args.status) if args.status else None
    drafts = store.list_drafts(status=status)
    header = f"{len(drafts)} outreach draft(s)"
    print(header)
    print("=" * len(header))
    if not drafts:
        print("\nNo drafts. Assemble one with `outreach draft <job> <name> ...`.")
        return 0
    for sd in drafts:
        print("")
        print(render_draft(sd))
    return 0


def _run_outreach_send(args: argparse.Namespace) -> int:
    """The draft-and-approve SEND gate — the only path that can put mail on the wire.

    Without ``--confirm`` this is a DRY RUN: it prints exactly what WOULD be sent
    and sends nothing. ``--confirm`` is the explicit, separate, deliberate approval
    that actually sends via the gmail.send seam. Even with ``--confirm`` the
    do-not-contact list and business-only rule are RE-CHECKED here (defence in
    depth — never trust that the draft layer already filtered, since a draft can be
    stale relative to a later `dnc` entry), and an already-sent draft is not
    re-sent.
    """
    try:
        store = _open_store(args.db)
    except SQLAlchemyError as exc:
        print(f"outreach send: could not open {args.db}: {exc}", file=sys.stderr)
        return 2
    draft_id = _resolve_draft_id(store, args.draft_id, cmd="outreach send")
    if draft_id is None:
        return 2
    sd = store.get_draft(draft_id)
    if sd is None:
        print(f"outreach send: draft {draft_id!r} no longer exists.", file=sys.stderr)
        return 2
    if sd.status is DraftStatus.SENT:
        print(
            f"outreach send: draft {sd.id} was already sent "
            f"({sd.sent_at.isoformat() if sd.sent_at else 'unknown time'}); "
            "refusing to re-send.",
            file=sys.stderr,
        )
        return 2

    # Defence in depth: re-check the recipient at send time, NOT trusting that the
    # draft was filtered when assembled. A `dnc` entry (or a bad address) added
    # after the draft was written must still block the SEND. Computed for both
    # paths: the dry run surfaces it (so a stale draft can still be reviewed and
    # the reason seen), and the real send refuses on it.
    to_email = sd.email.to_email
    blocked: str | None = None
    if is_personal_domain(to_email):
        blocked = f"{to_email!r} is a personal domain; business emails only"
    elif not is_valid_business_email(to_email):
        blocked = f"{to_email!r} is not a valid business email"
    elif store.is_suppressed(to_email):
        blocked = f"{to_email!r} is on the do-not-contact list"

    if not args.confirm:
        # DRY RUN — the safe default. Show what WOULD be sent; send nothing. A
        # blocked recipient is still shown (for review) with why it can't be sent.
        print("DRY RUN — nothing sent. Re-run with --confirm to actually send.")
        if blocked is not None:
            print(f"NOTE: this draft cannot be sent — {blocked}.")
        print(render_draft(sd, full=True))
        return 0

    if blocked is not None:
        print(
            f"outreach send: {blocked} — refusing to send (even though a draft "
            "exists for it).",
            file=sys.stderr,
        )
        return 2

    # Explicit approval given: send via the injected gmail.send seam.
    try:
        sender = GmailSender.from_env()
    except GmailSendError as exc:
        print(f"outreach send: {exc}", file=sys.stderr)
        return 2
    if sender is None:
        print(
            "outreach send: --confirm needs Gmail send credentials on disk: place "
            f"{CLIENT_SECRET_FILENAME} under {DEFAULT_CRED_DIR} and authorize the "
            f"send scope once (writes {SEND_TOKEN_FILENAME}). See "
            "jobfinder/jobsearch/sources/gmail_send.py.",
            file=sys.stderr,
        )
        return 2
    e = sd.email
    try:
        message_id = sender.send_email(
            to_email=e.to_email,
            to_name=e.to_name,
            from_email=e.from_email,
            from_name=e.from_name,
            subject=e.subject,
            body=render_email_text(e),
        )
    except GmailSendError as exc:
        print(f"outreach send: {exc}", file=sys.stderr)
        return 2
    suffix = f" (message id {message_id})" if message_id else ""
    # The email is out. Record it as SENT so the re-send guard engages next time.
    # If the row vanished between get_draft and here, the send still happened —
    # say so loudly rather than print a clean success that hides the drift (a
    # DRAFTED-looking row would otherwise invite a duplicate send).
    if not store.mark_sent(sd.id):
        print(
            f"outreach send: WARNING — sent to {e.to_email}{suffix}, but draft "
            f"{sd.id} could not be marked sent (it may have been removed). Do NOT "
            "re-send: the email was already delivered.",
            file=sys.stderr,
        )
        return 1
    print(f"Sent draft {sd.id} to {e.to_email}{suffix}.")
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

    contacts_p = sub.add_parser(
        "contacts",
        help="Show a manual LinkedIn contact checklist for a saved job (or the "
        "contacts you have recorded, with --show).",
    )
    contacts_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database to read."
    )
    contacts_p.add_argument(
        "job_id",
        metavar="JOB_ID",
        help="The job id (or an unambiguous leading fragment) — see `list`.",
    )
    contacts_p.add_argument(
        "--show",
        action="store_true",
        help="Show contacts already recorded for this job (with business-email "
        "guesses) instead of the manual search checklist.",
    )

    add_p = sub.add_parser(
        "add-contact",
        help="Record a contact found via manual LinkedIn review (paste-back). "
        "No scraping: you do the LinkedIn lookup by hand and record the name here.",
    )
    add_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database to update."
    )
    add_p.add_argument(
        "job_id",
        metavar="JOB_ID",
        help="The job this contact is for (id or unambiguous fragment) — see `list`.",
    )
    add_p.add_argument("name", metavar="NAME", help="The person's full name.")
    add_p.add_argument(
        "--role",
        choices=[r.value for r in ContactRole],
        default=ContactRole.OTHER.value,
        help="How they relate to the role (default other).",
    )
    add_p.add_argument("--title", default=None, help="Their job title, if known.")
    add_p.add_argument(
        "--linkedin-url", default=None, help="Their LinkedIn profile URL, if known."
    )
    add_p.add_argument(
        "--domain",
        default=None,
        metavar="DOMAIN",
        help="The company's BUSINESS email domain (e.g. 'acme.com') for email "
        "guesses. Personal domains are rejected.",
    )
    add_p.add_argument("--notes", default=None, help="Free-text notes.")

    email_p = sub.add_parser(
        "email",
        help="Construct confidence-ranked BUSINESS email guesses for a name at a "
        "domain. Personal domains rejected; the do-not-contact list (in --db) is "
        "always honored.",
    )
    email_p.add_argument("name", metavar="NAME", help="The person's full name.")
    email_p.add_argument(
        "domain", metavar="DOMAIN", help="The company's business email domain."
    )
    email_p.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="The CRM database whose do-not-contact list is honored (always "
        "consulted so a suppressed address is never constructed).",
    )

    dnc_p = sub.add_parser(
        "dnc",
        help="Manage the do-not-contact list (always honored). With a VALUE, add "
        "an email or domain; with none, list the current entries.",
    )
    dnc_p.add_argument("--db", required=True, metavar="PATH", help="The CRM database.")
    dnc_p.add_argument(
        "value",
        metavar="EMAIL_OR_DOMAIN",
        nargs="?",
        default=None,
        help="An email or bare domain to suppress. Omit to list the current entries.",
    )
    dnc_p.add_argument(
        "--reason", default=None, help="Why this entry is suppressed (optional)."
    )

    # ----- outreach (Slice F): draft-and-approve, human-in-the-loop ----- #
    outreach_p = sub.add_parser(
        "outreach",
        help="Draft-and-approve outreach: assemble a tailored email for a contact, "
        "review it, and send it only via an explicit, separate confirm step. Never "
        "sends autonomously.",
    )
    outreach_sub = outreach_p.add_subparsers(dest="outreach_command", required=True)

    draft_p = outreach_sub.add_parser(
        "draft",
        help="Assemble a tailored outreach draft for a saved job + a contact. "
        "Stores the draft and sends NOTHING.",
    )
    draft_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database (jobs + DNC)."
    )
    draft_p.add_argument(
        "job_id",
        metavar="JOB_ID",
        help="The job this outreach is about (id or unambiguous fragment) — see `list`.",
    )
    draft_p.add_argument("name", metavar="NAME", help="The recipient's full name.")
    draft_p.add_argument(
        "--to-email",
        default=None,
        metavar="EMAIL",
        help="The exact business address to write to. If omitted, the top "
        "confidence-ranked business guess for NAME at --domain is used (the "
        "do-not-contact list is always honored). Personal domains are refused.",
    )
    draft_p.add_argument(
        "--domain",
        default=None,
        metavar="DOMAIN",
        help="The company's BUSINESS email domain, used to guess --to-email when "
        "it isn't given (e.g. 'acme.com'). Personal domains are rejected.",
    )
    draft_p.add_argument(
        "--role",
        choices=[r.value for r in ContactRole],
        default=ContactRole.OTHER.value,
        help="How the recipient relates to the role (tailors the body; default other).",
    )
    draft_p.add_argument(
        "--from-name",
        required=True,
        metavar="NAME",
        help="YOUR real name (a truthful CAN-SPAM 'From').",
    )
    draft_p.add_argument(
        "--from-email",
        required=True,
        metavar="EMAIL",
        help="YOUR real reply-to address (where an opt-out reply lands).",
    )
    draft_p.add_argument(
        "--tailor",
        action="store_true",
        help="Opt-in: sharpen the body with the Gemini tailoring seam (needs "
        "GEMINI_API_KEY). Degrades to the deterministic template without a key or "
        "on any LLM error; the subject, identity and opt-out stay deterministic.",
    )

    list_drafts_p = outreach_sub.add_parser(
        "list-drafts", help="List assembled outreach drafts, newest first."
    )
    list_drafts_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database to read."
    )
    list_drafts_p.add_argument(
        "--status",
        choices=[s.value for s in DraftStatus],
        default=None,
        help="Show only drafts in this state (drafted | sent).",
    )

    send_p = outreach_sub.add_parser(
        "send",
        help="Review a draft (dry run) or, with --confirm, actually send it. This "
        "is the ONLY path that puts mail on the wire; the do-not-contact list is "
        "re-checked at send time.",
    )
    send_p.add_argument(
        "--db", required=True, metavar="PATH", help="The CRM database (drafts + DNC)."
    )
    send_p.add_argument(
        "draft_id",
        metavar="DRAFT_ID",
        help="The draft to send (id or unambiguous fragment) — see `outreach "
        "list-drafts`.",
    )
    send_p.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly approve and SEND this draft via Gmail (gmail.send scope). "
        "Without it, this is a DRY RUN that prints the email and sends nothing.",
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
    if args.command == "contacts":
        return _run_contacts(args)
    if args.command == "add-contact":
        return _run_add_contact(args)
    if args.command == "email":
        return _run_email(args)
    if args.command == "dnc":
        return _run_dnc(args)
    if args.command == "outreach":
        if args.outreach_command == "draft":
            return _run_outreach_draft(args)
        if args.outreach_command == "list-drafts":
            return _run_outreach_list(args)
        if args.outreach_command == "send":
            return _run_outreach_send(args)
    return 2  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
