"""Hiring-pattern signals from public ATS job boards (Pillar I).

A company's open-roles board is a leading indicator of where it is investing.
Three patterns matter for spotting a *forming senior role* before it is posted:

    ats_hiring_velocity  A burst of recently-opened reqs -> the org is scaling
                         and will need (or is about to need) leadership over the
                         new headcount.
    department_surge     One department concentrates that hiring -> that function
                         is being built out, a classic precursor to a VP/Head hire.
    greenfield_team      A posting that is explicitly a *new* team's first hire
                         (founding/inaugural/"first X hire"), or a lone leadership
                         req with no team under it yet -> a function being stood
                         up from scratch.

These map onto the scorer's two ATS-fed components (``jobfinder.scoring``):
``ats_hiring_velocity`` + ``department_surge`` drive ``hiring_velocity``;
``greenfield_team`` drives ``strategic_language``. Both have scored 0 since
Slice 2 — this module is what lights them up.

Like ``form_d``, the input is already-fetched structured data (a ``JobBoard``
of normalized ``JobPosting`` records, see ``jobfinder.sources.ats``), so this
stage is purely deterministic — no LLM and no network of its own, fully
offline-testable. Every emitted ``Signal`` cites the board (and the specific
posting, for greenfield) as ``Evidence``.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone

from jobfinder.schemas import Evidence, Signal
from jobfinder.sources.ats import JobBoard, JobPosting

# A posting "updated" within this window counts toward current hiring velocity.
# Senior-role formation is a weeks-to-a-couple-months horizon, so a 30-day
# window captures an active burst without dragging in stale evergreen reqs.
RECENT_WINDOW_DAYS = 30.0

# A posting dated slightly in the future is plausible clock skew between us and
# the provider; one dated *far* in the future is bad data (a misparsed epoch, a
# data-entry typo) and must NOT be counted as recent — otherwise it would
# silently inflate the velocity/surge counts.
MAX_FUTURE_SKEW_DAYS = 2.0

# Velocity thresholds (count of recent openings). Below MIN we emit nothing (a
# couple of open reqs is just normal attrition backfill); at/above STRONG the
# velocity component saturates.
MIN_RECENT_OPENINGS = 3
STRONG_RECENT_OPENINGS = 15

# A single department with at least this many recent openings is a surge.
DEPT_SURGE_MIN = 3

# Leadership titles. A lone leadership req in a department with no team beneath
# it reads as a new function being stood up (greenfield).
_LEADERSHIP_RE = re.compile(
    r"\b(chief\s+\w+\s+officer|c[eftori]o|head\s+of|vp\b|vice\s+president|"
    r"svp\b|director\s+of|head\s*,)",  # "Head, Engineering" — comma may precede a space
    re.IGNORECASE,
)

# Support / operations / enablement functions. A lone "Head of <support function>"
# on a board is routine org structure (every company has one), NOT a zero-to-one
# team being stood up — so the lone-leadership greenfield heuristic must NOT fire
# on it. This denylist gates ONLY that heuristic; the explicit-language branch
# (_GREENFIELD_RE) is unaffected, and a genuine product/eng/GTM/data function
# (e.g. "Head of Data") is permissive — it is absent here, so it still fires.
# Live finding D: "Head of Learning & Quality, Stripe Delivery Center" wrongly
# read as greenfield because "Head of" + a lone support function looked like a
# forming seat.
#
# Matched against the TITLE only (the function being led), NOT the department:
# a support word in the dept label would suppress a clean core title (dept
# "Workplace" must not veto "Head of Engineering"), and the live cases all carry
# the function in the title. Tokens are deliberately SPECIFIC, not bare words, to
# avoid swallowing genuine technical/GTM seats: "learning" only in an L&D phrase
# ("Learning & …"/"L&D"), never "Machine Learning"; "service/help/IT/technical
# desk/support" forms rather than a bare "support" that would kill "Head of
# Developer Support"; and no bare "quality" (it lives in "Data Quality"/"Quality
# Engineering"). NOT routed through scoring.match_persona on purpose — greenfield
# classification is a signal-layer concern, and that classifier maps to target
# personas, not to "is this a routine back-office function".
_SUPPORT_FUNCTION_RE = re.compile(
    r"\b("
    r"learning\s*(?:&|and)\s*\w+|l\s*&\s*d|"  # L&D / "Learning & Quality" (not "Machine Learning")
    r"training|enablement|"
    r"helpdesk|help\s+desk|service\s+desk|it\s+support|technical\s+support|"
    r"facilities|workplace|real\s+estate|office\s+management|"
    r"it\s+operations|information\s+technology|"
    r"compliance|payroll|procurement"
    r")\b",
    re.IGNORECASE,
)

# Explicit "new team / first hire" language in a posting title. The "first hire"
# branch allows zero filler words ("First Hire"), an optional qualifier
# ("first product hire"), and plural ("first hires").
_GREENFIELD_RE = re.compile(
    r"\b(founding|inaugural|first\s+(?:\w+\s+){0,2}hires?\b|"
    r"zero[\s-]?to[\s-]?one|0\s*(?:-+|to|→)\s*1|"
    r"build(?:ing)?\s+(?:from\s+)?(?:the\s+)?ground\s+up|"
    r"stand(?:ing)?\s+up\s+(?:a\s+)?(?:new\s+)?team|net[\s-]?new\s+team)\b",
    re.IGNORECASE,
)


def _utcnow() -> datetime:
    # Wrapped so callers/tests can monkeypatch if they need determinism.
    return datetime.now(timezone.utc)


def is_recent(posting: JobPosting, now: datetime) -> bool:
    """True if the posting was updated within the recent window.

    Postings with no timestamp are conservatively treated as NOT recent — we do
    not invent freshness the provider did not give us.

    Public because the listed-roles corroboration (``jobfinder.listings``) reuses
    the same recency definition (window + future-skew clamp) when counting how
    many live reqs are *recent*, so the two never disagree on what "recent" means.
    """
    if posting.updated_at is None:
        return False
    age_days = (now - posting.updated_at).total_seconds() / 86_400.0
    # Within the window, but reject timestamps far in the future (bad data): a
    # small negative age is plausible clock skew, a large one is a misparse.
    return -MAX_FUTURE_SKEW_DAYS <= age_days <= RECENT_WINDOW_DAYS


def _ramp(count: int, low: int, high: int) -> float:
    """Linear ramp from 0.3 at `low` to 1.0 at `high`, clamped."""
    if count < low:
        return 0.0
    if count >= high:
        return 1.0
    return round(0.3 + 0.7 * ((count - low) / (high - low)), 3)


def _id_part(text: str) -> str:
    """A readable, collision-resistant id fragment for `text`.

    Signal ids are the store's upsert key, so two *distinct* departments (or
    postings) must never produce the same fragment — otherwise one row silently
    overwrites the other. A bare slug is not enough: 'Sales/Ops' and 'Sales-Ops'
    both slugify to 'sales-ops', and any all-non-ASCII name collapses to the
    empty string. So we keep the human-readable slug for legibility but append a
    short deterministic hash of the *raw* text, which is stable across runs
    (the property the idempotent upsert relies on) and unique per input.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
    return f"{slug}-{digest}"


def _freshest(postings: list[JobPosting]) -> datetime | None:
    """Most recent posting timestamp among `postings`, or None."""
    stamps = [p.updated_at for p in postings if p.updated_at is not None]
    return max(stamps) if stamps else None


def _evidence(
    board: JobBoard, observed: datetime, *, url: str | None = None, excerpt: str = ""
) -> Evidence:
    return Evidence(
        source=board.provider,
        url=url or board.url,
        locator=f"{board.provider}:{board.token}",
        excerpt=excerpt[:300] or None,
        retrieved_at=observed,
    )


def _velocity_signal(
    board: JobBoard,
    recent: list[JobPosting],
    *,
    company_id: str,
    observed: datetime,
) -> Signal | None:
    """Whole-board hiring-velocity signal from the count of recent openings."""
    if len(recent) < MIN_RECENT_OPENINGS:
        return None
    strength = _ramp(len(recent), MIN_RECENT_OPENINGS, STRONG_RECENT_OPENINGS)
    effective = _freshest(recent)
    summary = (
        f"{len(recent)} roles opened in the last {int(RECENT_WINDOW_DAYS)} days on "
        f"the {board.provider} board — active hiring velocity, the kind of "
        "scaling that typically pulls in senior leadership to manage the new "
        "headcount."
    )
    return Signal(
        id=f"ats:{board.provider}:{board.token}:velocity",
        company_id=company_id,
        signal_type="ats_hiring_velocity",
        source=board.provider,
        observed_at=observed,
        effective_at=effective,
        title=f"Hiring velocity: {len(recent)} recent openings ({board.provider})",
        summary=summary,
        extracted_facts={
            "provider": board.provider,
            "token": board.token,
            "recent_openings": len(recent),
            "total_openings": len(board.postings),
            "window_days": RECENT_WINDOW_DAYS,
            "sample_titles": [p.title for p in recent[:10]],
        },
        evidence=[_evidence(board, observed, excerpt=summary)],
        confidence=0.85,
        strength=strength,
    )


def _department_surge_signals(
    board: JobBoard,
    recent: list[JobPosting],
    *,
    company_id: str,
    observed: datetime,
) -> list[Signal]:
    """One signal per department concentrating recent hiring."""
    by_dept: dict[str, list[JobPosting]] = defaultdict(list)
    for p in recent:
        dept = p.department or p.team
        if dept:
            by_dept[dept].append(p)

    signals: list[Signal] = []
    # Stable order (most openings first, then name) so output is deterministic.
    for dept, posts in sorted(by_dept.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(posts) < DEPT_SURGE_MIN:
            continue
        # Strength scales with the department's share of recent hiring and its
        # absolute size; a department that is most of the recent reqs is a clear
        # build-out.
        share = len(posts) / len(recent)
        size = _ramp(len(posts), DEPT_SURGE_MIN, STRONG_RECENT_OPENINGS)
        strength = round(min(0.6 * size + 0.4 * share, 1.0), 3)
        summary = (
            f"{len(posts)} of {len(recent)} recent openings are in "
            f"{dept} ({share:.0%}) — a concentrated build-out of that function, "
            "a common precursor to a new VP/Head-of hire to lead it."
        )
        signals.append(
            Signal(
                id=f"ats:{board.provider}:{board.token}:dept:{_id_part(dept)}",
                company_id=company_id,
                signal_type="department_surge",
                source=board.provider,
                observed_at=observed,
                effective_at=_freshest(posts),
                title=f"Department surge: {len(posts)} {dept} openings ({board.provider})",
                summary=summary,
                extracted_facts={
                    "provider": board.provider,
                    "token": board.token,
                    "department": dept,
                    "department_openings": len(posts),
                    "recent_openings": len(recent),
                    "share": round(share, 3),
                    "titles": [p.title for p in posts][:10],
                },
                evidence=[_evidence(board, observed, excerpt=summary)],
                confidence=0.85,
                strength=strength,
            )
        )
    return signals


def _greenfield_reason(posting: JobPosting, dept_counts: dict[str, int]) -> str | None:
    """Why (if at all) this posting reads as a greenfield-team first hire."""
    if _GREENFIELD_RE.search(posting.title):
        return "explicit new-team / first-hire language in the title"
    # A lone leadership req in a department with no other openings reads as a
    # leader hired to stand up a function that has no team yet — UNLESS the TITLE
    # names a routine support/ops/enablement org (L&D, IT/service desk,
    # facilities, compliance, …). Every company has those; a lone Head-of one is
    # standard structure, not a zero-to-one build (live finding D). We test the
    # title, not the department: a support word in a dept label must not veto a
    # genuine core title (dept "Workplace" must not suppress "Head of Engineering").
    dept = posting.department or posting.team
    if (
        _LEADERSHIP_RE.search(posting.title)
        and dept
        and dept_counts.get(dept, 0) == 1
        and not _SUPPORT_FUNCTION_RE.search(posting.title)
    ):
        return f"lone leadership req in {dept} with no team beneath it"
    return None


def _greenfield_signals(
    board: JobBoard,
    *,
    company_id: str,
    observed: datetime,
) -> list[Signal]:
    """Per-posting greenfield-team signals (strategic-language pillar).

    Considers ALL postings, not just recent ones: a founding/leadership req is a
    strategic signal whether or not it was opened this month.
    """
    # Count departments across the whole board so "lone leadership req" is
    # judged against the full board, not a recency-filtered subset.
    dept_counts: dict[str, int] = defaultdict(int)
    for p in board.postings:
        dept = p.department or p.team
        if dept:
            dept_counts[dept] += 1

    signals: list[Signal] = []
    for posting in board.postings:
        reason = _greenfield_reason(posting, dept_counts)
        if reason is None:
            continue
        dept = posting.department or posting.team
        explicit = bool(_GREENFIELD_RE.search(posting.title))
        summary = (
            f"'{posting.title}'"
            + (f" in {dept}" if dept else "")
            + f" — {reason}. A net-new team being stood up signals strategic "
            "investment in a brand-new function and a forming leadership seat."
        )
        signals.append(
            Signal(
                id=f"ats:{board.provider}:{board.token}:greenfield:{_id_part(posting.id or posting.title)}",
                company_id=company_id,
                signal_type="greenfield_team",
                source=board.provider,
                observed_at=observed,
                effective_at=posting.updated_at,
                title=f"Greenfield team: {posting.title} ({board.provider})",
                summary=summary,
                extracted_facts={
                    "provider": board.provider,
                    "token": board.token,
                    "posting_id": posting.id,
                    "posting_title": posting.title,
                    "department": dept,
                    "reason": reason,
                    "explicit_language": explicit,
                },
                evidence=[_evidence(board, observed, url=posting.url, excerpt=summary)],
                confidence=0.85,
                # Explicit founding language is a stronger signal than the
                # lone-leadership heuristic.
                strength=0.8 if explicit else 0.6,
            )
        )
    return signals


def signals_from_board(
    board: JobBoard,
    *,
    company_id: str,
    observed_at: datetime | None = None,
    now: datetime | None = None,
) -> list[Signal]:
    """Produce hiring-pattern Signals from one company's ATS board snapshot.

    `observed_at` stamps when the system saw the board; `now` is the reference
    point for the recency window (defaults to `observed_at`, else utcnow). Both
    are injectable so runs are deterministic and offline-testable.

    Returns an empty list when the board shows no meaningful hiring pattern.
    """
    observed = observed_at or _utcnow()
    # The recency reference defaults to when we observed the board; reuse the
    # same instant rather than a second clock read so the two never disagree.
    reference = now or observed

    recent = [p for p in board.postings if is_recent(p, reference)]

    signals: list[Signal] = []
    velocity = _velocity_signal(board, recent, company_id=company_id, observed=observed)
    if velocity is not None:
        signals.append(velocity)
    signals.extend(
        _department_surge_signals(
            board, recent, company_id=company_id, observed=observed
        )
    )
    signals.extend(_greenfield_signals(board, company_id=company_id, observed=observed))
    return signals
