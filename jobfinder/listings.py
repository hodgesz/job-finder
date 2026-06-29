"""Listed-roles corroboration (Pillar I, corroboration view).

The product is the *hidden* role — the seat a leadership vacuum or a hiring
build-out implies before it is ever posted. But the public ATS boards we already
fetch (``jobfinder.sources.ats``, Slice 5) carry the company's *listed* roles,
and those are powerful corroboration: an 8-K CFO departure next to four open
Finance reqs reads as a function genuinely being built, whereas the same
departure next to a board of only Sales reqs reads as a routine backfill. That
distinction — surfacing the live reqs that sit *in the same function* as the
opportunity's target persona — is exactly the cure for the live-run finding that
successor-named/backfill departures were indistinguishable from real forming
roles.

This module is a pure, render-time *join*: it reads the already-fetched
``JobBoard``s for a company and an authoritative target function and produces a
``RoleCorroboration`` view. It deliberately adds **no** field to the wire
contract (``schemas.py`` stays pure) and persists nothing — listed roles are
corroboration shown next to an opportunity, not a stored artifact. Like the
reporter and fit modules it takes no network and an injected ``now``, so it is
fully deterministic and offline-testable.

Two facts are reused rather than re-implemented, so corroboration can never
disagree with scoring about what they mean:

- **"recent"** is ``jobfinder.signals.ats_hiring.is_recent`` (the same window +
  future-skew clamp the velocity/surge signals use).
- **"in the same function"** is ``jobfinder.scoring.match_persona`` (the same
  role/department -> persona table the scorer derives target personas from); a
  posting is in-function when its derived persona equals the *authoritative*
  target persona.

Crucially, ``target_persona`` here must be the persona a *signal actually
derived* for the opportunity, not the ``DEFAULT_PERSONA`` fallback the scorer
uses when no signal names a role. A funding-only opportunity defaults to a CFO
persona with no signal behind it; flagging the company's routine Finance reqs as
"in-function" for that opportunity would manufacture corroboration the evidence
never supported. The caller passes ``None`` in that case (see
``jobfinder.scoring.derive_persona``'s source id), and nothing is flagged
in-function — total/recent counts still render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from jobfinder.scoring import match_persona
from jobfinder.signals.ats_hiring import RECENT_WINDOW_DAYS, is_recent
from jobfinder.sources.ats import JobBoard, JobPosting


@dataclass(frozen=True)
class CorroboratingRole:
    """One live ATS posting shown as corroboration next to an opportunity."""

    title: str
    where: str | None  # department / team / location, best available
    url: str | None
    in_function: bool  # matches the opportunity's authoritative target persona
    recent: bool  # active within the hiring-velocity recency window


@dataclass(frozen=True)
class RoleCorroboration:
    """Listed-roles corroboration for one opportunity.

    ``total``/``recent``/``in_function`` are full counts across every board for
    the company; ``sample`` is a capped, ordered slice for display (in-function
    and recent roles first). ``board_urls`` are the human-facing board links so a
    reader can jump straight to the live reqs.
    """

    total: int
    recent: int
    in_function: int
    sample: list[CorroboratingRole] = field(default_factory=list)
    board_urls: list[str] = field(default_factory=list)

    @property
    def has_roles(self) -> bool:
        return self.total > 0


def _posting_persona(posting: JobPosting) -> str | None:
    """The persona a posting reads as, via the shared scorer rules.

    Fragments are tried title-first (the role itself is the strongest signal),
    then department/team, mirroring how ``scoring`` matches role text. Location
    is intentionally excluded — a place name must not match a persona rule.
    """
    fragments = [f for f in (posting.title, posting.department, posting.team) if f]
    return match_persona(fragments)


def _where(posting: JobPosting) -> str | None:
    """Best human "where" label for a posting: department/team, then location."""
    return posting.department or posting.team or posting.location


def corroborate_roles(
    boards: list[JobBoard],
    *,
    target_persona: str | None,
    now: datetime,
    limit: int = 5,
) -> RoleCorroboration:
    """Build the listed-roles corroboration view for one opportunity.

    Counts every posting across ``boards`` (a company may publish more than one),
    flags each as *recent* (within the ATS recency window, via the shared
    ``is_recent``) and *in-function* (its derived persona equals
    ``target_persona``). ``target_persona`` must be the persona a signal actually
    derived; pass ``None`` when the opportunity's persona is the scorer's default
    fallback, so unrelated reqs are never flagged in-function. The display
    ``sample`` is ordered in-function-first, then recent-first, then by title, and
    capped at ``limit`` so a long board does not bury the roles that actually
    corroborate the opportunity. Returns an all-zero view when there are no
    postings.
    """
    roles: list[CorroboratingRole] = []
    recent_count = 0
    in_function_count = 0
    for board in boards:
        for posting in board.postings:
            in_function = (
                target_persona is not None
                and _posting_persona(posting) == target_persona
            )
            recent = is_recent(posting, now)
            if recent:
                recent_count += 1
            if in_function:
                in_function_count += 1
            roles.append(
                CorroboratingRole(
                    title=posting.title,
                    where=_where(posting),
                    url=posting.url,
                    in_function=in_function,
                    recent=recent,
                )
            )

    # In-function first, then recent, then title — so the roles that corroborate
    # *this* opportunity surface ahead of unrelated reqs when the sample is
    # capped. (booleans sort False<True, so negate to put True first.)
    roles.sort(key=lambda r: (not r.in_function, not r.recent, r.title.lower()))

    # Preserve board order, de-duplicated, for the "see the board" links.
    board_urls = list(dict.fromkeys(b.url for b in boards if b.url))

    return RoleCorroboration(
        total=len(roles),
        recent=recent_count,
        in_function=in_function_count,
        sample=roles[:limit],
        board_urls=board_urls,
    )


def corroboration_lines(corro: RoleCorroboration, *, indent: str = "   ") -> list[str]:
    """Render a ``RoleCorroboration`` as indented digest lines (or none).

    Returns an empty list when there are no live reqs, so a pure-SEC opportunity
    (no ATS board fetched) prints nothing rather than an empty section.
    """
    if not corro.has_roles:
        return []
    # "active" not "opened": the recency timestamp is the provider's last-update
    # field (Greenhouse updated_at, Ashby publishedAt), which is an edit time, not
    # strictly an open date — so we don't overstate it as "opened".
    headline = (
        f"Listed roles: {corro.total} live "
        f"({corro.recent} active in the last {int(RECENT_WINDOW_DAYS)}d, "
        f"{corro.in_function} in-function)"
    )
    lines = [f"{indent}{headline}"]
    for role in corro.sample:
        tags = []
        if role.in_function:
            tags.append("in-function")
        if role.recent:
            tags.append("recent")
        tag = f" [{', '.join(tags)}]" if tags else ""
        where = f" — {role.where}" if role.where else ""
        link = f"  {role.url}" if role.url else ""
        lines.append(f"{indent}  • {role.title}{where}{tag}{link}")
    for url in corro.board_urls:
        lines.append(f"{indent}  Board: {url}")
    return lines
