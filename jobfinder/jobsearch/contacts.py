"""Contact discovery for a job — *who to reach* + a manual LinkedIn checklist.

For a VP-of-AI search, getting a human at the company to notice you matters as
much as the application itself. This module turns a ``CanonicalJob`` into a
**deterministic, ranked list of roles worth contacting** (the hiring-manager
chain, the existing AI/data function leader, an executive sponsor, the recruiter
owning the search) and renders it as a **manual checklist** the user works *by
hand*: each target carries plain-text the user pastes into LinkedIn's own search
box, looks up the person, and records the name back via the CLI (``add-contact``).

The operating model is strict and unchanged: **no LinkedIn scraping or
automation.** This module never touches the network — it only suggests *what to
search for*. LinkedIn enters solely through the user's manual review and
paste-back. Nothing here sends anything (outreach is a later, human-approved
slice); it assembles the contact data and the checklist only.

The target list is built from the company name on the canonical job (the only
identity we reliably have offline). It is intentionally generic-but-explainable:
every target says *why* it's worth contacting, so the user can skip the ones that
don't apply rather than trust an opaque ranking.
"""

from __future__ import annotations

from jobfinder.jobsearch.models import (
    CanonicalJob,
    ContactRole,
    ContactTarget,
)

# The ranked target-role template for a VP-of-AI search. Order IS the priority:
# the person the role most likely reports to (the exec hiring chain) first, the
# existing function leadership next, then the executive sponsor, then the
# recruiter who owns the search. Each entry is (role, label, rationale, and a
# LinkedIn *search hint* template). ``{company}`` is filled with the real company
# name; the search string is plain text the user pastes into LinkedIn BY HAND —
# the tool never issues it.
_TARGET_TEMPLATE: tuple[tuple[ContactRole, str, str, str], ...] = (
    (
        ContactRole.HIRING_MANAGER,
        "Likely hiring manager (CTO / Chief Data or AI Officer)",
        "A VP of AI most often reports to the CTO or a Chief Data/AI Officer — "
        "the person who owns the req and whose endorsement carries the search.",
        '"{company}" (CTO OR "Chief Technology Officer" OR "Chief Data Officer" '
        'OR "Chief AI Officer")',
    ),
    (
        ContactRole.EXECUTIVE,
        "CEO / founder (executive sponsor)",
        "For a senior exec hire the CEO or a founder is often directly involved; "
        "a warm note here can route you straight to the decision-maker.",
        '"{company}" (CEO OR founder OR "Chief Executive Officer")',
    ),
    (
        ContactRole.FUNCTION_LEADER,
        "Existing AI / data / analytics leadership",
        "Current VP/Head/Director of AI, Data or Analytics — a peer or the person "
        "you'd succeed or sit beside; they shape the role and can refer you in.",
        '"{company}" (VP OR Head OR Director) (AI OR "data science" OR '
        '"machine learning" OR analytics)',
    ),
    (
        ContactRole.RECRUITER,
        "Talent / executive recruiter owning the search",
        "The recruiter or talent partner running the search controls scheduling "
        "and can flag your application internally — often the fastest first touch.",
        '"{company}" (recruiter OR "talent acquisition" OR "executive recruiter" '
        'OR "technical recruiter")',
    ),
)


def target_contacts(job: CanonicalJob) -> list[ContactTarget]:
    """Build the ranked target-role list for one job. Deterministic + explainable.

    Returns the targets in priority order (1 = reach first), each with a
    paste-by-hand LinkedIn search hint scoped to the job's company. Returns an
    empty list when the job has no usable company name (nothing to search for).
    """
    company = (job.company or "").strip()
    if not company:
        return []
    targets: list[ContactTarget] = []
    for priority, (role, label, rationale, search_tmpl) in enumerate(
        _TARGET_TEMPLATE, start=1
    ):
        targets.append(
            ContactTarget(
                role=role,
                priority=priority,
                label=label,
                rationale=rationale,
                linkedin_search=search_tmpl.format(company=company),
            )
        )
    return targets


def render_checklist(job: CanonicalJob, targets: list[ContactTarget]) -> str:
    """Render the manual LinkedIn checklist for a job's target contacts.

    Each target prints its priority, who to look for, why, and the exact text to
    paste into LinkedIn's search box by hand — followed by the ``add-contact``
    command to record whoever the user finds. No network, no automation: the user
    drives LinkedIn manually and pastes names back.
    """
    company = (job.company or "").strip()
    title = job.title or "(untitled role)"
    lines: list[str] = []
    header = f"Manual contact checklist — {title} @ {company or '(unknown company)'}"
    lines.append(header)
    lines.append("=" * len(header))
    if not targets:
        lines.append("")
        lines.append("No company name on this job — nothing to search for.")
        return "\n".join(lines)
    lines.append("")
    lines.append(
        "Run each LinkedIn search BY HAND (the tool never queries LinkedIn), then "
        "record whoever you find with `add-contact`. Business emails only; anyone "
        "on the do-not-contact list is suppressed everywhere."
    )
    for target in targets:
        lines.append("")
        lines.append(f"{target.priority}. [{target.role.value}] {target.label}")
        lines.append(f"   Why: {target.rationale}")
        lines.append(f"   LinkedIn search (paste by hand): {target.linkedin_search}")
    return "\n".join(lines)
