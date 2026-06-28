"""Reporter — the last stage of the pipeline (plan section, README diagram).

Collectors -> Normalizer -> Correlator -> Scorer -> **Reporter**. Everything
upstream turns raw filings and job boards into ranked, evidence-backed
``Opportunity`` objects; the Reporter turns the *accumulated* store of those
opportunities into something a human reads each week.

Its reason to exist beyond ``cli.render`` is the cross-run view: "what changed
since last week". That is the whole point of the persistence layer stamping
``first_seen_at`` once and advancing ``updated_at`` on every save, plus the
carried-forward ``previous_score`` — until now no consumer read any of them.
The Reporter does: it annotates each ranked opportunity as **new** (first seen
this window) or **recurring**, shows score/rank **movement**, and lists the
**newly-appeared signals** that drove it.

Design constraints kept consistent with the rest of the codebase:

- **No network, no live clock.** It renders an already-loaded ``StoreDiff``
  (built by ``Store.diff(since=...)``); ``now`` is injected, never read from the
  wall clock, so output is fully reproducible and unit-testable offline.
- **Reads the wire contract, never weakens it.** Every digest line is traceable
  to the ``supporting_signal_ids`` already on each Opportunity and the
  ``Evidence`` already on each Signal — the explainable, cited property holds
  end to end.
"""

from __future__ import annotations

from datetime import datetime

from jobfinder.store.db import OpportunityChange, SignalChange, StoreDiff


def _age_phrase(when: datetime, now: datetime) -> str:
    """A coarse, human "n days ago" for the digest. Future dates read as 'today'
    rather than negative (clock skew / future-effective filings)."""
    days = (now - when).total_seconds() / 86_400.0
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    return f"{int(days)} days ago"


def _movement_tag(change: OpportunityChange) -> str:
    """A short ``[NEW]`` / ``[↑ +0.07]`` / ``[recurring]`` tag for one row.

    A row that was re-saved this window but whose score is unchanged at display
    precision reads ``[updated]`` rather than ``[recurring]`` so a stale prior
    delta isn't shown as movement; a never-touched row stays ``[recurring]``.
    """
    if change.is_new:
        return "[NEW]"
    delta = change.score_delta
    # Round to the SAME precision we display at before deciding "no movement",
    # so a sub-0.005 delta can't render the self-contradictory "[↑ +0.00]".
    if delta is not None and round(delta, 2) != 0:
        arrow = "↑" if delta > 0 else "↓"
        return f"[{arrow} {delta:+.2f}]"
    if change.changed_in_window:
        return "[updated]"
    return "[recurring]"


def _render_opportunity(
    rank: int, change: OpportunityChange, now: datetime
) -> list[str]:
    """One ranked opportunity block, mirroring ``cli.render``'s style plus the
    cross-run annotations."""
    opp = change.opportunity
    lines = [
        f"{rank}. {opp.company_id}  —  score {opp.score:.2f}  "
        f"(confidence {opp.confidence:.0%}, urgency {opp.urgency:.0%})  "
        f"{_movement_tag(change)}",
        f"   Target: {opp.target_persona}",
        f"   Why now: {opp.why_now}",
        f"   Next: {opp.recommended_next_action}",
        f"   First seen: {_age_phrase(change.first_seen_at, now)}; "
        f"last updated: {_age_phrase(change.updated_at, now)}",
        f"   Evidence (supporting signals): {', '.join(opp.supporting_signal_ids)}",
    ]
    return lines


def _render_new_signals(new_signals: list[SignalChange], now: datetime) -> list[str]:
    """The 'newly appeared signals' section — each cited to its evidence so the
    'what changed' view stays as explainable as the ranking itself."""
    lines = ["", f"Newly appeared signals ({len(new_signals)}):"]
    if not new_signals:
        lines.append("  (none)")
        return lines
    for change in new_signals:
        sig = change.signal
        cite = next(
            (e.url or e.locator for e in sig.evidence if (e.url or e.locator)),
            sig.source,
        )
        lines.append(
            f"  - [{sig.company_id}] {sig.signal_type}: {sig.title} "
            f"({_age_phrase(change.first_seen_at, now)})  — {cite}"
        )
    return lines


def render_digest(
    diff: StoreDiff,
    *,
    now: datetime,
    top: int | None = None,
) -> str:
    """Render a ``StoreDiff`` as a prioritized, cross-run text digest.

    With ``diff.since`` set, the header frames the report as "what changed since
    <date>" and opportunities carry new/recurring/movement tags; with no cutoff
    it degrades to a plain ranked standings digest. ``now`` is injected so the
    relative ages ("3 days ago") are reproducible.
    """
    opportunities = diff.opportunities
    shown = opportunities if top is None else opportunities[:top]

    lines: list[str] = []
    if diff.since is not None:
        header = (
            f"Opportunity digest — what changed since {diff.since.date().isoformat()}"
        )
    else:
        header = "Opportunity digest — current standings"
    lines.append(header)
    lines.append("=" * len(header))

    if not opportunities:
        lines.append("")
        lines.append("No opportunities on file.")
        return "\n".join(lines)

    new_count = sum(1 for c in opportunities if c.is_new)
    lines.append(
        f"{len(opportunities)} opportunities on file "
        f"({new_count} new this window); showing {len(shown)}."
    )

    for rank, change in enumerate(shown, start=1):
        lines.append("")
        lines.extend(_render_opportunity(rank, change, now))

    lines.extend(_render_new_signals(diff.new_signals, now))
    return "\n".join(lines)
