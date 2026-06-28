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


def _movement_tag(change: OpportunityChange, *, has_window: bool) -> str:
    """A short ``[NEW]`` / ``[↑ +0.07]`` / ``[updated]`` / ``[recurring]`` tag.

    ``previous_score`` records movement on the *last* upsert only. In a
    "what changed since <date>" report (``has_window``) that delta is only
    honest if the row was actually (re)saved within the window — otherwise the
    movement predates the cutoff, so an untouched row reads ``[recurring]`` even
    if its stored prior score differs. In the plain standings view (no window)
    there is no cutoff to mislead about, so the last-upsert delta is shown
    directly. A sub-display-precision delta is rounded to the displayed 2dp
    before the zero check so it can't render a self-contradictory ``[↑ +0.00]``.
    """
    if change.is_new:
        return "[NEW]"
    delta = change.score_delta
    moved = delta is not None and round(delta, 2) != 0
    if has_window and not change.changed_in_window:
        # Untouched this window: any stored delta happened before the cutoff.
        return "[recurring]"
    if moved:
        arrow = "↑" if delta > 0 else "↓"
        return f"[{arrow} {delta:+.2f}]"
    # Touched this window with no real movement reads [updated]; in the
    # window-less standings view an unmoved row is simply [recurring].
    return "[updated]" if has_window else "[recurring]"


def _render_opportunity(
    rank: int, change: OpportunityChange, now: datetime, *, has_window: bool
) -> list[str]:
    """One ranked opportunity block, mirroring ``cli.render``'s style plus the
    cross-run annotations."""
    opp = change.opportunity
    lines = [
        f"{rank}. {opp.company_id}  —  score {opp.score:.2f}  "
        f"(confidence {opp.confidence:.0%}, urgency {opp.urgency:.0%})  "
        f"{_movement_tag(change, has_window=has_window)}",
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
    has_window = diff.since is not None
    opportunities = diff.opportunities
    shown = opportunities if top is None else opportunities[:top]

    lines: list[str] = []
    if has_window:
        header = (
            f"Opportunity digest — what changed since {diff.since.date().isoformat()}"
        )
    else:
        header = "Opportunity digest — current standings"
    lines.append(header)
    lines.append("=" * len(header))

    if opportunities:
        # The "new this window" count and the newly-appeared-signals section only
        # mean anything when there is a cutoff to be new *relative to*; without
        # --since this is a plain ranked standings view.
        count_line = f"{len(opportunities)} opportunities on file"
        if has_window:
            new_count = sum(1 for c in opportunities if c.is_new)
            count_line += f" ({new_count} new this window)"
        count_line += f"; showing {len(shown)}."
        lines.append(count_line)

        for rank, change in enumerate(shown, start=1):
            lines.append("")
            lines.extend(_render_opportunity(rank, change, now, has_window=has_window))
    else:
        lines.append("")
        lines.append("No opportunities on file.")

    # Even with no opportunities, a windowed report must still surface the
    # signals that newly appeared — they ARE the "what changed" the title
    # promises, and a signal can land before it scores into an opportunity.
    if has_window:
        lines.extend(_render_new_signals(diff.new_signals, now))
    return "\n".join(lines)
