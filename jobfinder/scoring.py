"""Weighted composite intent scoring (plan section 4).

This module turns a company's `Signal`s into a single ranked `Opportunity`.
The premise of the whole system is that *concurrent* signals — fresh capital
*and* a leadership gap, say — predict a forming senior role far better than any
one signal alone. The score is an explainable weighted sum of independent
components, so a human can always read *why* a company ranks where it does.

Component weights (plan section 4):

    liquidity          0.30   capital raised and ready to deploy (Form D)
    leadership_vacuum  0.25   an exec seat opened with no named successor (8-K)
    hiring_velocity    0.20   surge of junior reqs implying a leadership hire (ATS)
    strategic_language 0.10   "first hire" / "greenfield" / stack-overhaul wording
    company_fit        0.10   how well the candidate matches this company
    recency            0.05   how fresh the freshest supporting signal is

In Slice 2 only the liquidity, leadership_vacuum, recency, and (caller-supplied)
company_fit components have sources wired in; the ATS-driven hiring_velocity and
strategic_language components score 0 until those collectors land (Slice 5+).
That is by design — the scorer is the stable seam; adding a source later just
lights up a component without changing this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from jobfinder.schemas import Opportunity, Signal

# Component weights. Kept as an ordered dict so the breakdown reads in a stable,
# human-meaningful order and the weights are visible in one place.
WEIGHTS: dict[str, float] = {
    "liquidity": 0.30,
    "leadership_vacuum": 0.25,
    "hiring_velocity": 0.20,
    "strategic_language": 0.10,
    "company_fit": 0.10,
    "recency": 0.05,
}

# How signal types map onto scoring components. A component takes the strongest
# (max) contributing signal so multiple weak filings don't inflate it.
_LIQUIDITY_TYPES = {"form_d_funding", "form_d_amendment", "news_funding"}
_VACUUM_TYPES = {"8k_exec_departure"}
_HIRING_TYPES = {"ats_hiring_velocity", "department_surge"}
_STRATEGIC_TYPES = {"greenfield_team", "tech_stack_change"}

# A supporting signal older than this (days) contributes no recency. Senior
# searches move on a months-not-years horizon; ~120 days is a generous window.
RECENCY_HORIZON_DAYS = 120.0

# The persona this slice's demo hunts for. Callers can override per-opportunity
# via `build_opportunity(..., target_persona=...)`; a signal-driven persona
# model (e.g. read the departed role) is future work, not wired yet.
DEFAULT_PERSONA = "CFO / VP Finance"


@dataclass(frozen=True)
class ScoreComponent:
    """One weighted component of an opportunity score, with its provenance."""

    name: str
    weight: float
    raw: float  # the component's own [0, 1] score before weighting
    contribution: float  # weight * raw
    signal_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class ScoreBreakdown:
    """Explainable result of scoring one company's signals."""

    company_id: str
    score: float
    components: list[ScoreComponent]
    supporting_signal_ids: list[str]

    def component(self, name: str) -> ScoreComponent:
        return next(c for c in self.components if c.name == name)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _signal_time(signal: Signal) -> datetime:
    """Best available timestamp for recency: the underlying event if known,
    else when we observed it."""
    return signal.effective_at or signal.observed_at


def _max_strength(signals: list[Signal], types: set[str]) -> tuple[float, list[str]]:
    """Strength of the strongest signal whose type is in `types`, with its id."""
    matches = [s for s in signals if s.signal_type in types]
    if not matches:
        return 0.0, []
    best = max(matches, key=lambda s: s.strength)
    return best.strength, [best.id]


def _vacuum_score(signals: list[Signal]) -> tuple[float, list[str], str]:
    """Leadership-vacuum component.

    A departure flagged as a leadership_vacuum (no named successor) is the
    high-value case; a departure *with* a successor still counts, but weakly.
    """
    departures = [s for s in signals if s.signal_type in _VACUUM_TYPES]
    if not departures:
        return 0.0, [], "no executive departure on file"
    best = max(departures, key=lambda s: s.strength)
    is_vacuum = bool(best.extracted_facts.get("leadership_vacuum"))
    note = (
        "departure with no named successor (open search)"
        if is_vacuum
        else "departure, but a successor was named"
    )
    return best.strength, [best.id], note


def _recency_score(signals: list[Signal], now: datetime) -> tuple[float, list[str]]:
    """Linear decay from 1.0 (today) to 0.0 at the horizon, on the freshest
    supporting signal."""
    if not signals:
        return 0.0, []
    freshest = max(signals, key=_signal_time)
    age_days = (now - _signal_time(freshest)).total_seconds() / 86_400.0
    # Clamp both ends: a future-dated signal (clock skew, a filing whose event
    # is "effective [future date]") has negative age and must not push recency
    # — and therefore the composite score — above 1.0, which would fail the
    # Opportunity score validator.
    raw = max(0.0, min(1.0, 1.0 - age_days / RECENCY_HORIZON_DAYS))
    return round(raw, 3), [freshest.id]


def score_company(
    company_id: str,
    signals: list[Signal],
    *,
    company_fit: float = 0.5,
    now: datetime | None = None,
) -> ScoreBreakdown:
    """Compute the weighted composite score for one company's signals.

    `company_fit` is a caller-supplied [0, 1] match score (firmographics,
    candidate background); it defaults to a neutral 0.5 until a fit model exists.
    """
    now = now or _utcnow()

    liquidity_raw, liquidity_ids = _max_strength(signals, _LIQUIDITY_TYPES)
    vacuum_raw, vacuum_ids, vacuum_note = _vacuum_score(signals)
    hiring_raw, hiring_ids = _max_strength(signals, _HIRING_TYPES)
    strategic_raw, strategic_ids = _max_strength(signals, _STRATEGIC_TYPES)
    recency_raw, recency_ids = _recency_score(signals, now)
    fit_raw = max(0.0, min(company_fit, 1.0))

    raws: dict[str, tuple[float, list[str], str]] = {
        "liquidity": (liquidity_raw, liquidity_ids, "Form D capital raised"),
        "leadership_vacuum": (vacuum_raw, vacuum_ids, vacuum_note),
        "hiring_velocity": (hiring_raw, hiring_ids, "junior-req surge (ATS)"),
        "strategic_language": (
            strategic_raw,
            strategic_ids,
            "greenfield/stack wording",
        ),
        "company_fit": (fit_raw, [], "caller-supplied fit"),
        "recency": (recency_raw, recency_ids, "freshness of newest signal"),
    }

    components: list[ScoreComponent] = []
    for name, weight in WEIGHTS.items():
        raw, ids, note = raws[name]
        components.append(
            ScoreComponent(
                name=name,
                weight=weight,
                raw=round(raw, 3),
                contribution=round(weight * raw, 4),
                signal_ids=ids,
                note=note,
            )
        )

    score = round(sum(c.contribution for c in components), 4)
    # Citation rule: only the components actually backed by signals contribute
    # ids. (company_fit/recency reuse existing signals; dedupe preserving order.)
    supporting = list(dict.fromkeys(sid for c in components for sid in c.signal_ids))
    return ScoreBreakdown(
        company_id=company_id,
        score=score,
        components=components,
        supporting_signal_ids=supporting,
    )


def _why_now(breakdown: ScoreBreakdown) -> str:
    """Human-readable 'why now' built from the components that actually fired."""
    parts: list[str] = []
    for c in breakdown.components:
        if c.raw <= 0 or not c.signal_ids:
            continue
        parts.append(f"{c.note} ({c.raw:.0%}, weight {c.weight:.0%})")
    if not parts:
        return "No active signals; ranked on baseline fit only."
    return "Concurrent signals: " + "; ".join(parts) + "."


def build_opportunity(
    breakdown: ScoreBreakdown,
    *,
    signals: list[Signal],
    target_persona: str | None = None,
    opportunity_id: str | None = None,
) -> Opportunity:
    """Turn a score breakdown into an evidence-backed `Opportunity`.

    Requires at least one supporting signal (the schema enforces this); callers
    should filter out zero-signal companies before building.
    """
    persona = target_persona or DEFAULT_PERSONA

    # Confidence: how sure we are the signals are real — take the strongest
    # supporting signal's confidence (they're independent observations).
    supporting = [s for s in signals if s.id in set(breakdown.supporting_signal_ids)]
    confidence = round(max((s.confidence for s in supporting), default=0.0), 3)

    # Urgency: a leadership vacuum on a fresh filing is the time-critical case.
    vacuum = breakdown.component("leadership_vacuum").raw
    recency = breakdown.component("recency").raw
    urgency = round(min(0.6 * vacuum + 0.4 * recency, 1.0), 3)

    fit_score = breakdown.component("company_fit").raw

    return Opportunity(
        id=opportunity_id or f"opp:{breakdown.company_id}",
        company_id=breakdown.company_id,
        target_persona=persona,
        opportunity_type="hidden_role_likely",
        score=breakdown.score,
        confidence=confidence,
        urgency=urgency,
        fit_score=fit_score,
        why_now=_why_now(breakdown),
        recommended_next_action=(
            "Warm intro to the CEO/board citing the funding and the open seat; "
            "position as a pre-search candidate before the role is posted."
        ),
        supporting_signal_ids=breakdown.supporting_signal_ids,
    )


def rank_opportunities(
    signals_by_company: dict[str, list[Signal]],
    *,
    company_fit: dict[str, float] | None = None,
    now: datetime | None = None,
) -> list[Opportunity]:
    """Score every company and return Opportunities ranked best-first.

    Companies with no supporting signals are skipped (the schema forbids an
    Opportunity with no citations, and they carry no intent anyway).
    """
    company_fit = company_fit or {}
    opportunities: list[Opportunity] = []
    for company_id, signals in signals_by_company.items():
        if not signals:
            continue
        breakdown = score_company(
            company_id,
            signals,
            company_fit=company_fit.get(company_id, 0.5),
            now=now,
        )
        if not breakdown.supporting_signal_ids:
            continue
        opportunities.append(build_opportunity(breakdown, signals=signals))
    opportunities.sort(key=lambda o: o.score, reverse=True)
    return opportunities
