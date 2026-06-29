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

All six components now have sources wired in: liquidity (Form D) and
leadership_vacuum (8-K) since Slice 2; hiring_velocity and strategic_language,
fed by the ATS hiring-pattern signals (``jobfinder.signals.ats_hiring``), since
Slice 5; and — as of Slice 8 — ``company_fit``, derived from firmographics by
``jobfinder.fit`` and threaded in by the pipeline as a score + a human reason.
``recency`` derives from the freshest supporting signal. This validated the
design repeatedly: lighting up each of those components required *no* change to
this scorer — the new signal types were already mapped (see
``_HIRING_TYPES``/``_STRATEGIC_TYPES``), ``company_fit`` was already a
caller-supplied float, and the weights already summed to 1.0; the scorer is the
stable seam.
"""

from __future__ import annotations

import re
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

# Persona used when no supporting signal names a role or department of its own
# (e.g. a funding-only opportunity — capital raised is a liquidity signal, not a
# persona source). Also the explicit override fallback for `build_opportunity`.
DEFAULT_PERSONA = "CFO / VP Finance"

# Which signal types carry a role/department we can read a persona from, in
# priority order. A confirmed leadership *vacuum* (an exec seat already vacated,
# from an 8-K) is the strongest persona source — there is a specific open role —
# ahead of a hiring build-out (a department surge) and a founding/greenfield req,
# which only *imply* a forming seat. Funding signals carry no role and are absent
# here, so a funding-only opportunity falls back to DEFAULT_PERSONA.
_PERSONA_SIGNAL_PRIORITY = (
    "8k_exec_departure",
    "department_surge",
    "greenfield_team",
)

# Maps a role/department mention (matched case-insensitively, first hit wins) to
# the persona to hunt for. The patterns read both the explicit role strings 8-K
# extraction carries (`extracted_facts["roles"]`, e.g. "Chief Financial Officer",
# "CFO") and the department/title text the ATS signals carry.
#
# ORDER MATTERS — rules are tried top to bottom, first match wins, so the table
# is ordered SPECIFIC FUNCTION FIRST, BROAD CONTAINER LAST. Compound department
# names pair a function with a container word ("People Operations", "Revenue
# Operations", "Data Platform"); the function is the persona-bearing half, so its
# rule must precede the container's. Concretely: the broad "operations"/"ops"
# rule is last (so "<Function> Operations" maps to <Function>, and only a bare
# "Operations"/"COO" reaches it), and "data" precedes engineering (so "Data
# Platform" reads as data, not engineering's "platform"). The remaining function
# rules are mutually exclusive on realistic inputs, so their relative order is
# immaterial.
_PERSONA_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), persona)
    for pat, persona in (
        (
            r"chief financial officer|\bCFO\b|finance|controller|accounting|FP&A",
            "CFO / VP Finance",
        ),
        (
            r"chief revenue officer|\bCRO\b|revenue|\bsales\b|go-to-market|\bGTM\b",
            "CRO / VP Sales",
        ),
        (
            r"chief marketing officer|\bCMO\b|marketing|growth|\bbrand\b",
            "CMO / VP Marketing",
        ),
        (r"chief product officer|\bCPO\b|\bproduct\b", "VP Product / Head of Product"),
        (
            r"\bpeople\b|\bHR\b|human resources|talent|recruiting",
            "VP People / Head of HR",
        ),
        (
            r"chief information security officer|\bCISO\b|security",
            "CISO / Head of Security",
        ),
        # Before engineering so "Data Platform" reads as data, not "platform".
        (r"\bdata\b|machine learning|\bML\b|analytics", "VP Data / Head of Data"),
        (
            r"chief technology officer|\bCTO\b|engineering|\bplatform\b|infrastructure|software",
            "VP Engineering / Engineering leader",
        ),
        # "president" must not be preceded by "vice"/"VP": a *Vice* President is
        # not a CEO. A bare "President" (or CEO) still maps here.
        (
            r"chief executive officer|\bCEO\b|(?<!vice )(?<!vice-)(?<!v )\bpresident\b",
            "CEO / President",
        ),
        # Broad container word, matched last: "<Function> Operations" already
        # resolved to <Function> above; only bare ops / a COO reaches here.
        (r"chief operating officer|\bCOO\b|operations|\bops\b", "COO / VP Operations"),
    )
)


def _persona_fragments(signal: Signal) -> list[str]:
    """The role/department text fragments to match persona rules against, in the
    order they were disclosed.

    8-K departures carry the vacated role(s) in ``extracted_facts['roles']``; the
    ATS signals carry a ``department`` and (for greenfield) a ``posting_title``.
    Each is kept as its own fragment — NOT joined into one blob — so a signal
    disclosing several roles (e.g. a CTO *and* a CFO departing in one filing) is
    matched role-by-role in listed order, letting the *primary* (first-listed)
    role win rather than whichever role a `_PERSONA_RULES` pattern happens to sit
    earliest in the table.
    """
    facts = signal.extracted_facts
    fragments: list[str] = []
    roles = facts.get("roles")
    if isinstance(roles, list):
        fragments.extend(str(r) for r in roles if r)
    for key in ("department", "posting_title"):
        value = facts.get(key)
        if value:
            fragments.append(str(value))
    return fragments


def match_persona(fragments: list[str]) -> str | None:
    """Persona for the first fragment that matches any rule, or None.

    Fragments are tried in order (disclosed order), and within a fragment the
    rule table's first match wins. Trying fragment-by-fragment means the
    first-listed role/department drives the persona, so role *listing* order beats
    rule-table order when a signal carries more than one role.

    Public because the listed-roles corroboration (``jobfinder.listings``) reads
    the same persona rules to decide whether a live ATS posting is *in the same
    function* as an opportunity's target persona — one source of truth for the
    role->persona mapping across scoring and corroboration.
    """
    for fragment in fragments:
        for pattern, persona in _PERSONA_RULES:
            if pattern.search(fragment):
                return persona
    return None


def derive_persona(signals: list[Signal]) -> tuple[str, str | None]:
    """Derive the target persona from the signals backing an opportunity.

    Returns ``(persona, source_signal_id)``. The source id keeps the choice
    explainable — the digest can cite which signal set the persona — and is None
    when no signal named a usable role/department (funding-only opportunities),
    in which case the persona falls back to ``DEFAULT_PERSONA``.

    Deterministic and table-driven (no LLM): we walk the persona-bearing signal
    types in priority order (vacuum > surge > greenfield) and, within a type, the
    strongest signal first, returning the first persona a rule matches.
    """
    by_type: dict[str, list[Signal]] = {}
    for s in signals:
        by_type.setdefault(s.signal_type, []).append(s)

    for signal_type in _PERSONA_SIGNAL_PRIORITY:
        for signal in sorted(
            by_type.get(signal_type, []), key=lambda s: s.strength, reverse=True
        ):
            persona = match_persona(_persona_fragments(signal))
            if persona is not None:
                return persona, signal.id
    return DEFAULT_PERSONA, None


@dataclass(frozen=True)
class ScoreComponent:
    """One weighted component of an opportunity score, with its provenance.

    `note` is a short human label for the component. `reason` carries a
    *structured* explanation for components whose provenance is not a signal id
    (today only `company_fit`, whose firmographic reason has no Signal behind
    it) — so `why_now` can surface it directly rather than re-parsing it out of
    the display label.
    """

    name: str
    weight: float
    raw: float  # the component's own [0, 1] score before weighting
    contribution: float  # weight * raw
    signal_ids: list[str] = field(default_factory=list)
    note: str = ""
    reason: str = ""


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
    company_fit_reason: str | None = None,
    now: datetime | None = None,
) -> ScoreBreakdown:
    """Compute the weighted composite score for one company's signals.

    `company_fit` is a [0, 1] match score derived from firmographics against the
    candidate profile (see `jobfinder.fit.assess_fit`); it defaults to a neutral
    0.5 when a caller supplies no fit. `company_fit_reason` is the matching short
    human explanation that rides in the component note (e.g. "Robotics matches
    target sector; Series B near target stage"), keeping a derived fit as
    explainable as the signal-backed components.
    """
    now = now or _utcnow()

    liquidity_raw, liquidity_ids = _max_strength(signals, _LIQUIDITY_TYPES)
    vacuum_raw, vacuum_ids, vacuum_note = _vacuum_score(signals)
    hiring_raw, hiring_ids = _max_strength(signals, _HIRING_TYPES)
    strategic_raw, strategic_ids = _max_strength(signals, _STRATEGIC_TYPES)
    recency_raw, recency_ids = _recency_score(signals, now)
    fit_raw = max(0.0, min(company_fit, 1.0))
    # The fit reason is carried structurally (not folded into the note) so
    # `why_now` can surface it without re-parsing the display label.
    fit_note = "firmographic fit" if company_fit_reason else "caller-supplied fit"

    # name -> (raw, signal_ids, note, reason). Only company_fit carries a
    # structured reason today (its provenance is firmographic, not a Signal).
    raws: dict[str, tuple[float, list[str], str, str]] = {
        "liquidity": (liquidity_raw, liquidity_ids, "Form D capital raised", ""),
        "leadership_vacuum": (vacuum_raw, vacuum_ids, vacuum_note, ""),
        "hiring_velocity": (hiring_raw, hiring_ids, "junior-req surge (ATS)", ""),
        "strategic_language": (
            strategic_raw,
            strategic_ids,
            "greenfield/stack wording",
            "",
        ),
        "company_fit": (fit_raw, [], fit_note, company_fit_reason or ""),
        "recency": (recency_raw, recency_ids, "freshness of newest signal", ""),
    }

    components: list[ScoreComponent] = []
    for name, weight in WEIGHTS.items():
        raw, ids, note, reason = raws[name]
        components.append(
            ScoreComponent(
                name=name,
                weight=weight,
                raw=round(raw, 3),
                contribution=round(weight * raw, 4),
                signal_ids=ids,
                note=note,
                reason=reason,
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


def _why_now(
    breakdown: ScoreBreakdown,
    *,
    persona: str,
    persona_source: str | None = None,
) -> str:
    """Human-readable 'why now' built from the components that actually fired.

    When a specific signal drove the target persona, name it (and the persona)
    so the digest can explain *why this role* — keeping the persona as cited as
    the score itself.
    """
    parts: list[str] = []
    for c in breakdown.components:
        if c.raw <= 0 or not c.signal_ids:
            continue
        parts.append(f"{c.note} ({c.raw:.0%}, weight {c.weight:.0%})")
    base = (
        "Concurrent signals: " + "; ".join(parts) + "."
        if parts
        else "No active signals; ranked on baseline fit only."
    )
    # The company_fit component carries no signal ids (it's firmographic, not
    # signal-backed) so it's excluded from the loop above; surface a *derived*
    # fit explicitly — reading the structured `reason`, not re-parsing the note —
    # so the reason a company fits stays as visible as the score.
    fit = breakdown.component("company_fit")
    if fit.reason:
        base += f" Fit {fit.raw:.0%}: {fit.reason}."
    if persona_source is not None:
        base += f" Persona '{persona}' inferred from signal {persona_source}."
    return base


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

    When `target_persona` is None the persona is *derived* from the signals that
    actually scored the opportunity (`derive_persona`) rather than hardcoded — a
    CFO departure targets a finance leader, an Engineering surge targets an
    engineering leader — and the signal that set it is named in `why_now` so the
    choice stays explainable. An explicit `target_persona` always wins.
    """
    # Confidence: how sure we are the signals are real — take the strongest
    # supporting signal's confidence (they're independent observations).
    supporting = [s for s in signals if s.id in set(breakdown.supporting_signal_ids)]
    confidence = round(max((s.confidence for s in supporting), default=0.0), 3)

    # Derive the persona from the cited supporting signals only, so the choice is
    # traceable to a signal that actually scored this opportunity. An explicit
    # override skips derivation (and its provenance clause).
    persona_source: str | None = None
    if target_persona is not None:
        persona = target_persona
    else:
        persona, persona_source = derive_persona(supporting)

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
        why_now=_why_now(breakdown, persona=persona, persona_source=persona_source),
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
    company_fit_reasons: dict[str, str] | None = None,
    now: datetime | None = None,
) -> list[Opportunity]:
    """Score every company and return Opportunities ranked best-first.

    Companies with no supporting signals are skipped (the schema forbids an
    Opportunity with no citations, and they carry no intent anyway).

    `company_fit` / `company_fit_reasons` are per-company maps the pipeline
    builds from `jobfinder.fit`: the derived [0, 1] fit score and its matching
    human reason. A company absent from the maps falls back to the neutral 0.5.
    """
    company_fit = company_fit or {}
    company_fit_reasons = company_fit_reasons or {}
    opportunities: list[Opportunity] = []
    for company_id, signals in signals_by_company.items():
        if not signals:
            continue
        breakdown = score_company(
            company_id,
            signals,
            company_fit=company_fit.get(company_id, 0.5),
            company_fit_reason=company_fit_reasons.get(company_id),
            now=now,
        )
        if not breakdown.supporting_signal_ids:
            continue
        opportunities.append(build_opportunity(breakdown, signals=signals))
    opportunities.sort(key=lambda o: o.score, reverse=True)
    return opportunities
