"""Form D funding signal extraction.

SEC Form D reports a private securities offering (Reg D), filed within 15 days
of first sale — well ahead of any curated funding press release. A company
that has just closed a sizeable round has both a *mandate* to build go-to-market
and headcount infrastructure and the *budget* to hire it. That is the funding
signal this module emits.

The interpretation (plan Pillar II):

    Large totalAmountSold + little totalRemaining
        -> capital is in the door, ready to deploy on hiring.

Unlike 8-K prose, Form D is fully structured XML, so this stage is purely
deterministic — no LLM pass is needed. We read the already-parsed `FormD`
record (see `jobfinder.sources.edgar.parse_form_d`) and turn it into an
evidence-backed `Signal`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from jobfinder.schemas import Evidence, Signal
from jobfinder.sources.edgar import Filing, FormD

# Below this raised amount a round is unlikely to fund a senior GTM/finance
# build-out; we still emit a signal but at low strength. Above the high
# threshold the round is squarely in "must scale now" territory.
MIN_MEANINGFUL_RAISE = 1_000_000.0
STRONG_RAISE = 25_000_000.0

# A Form D reports *when* capital was raised; that recency is the whole point of
# the funding signal. A raise that closed years ago has already funded whatever
# build-out it was going to fund — it is not evidence of a hiring window *today*.
# So beyond this horizon we emit no funding signal at all (the recency floor),
# and within it the signal's strength decays linearly with age. ~1 year covers a
# typical post-raise hiring ramp. This is the single source of truth for "how old
# is too old"; the live collector reads it to avoid fetching ancient Form D XML
# it would only discard (see `EdgarClient.recent_form_d(since=...)`).
#
# The scorer's own `recency` component (weight 0.05, `scoring.RECENCY_HORIZON_DAYS`
# = 120) is far too small to dampen a stale-but-large raise on its own — a 2020
# Form D scoring "capital raised 100%" still topped the ranking — which is why the
# floor and decay belong here, at the signal level, not in the composite score.
# This horizon is deliberately distinct from (and longer than) the scorer's: it
# governs whether a funding signal *exists at all*, whereas the scorer's recency
# fine-tunes an already-emitted signal's freshness. The within-horizon decay and
# the scorer's recency only overlap in the ~90-120 day band, where mild
# double-dampening of a months-old raise is acceptable and intended.
FORM_D_RECENCY_HORIZON_DAYS = 365.0
# Within this many days of filing the raise is treated as fully fresh (the
# capital just landed); past it the age factor ramps down linearly to 0 at the
# horizon.
FORM_D_FRESH_DAYS = 90.0


def _utcnow() -> datetime:
    # Wrapped so callers/tests can monkeypatch if they need determinism.
    return datetime.now(timezone.utc)


def _age_factor(age_days: float) -> float:
    """Freshness multiplier in [0, 1] for a raise ``age_days`` old.

    Full weight while the capital is fresh (<= ``FORM_D_FRESH_DAYS``), then a
    linear decay to 0 at ``FORM_D_RECENCY_HORIZON_DAYS``. A negative age (a
    future-dated filing / clock skew) clamps to fully fresh.
    """
    if age_days <= FORM_D_FRESH_DAYS:
        return 1.0
    span = FORM_D_RECENCY_HORIZON_DAYS - FORM_D_FRESH_DAYS
    return max(0.0, 1.0 - (age_days - FORM_D_FRESH_DAYS) / span)


def _fraction_sold(form_d: FormD) -> float | None:
    """Fraction of the offering already sold, in [0, 1], or None if unknown.

    Uses totalRemaining when present (the most direct measure of how much
    capital is in the door); otherwise derives it from offering vs. sold.
    """
    sold = form_d.total_amount_sold
    if sold is None:
        return None
    if form_d.total_remaining is not None:
        denom = sold + form_d.total_remaining
        return sold / denom if denom > 0 else None
    if form_d.total_offering_amount and form_d.total_offering_amount > 0:
        return min(sold / form_d.total_offering_amount, 1.0)
    return None


def _strength(form_d: FormD) -> float:
    """Score the funding signal in [0, 1].

    Two factors: how *large* the raise is (log-ish bucketing against the
    thresholds) and how *complete* it is (a fully-sold round is budget ready
    to deploy; a barely-started one is just an intent to raise).
    """
    sold = form_d.total_amount_sold or 0.0
    if sold < MIN_MEANINGFUL_RAISE:
        size = 0.2
    elif sold >= STRONG_RAISE:
        size = 1.0
    else:
        # Linear ramp between the meaningful and strong thresholds.
        span = STRONG_RAISE - MIN_MEANINGFUL_RAISE
        size = 0.4 + 0.6 * ((sold - MIN_MEANINGFUL_RAISE) / span)

    fraction = _fraction_sold(form_d)
    completeness = fraction if fraction is not None else 0.7  # neutral-ish prior
    # Weight size more than completeness: a huge half-sold round still matters.
    return round(min(0.7 * size + 0.3 * completeness, 1.0), 3)


def _money(value: float | None) -> str:
    return f"${value:,.0f}" if value is not None else "an undisclosed amount"


def signal_from_form_d(
    filing: Filing,
    form_d: FormD,
    *,
    company_id: str,
    observed_at: datetime | None = None,
) -> Signal | None:
    """Produce a funding Signal from one Form D filing + its parsed data.

    Returns ``None`` when the filing reports no amount sold (a bare notice of
    intent with nothing raised yet carries no budget signal), or when the raise
    is older than ``FORM_D_RECENCY_HORIZON_DAYS`` — a years-old round is not a
    hiring signal today (the recency floor).
    """
    if not form_d.total_amount_sold:
        return None

    observed = observed_at or _utcnow()
    effective = (
        datetime.combine(filing.filing_date, datetime.min.time(), tzinfo=timezone.utc)
        if filing.filing_date
        else None
    )

    # Recency floor + decay: capital raised long ago no longer signals a hiring
    # window. With no filing date we can't age it, so we treat it as current
    # (fresh) rather than silently dropping it. Age is measured from the event
    # (filing date) to when we observed it.
    age_factor = 1.0
    if effective is not None:
        age_days = (observed - effective).total_seconds() / 86_400.0
        if age_days >= FORM_D_RECENCY_HORIZON_DAYS:
            return None
        age_factor = _age_factor(age_days)

    signal_type = "form_d_amendment" if form_d.is_amendment else "form_d_funding"

    fraction = _fraction_sold(form_d)
    fraction_str = (
        f"{fraction:.0%} of the offering sold"
        if fraction is not None
        else ("an open-ended offering")
    )
    summary = (
        f"Form D reports {_money(form_d.total_amount_sold)} raised "
        f"({fraction_str}"
        + (
            f", {_money(form_d.total_remaining)} remaining"
            if form_d.total_remaining is not None
            else ""
        )
        + "). Freshly available budget and a mandate to build out the team — "
        "an optimal window for a forming finance/GTM leadership role."
    )

    evidence = [
        Evidence(
            source="sec_edgar",
            url=filing.primary_document_url,
            locator=filing.accession_number,
            excerpt=summary[:300],
            retrieved_at=observed,
        )
    ]

    facts = {
        "issuer_name": form_d.issuer_name,
        "total_offering_amount": form_d.total_offering_amount,
        "total_amount_sold": form_d.total_amount_sold,
        "total_remaining": form_d.total_remaining,
        "fraction_sold": fraction,
        "industry_group": form_d.industry_group,
        "is_amendment": form_d.is_amendment,
        "related_persons": [
            {"name": p.name, "relationships": p.relationships}
            for p in form_d.related_persons
        ],
    }

    return Signal(
        id=f"{filing.accession_number}:form_d",
        company_id=company_id,
        signal_type=signal_type,
        source="sec_edgar",
        observed_at=observed,
        effective_at=effective,
        title=(
            f"Form D funding: {_money(form_d.total_amount_sold)} raised"
            + (" (amendment)" if form_d.is_amendment else "")
        ),
        summary=summary,
        extracted_facts=facts,
        evidence=evidence,
        # Structured XML — high confidence the numbers are what they say.
        confidence=0.95,
        # Strength blends the raise's size/completeness with its freshness: a
        # large raise that closed 10 months ago is a weaker hiring signal than
        # the same raise last month (and one past the horizon emitted no signal
        # at all, above).
        strength=round(_strength(form_d) * age_factor, 3),
    )


def signals_from_form_d(
    filing: Filing,
    form_d: FormD,
    *,
    company_id: str,
    observed_at: datetime | None = None,
) -> list[Signal]:
    """List-returning wrapper mirroring `sec_8k.signals_from_filing`."""
    signal = signal_from_form_d(
        filing, form_d, company_id=company_id, observed_at=observed_at
    )
    return [signal] if signal is not None else []
