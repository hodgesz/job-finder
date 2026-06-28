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


def _utcnow() -> datetime:
    # Wrapped so callers/tests can monkeypatch if they need determinism.
    return datetime.now(timezone.utc)


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
    intent with nothing raised yet carries no budget signal).
    """
    if not form_d.total_amount_sold:
        return None

    observed = observed_at or _utcnow()
    effective = (
        datetime.combine(filing.filing_date, datetime.min.time(), tzinfo=timezone.utc)
        if filing.filing_date
        else None
    )
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
        strength=_strength(form_d),
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
