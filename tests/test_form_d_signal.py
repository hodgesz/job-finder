"""Tests for Form D funding signal extraction (deterministic, offline)."""

from datetime import date, datetime, timezone

from jobfinder.signals.form_d import (
    FORM_D_FRESH_DAYS,
    FORM_D_RECENCY_HORIZON_DAYS,
    signal_from_form_d,
    signals_from_form_d,
)
from jobfinder.sources.edgar import Filing, FormD, RelatedPerson

OBSERVED = datetime(2026, 5, 1, tzinfo=timezone.utc)


def _filing(
    form: str = "D",
    accession: str = "0001950000-26-000003",
    filing_date: date | None = date(2026, 4, 20),
) -> Filing:
    return Filing(
        cik="1950000",
        accession_number=accession,
        form=form,
        filing_date=filing_date,
        report_date=None,
        items=[],
        primary_document="primary_doc.xml",
    )


def _form_d(**overrides) -> FormD:
    base = dict(
        issuer_cik="0001950000",
        issuer_name="Northwind Robotics Inc.",
        accession_number="0001950000-26-000003",
        total_offering_amount=60_000_000.0,
        total_amount_sold=55_000_000.0,
        total_remaining=5_000_000.0,
        industry_group="Technology",
        is_amendment=False,
        related_persons=[RelatedPerson(name="Ada Marsh", relationships=["Director"])],
    )
    base.update(overrides)
    return FormD(**base)


def test_funding_signal_has_cited_evidence_and_facts():
    signal = signal_from_form_d(
        _filing(), _form_d(), company_id="co-nw", observed_at=OBSERVED
    )
    assert signal is not None
    assert signal.signal_type == "form_d_funding"
    assert signal.company_id == "co-nw"
    # Citation rule holds end-to-end (schema enforces >=1 evidence).
    assert signal.evidence[0].locator == "0001950000-26-000003"
    assert signal.evidence[0].url.endswith("primary_doc.xml")
    assert signal.extracted_facts["total_amount_sold"] == 55_000_000.0
    assert signal.extracted_facts["fraction_sold"] is not None


def test_large_nearly_complete_raise_is_high_strength():
    signal = signal_from_form_d(
        _filing(), _form_d(), company_id="co-nw", observed_at=OBSERVED
    )
    # $55M sold, ~92% complete -> top strength bucket.
    assert signal.strength >= 0.9


def test_small_raise_is_low_strength():
    signal = signal_from_form_d(
        _filing(),
        _form_d(
            total_offering_amount=500_000.0,
            total_amount_sold=400_000.0,
            total_remaining=100_000.0,
        ),
        company_id="co-nw",
        observed_at=OBSERVED,
    )
    assert signal is not None
    assert signal.strength < 0.5


def test_no_amount_sold_yields_no_signal():
    assert (
        signal_from_form_d(
            _filing(),
            _form_d(total_amount_sold=None, total_remaining=None),
            company_id="co-nw",
            observed_at=OBSERVED,
        )
        is None
    )
    assert (
        signals_from_form_d(
            _filing(),
            _form_d(total_amount_sold=0.0),
            company_id="co-nw",
            observed_at=OBSERVED,
        )
        == []
    )


def test_amendment_uses_amendment_signal_type():
    signal = signal_from_form_d(
        _filing(form="D/A"),
        _form_d(is_amendment=True),
        company_id="co-nw",
        observed_at=OBSERVED,
    )
    assert signal.signal_type == "form_d_amendment"
    assert "amendment" in signal.title.lower()


def test_indefinite_remaining_handled_without_crashing():
    # Open-ended offering: remaining is None, fraction derives from offering.
    signal = signal_from_form_d(
        _filing(),
        _form_d(total_remaining=None),
        company_id="co-nw",
        observed_at=OBSERVED,
    )
    assert signal is not None
    # 55M of 60M offering ~= 92%.
    assert signal.extracted_facts["fraction_sold"] > 0.9


def test_stale_raise_past_horizon_yields_no_signal():
    # The headline live-run bug: a years-old Form D (MSFT 2020, Tesla 2018)
    # scored "capital raised 100%" and topped the ranking. A raise that closed
    # well past the recency horizon is not a hiring signal today, so no signal.
    stale = datetime(2026, 5, 1, tzinfo=timezone.utc)
    filing = _filing(filing_date=date(2024, 1, 1))  # ~16 months old
    assert (
        signal_from_form_d(filing, _form_d(), company_id="co-nw", observed_at=stale)
        is None
    )
    assert (
        signals_from_form_d(filing, _form_d(), company_id="co-nw", observed_at=stale)
        == []
    )


def test_age_decays_strength_within_horizon():
    # A large raise loses strength as it ages, even before the hard floor: the
    # same offering scores lower at 10 months old than freshly filed.
    fresh = signal_from_form_d(
        _filing(filing_date=date(2026, 4, 20)),  # 11 days before OBSERVED
        _form_d(),
        company_id="co-nw",
        observed_at=OBSERVED,
    )
    aging = signal_from_form_d(
        _filing(filing_date=date(2025, 7, 5)),  # ~300 days before OBSERVED
        _form_d(),
        company_id="co-nw",
        observed_at=OBSERVED,
    )
    assert fresh is not None and aging is not None
    assert fresh.strength >= 0.9
    assert aging.strength < fresh.strength


def test_just_inside_horizon_still_signals_just_outside_does_not():
    # Boundary check: a filing one day inside the horizon emits a (weak) signal;
    # one day past it emits nothing.
    from datetime import timedelta

    inside = _filing(
        filing_date=(OBSERVED - timedelta(days=FORM_D_RECENCY_HORIZON_DAYS - 1)).date()
    )
    outside = _filing(
        filing_date=(OBSERVED - timedelta(days=FORM_D_RECENCY_HORIZON_DAYS + 1)).date()
    )
    assert (
        signal_from_form_d(inside, _form_d(), company_id="co-nw", observed_at=OBSERVED)
        is not None
    )
    assert (
        signal_from_form_d(outside, _form_d(), company_id="co-nw", observed_at=OBSERVED)
        is None
    )


def test_exact_horizon_is_excluded():
    # The floor is `>=` the horizon, not `>`: a raise exactly at the horizon
    # would decay to strength 0 anyway, so it must emit no signal. Exercising the
    # exact boundary (not just ±1 day) guards against a `>=`->`>` regression that
    # would otherwise emit a dead 0-strength signal.
    from datetime import timedelta

    at_horizon = OBSERVED - timedelta(days=FORM_D_RECENCY_HORIZON_DAYS)
    # `_filing` truncates to a date, so build the filing so its midnight-UTC
    # effective time is exactly the horizon away from OBSERVED.
    filing = _filing(filing_date=at_horizon.date())
    # OBSERVED is midnight UTC, so age is an exact integer number of days here.
    assert (
        signal_from_form_d(filing, _form_d(), company_id="co", observed_at=OBSERVED)
        is None
    )


def test_fresh_window_does_not_decay():
    # Within the fresh window the capital is treated as fully current — no decay.
    from datetime import timedelta

    just_filed = _filing(filing_date=OBSERVED.date())
    edge_of_fresh = _filing(
        filing_date=(OBSERVED - timedelta(days=FORM_D_FRESH_DAYS)).date()
    )
    a = signal_from_form_d(just_filed, _form_d(), company_id="co", observed_at=OBSERVED)
    b = signal_from_form_d(
        edge_of_fresh, _form_d(), company_id="co", observed_at=OBSERVED
    )
    assert a is not None and b is not None
    assert a.strength == b.strength  # both fully fresh, identical strength


def test_missing_filing_date_is_treated_as_current():
    # With no date we can't age the raise; rather than silently dropping it we
    # keep it at full freshness (consistent with the collector keeping it).
    signal = signal_from_form_d(
        _filing(filing_date=None),
        _form_d(),
        company_id="co-nw",
        observed_at=OBSERVED,
    )
    assert signal is not None
    assert signal.strength >= 0.9
