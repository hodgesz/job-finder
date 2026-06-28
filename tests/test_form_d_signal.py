"""Tests for Form D funding signal extraction (deterministic, offline)."""

from datetime import date, datetime, timezone

from jobfinder.signals.form_d import signal_from_form_d, signals_from_form_d
from jobfinder.sources.edgar import Filing, FormD, RelatedPerson

OBSERVED = datetime(2026, 5, 1, tzinfo=timezone.utc)


def _filing(form: str = "D", accession: str = "0001950000-26-000003") -> Filing:
    return Filing(
        cik="1950000",
        accession_number=accession,
        form=form,
        filing_date=date(2026, 4, 20),
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
