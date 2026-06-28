"""Tests for 8-K Item 5.02 signal extraction.

Uses a real Apple Item 5.02 filing (Cook -> Executive Chair, Ternus appointed
CEO) as the successor-present fixture, plus a synthetic departure-only doc to
exercise the leadership-vacuum classifier.
"""

from datetime import date, datetime, timezone
from pathlib import Path

from jobfinder.signals.sec_8k import parse_item_502, signals_from_filing
from jobfinder.sources.edgar import Filing

FIXTURES = Path(__file__).parent / "fixtures"
OBSERVED = datetime(2026, 5, 1, tzinfo=timezone.utc)

APPLE_502 = (FIXTURES / "8k_item502_apple.txt").read_text()

# Synthetic: a departure with NO successor named in the filing.
# IMPORTANT: this carries the *full* standard Item 5.02 caption (which itself
# contains the words "Election" and "Appointment"). A realistic departure-only
# filing always does, so the parser must not read the caption as an event.
VACUUM_DOC = (
    "Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
    "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers. "
    "On May 1, 2026, Jane Doe, the Company's Chief Financial Officer, "
    "notified the Board of her resignation, effective immediately. "
    "The Company has commenced a search for a permanent successor."
)

# Synthetic: an appointment with NO departure disclosed in the body.
APPOINTMENT_ONLY_DOC = (
    "Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
    "Appointment of Certain Officers. On June 1, 2026, the Board appointed "
    "John Smith as Chief Financial Officer, effective immediately."
)


def _filing(items: list[str]) -> Filing:
    return Filing(
        cik="320193",
        accession_number="0001140361-26-015711",
        form="8-K",
        filing_date=date(2026, 4, 20),
        report_date=date(2026, 4, 17),
        items=items,
        primary_document="ef20071035_8k.htm",
    )


def test_parse_real_filing_detects_departure_and_appointment():
    events = parse_item_502(APPLE_502, item_known=True)
    assert events.has_item_502
    assert events.has_departure  # Cook transitions from CEO
    assert events.has_appointment  # Ternus appointed CEO
    assert events.successor_present
    assert not events.is_leadership_vacuum
    assert "CEO" in events.roles


def test_vacuum_doc_flags_leadership_gap():
    events = parse_item_502(VACUUM_DOC, item_known=True)
    assert events.has_departure
    # The full Item 5.02 caption mentions "Appointment"/"Election", but no
    # successor is actually disclosed in the body, so this is a vacuum.
    assert not events.has_appointment
    assert not events.successor_present
    assert events.is_leadership_vacuum
    assert "CFO" in events.roles


def test_appointment_only_does_not_emit_departure():
    events = parse_item_502(APPOINTMENT_ONLY_DOC, item_known=True)
    assert events.has_appointment
    # Caption says "Departure ...", but nobody actually departed in the body.
    assert not events.has_departure
    assert not events.is_leadership_vacuum

    signals = signals_from_filing(
        _filing(["5.02"]), APPOINTMENT_ONLY_DOC, company_id="co-x", observed_at=OBSERVED
    )
    assert {s.signal_type for s in signals} == {"8k_exec_appointment"}


def test_signals_from_real_filing_have_cited_evidence():
    signals = signals_from_filing(
        _filing(["5.02"]), APPLE_502, company_id="co-apple", observed_at=OBSERVED
    )
    types = {s.signal_type for s in signals}
    assert types == {"8k_exec_departure", "8k_exec_appointment"}
    for s in signals:
        # Citation rule (enforced by the schema) holds end-to-end.
        assert s.evidence and s.evidence[0].locator == "0001140361-26-015711"
        assert s.evidence[0].url.endswith("ef20071035_8k.htm")
    departure = next(s for s in signals if s.signal_type == "8k_exec_departure")
    assert departure.extracted_facts["leadership_vacuum"] is False


def test_vacuum_filing_produces_high_strength_departure():
    signals = signals_from_filing(
        _filing(["5.02"]), VACUUM_DOC, company_id="co-x", observed_at=OBSERVED
    )
    departure = next(s for s in signals if s.signal_type == "8k_exec_departure")
    assert departure.extracted_facts["leadership_vacuum"] is True
    assert departure.strength >= 0.7  # vacuum is the higher-value signal


def test_non_502_filing_yields_no_signals():
    # An earnings 8-K (Item 2.02) with no exec language.
    doc = "Item 2.02 Results of Operations. The Company reported quarterly revenue."
    signals = signals_from_filing(
        _filing(["2.02", "9.01"]), doc, company_id="co-x", observed_at=OBSERVED
    )
    assert signals == []
