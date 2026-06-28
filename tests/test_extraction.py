"""Tests for LLM-backed Item 5.02 extraction and its regex fallback.

No live API calls: a fake extractor is injected. These verify the
enhancement-not-dependency contract — LLM output is used when present, and
any failure degrades to the deterministic parser.
"""

from datetime import date, datetime, timezone

from jobfinder.signals.extraction import (
    ExecEvent,
    ExtractedEvents,
    RegexExtractor,
    extract_events,
    regex_fallback,
)
from jobfinder.signals.sec_8k import signals_from_filing
from jobfinder.sources.edgar import Filing

OBSERVED = datetime(2026, 5, 1, tzinfo=timezone.utc)

CAPTION = (
    "Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
    "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers. "
)
VACUUM_DOC = (
    CAPTION + "On May 1, 2026, Jane Doe, the Company's Chief Financial Officer, "
    "notified the Board of her resignation, effective immediately."
)


class FakeExtractor:
    """Returns a canned ExtractedEvents (or raises) — stands in for Gemini."""

    def __init__(self, result: ExtractedEvents | None = None, *, raises: bool = False):
        self._result = result
        self._raises = raises
        self.calls = 0

    def extract(self, document: str) -> ExtractedEvents:
        self.calls += 1
        if self._raises:
            raise RuntimeError("simulated LLM failure")
        return self._result


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


def test_injected_extractor_is_used():
    canned = ExtractedEvents(
        events=[ExecEvent(event_type="departure", officer_name="Jane Doe", role="CFO")],
        successor_named=False,
        extraction_method="llm",
    )
    fake = FakeExtractor(canned)
    result = extract_events(VACUUM_DOC, extractor=fake)
    assert fake.calls == 1
    assert result.extraction_method == "llm"
    assert result.is_leadership_vacuum


def test_extractor_failure_falls_back_to_regex():
    fake = FakeExtractor(raises=True)
    result = extract_events(VACUUM_DOC, extractor=fake, item_known=True)
    assert fake.calls == 1
    # Did not raise; degraded to the deterministic parser.
    assert result.extraction_method == "regex_fallback"
    assert result.has_departure
    assert result.is_leadership_vacuum


def test_regex_extractor_forces_deterministic_path():
    # RegexExtractor short-circuits the LLM even if one were configured, so a
    # caller (e.g. the CLI demo) gets reproducible, offline output.
    result = RegexExtractor().extract(VACUUM_DOC)
    assert result.extraction_method == "regex_fallback"
    assert result.has_departure
    assert result.is_leadership_vacuum
    # And injected into extract_events, it is used verbatim (no env lookup).
    via = extract_events(VACUUM_DOC, extractor=RegexExtractor())
    assert via.extraction_method == "regex_fallback"


def test_regex_fallback_maps_parser_output():
    result = regex_fallback(VACUUM_DOC, item_known=True)
    assert result.extraction_method == "regex_fallback"
    assert result.has_departure
    assert not result.has_appointment


def test_llm_officer_detail_flows_into_signal_facts():
    canned = ExtractedEvents(
        events=[
            ExecEvent(
                event_type="departure",
                officer_name="Jane Doe",
                role="Chief Financial Officer",
                effective_date="2026-05-01",
            )
        ],
        successor_named=False,
        extraction_method="llm",
    )
    signals = signals_from_filing(
        _filing(["5.02"]),
        VACUUM_DOC,
        company_id="co-x",
        observed_at=OBSERVED,
        extractor=FakeExtractor(canned),
    )
    departure = next(s for s in signals if s.signal_type == "8k_exec_departure")
    assert departure.confidence == 0.9  # LLM path is higher-confidence
    assert departure.extracted_facts["extraction_method"] == "llm"
    officers = departure.extracted_facts["officers"]
    assert officers[0]["name"] == "Jane Doe"
    assert officers[0]["effective_date"] == "2026-05-01"
    assert departure.extracted_facts["leadership_vacuum"] is True


def test_llm_can_classify_compensatory_only_as_no_event():
    # An LLM that correctly recognizes comp-only -> no departure/appointment.
    canned = ExtractedEvents(
        events=[ExecEvent(event_type="compensatory_only", role="CFO")],
        successor_named=False,
        extraction_method="llm",
    )
    signals = signals_from_filing(
        _filing(["5.02"]),
        CAPTION + "The Committee approved enhanced severance terms for the CFO.",
        company_id="co-x",
        observed_at=OBSERVED,
        extractor=FakeExtractor(canned),
    )
    assert signals == []
