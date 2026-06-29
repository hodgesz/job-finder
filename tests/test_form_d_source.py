"""Tests for the Form D source adapter. Network is injected, so offline."""

from datetime import date
from pathlib import Path

from jobfinder.sources.edgar import EdgarClient, Filing, parse_form_d

FIXTURES = Path(__file__).parent / "fixtures"

FORM_D_XML = (FIXTURES / "form_d_sample.xml").read_text()


def _fetcher(url: str) -> str:
    # Submissions index vs. primary document, switched on URL shape.
    if url.endswith(".json"):
        return (FIXTURES / "submissions_form_d.json").read_text()
    return FORM_D_XML


def test_parse_form_d_extracts_offering_amounts():
    form_d = parse_form_d(FORM_D_XML, accession_number="0001950000-26-000003")
    assert form_d.issuer_name == "Northwind Robotics Inc."
    assert form_d.issuer_cik == "0001950000"
    assert form_d.total_offering_amount == 60_000_000
    assert form_d.total_amount_sold == 55_000_000
    assert form_d.total_remaining == 5_000_000
    assert form_d.industry_group == "Technology"
    assert form_d.is_amendment is False
    assert form_d.accession_number == "0001950000-26-000003"


def test_parse_form_d_reads_related_persons():
    form_d = parse_form_d(FORM_D_XML)
    names = {p.name for p in form_d.related_persons}
    assert names == {"Ada Marsh", "Carlos Nguyen"}
    ada = next(p for p in form_d.related_persons if p.name == "Ada Marsh")
    assert "Executive Officer" in ada.relationships
    assert "Director" in ada.relationships


def test_parse_form_d_indefinite_remaining_is_none():
    xml = FORM_D_XML.replace(
        "<totalRemaining>5000000</totalRemaining>",
        "<totalRemaining>Indefinite</totalRemaining>",
    )
    form_d = parse_form_d(xml)
    assert form_d.total_remaining is None
    # The sold amount is still parsed normally.
    assert form_d.total_amount_sold == 55_000_000


def test_parse_form_d_detects_amendment():
    xml = FORM_D_XML.replace(
        "<submissionType>D</submissionType>",
        "<submissionType>D/A</submissionType>",
    )
    assert parse_form_d(xml).is_amendment is True


def test_parse_form_d_namespaced_document():
    # EDGAR has shipped the schema with a default namespace; parsing must be
    # namespace-agnostic.
    xml = FORM_D_XML.replace(
        "<edgarSubmission>",
        '<edgarSubmission xmlns="http://www.sec.gov/edgar/FormDXML">',
    )
    form_d = parse_form_d(xml)
    assert form_d.issuer_name == "Northwind Robotics Inc."
    assert form_d.total_amount_sold == 55_000_000


def test_recent_form_d_filters_and_includes_amendments():
    client = EdgarClient(_fetcher)
    filings = client.recent_form_d(1950000)
    assert len(filings) == 3
    assert {f.form for f in filings} == {"D", "D/A"}


def test_fetch_form_d_parses_primary_document():
    client = EdgarClient(_fetcher)
    filing = client.recent_form_d(1950000)[0]
    form_d = client.fetch_form_d(filing)
    assert form_d.total_amount_sold == 55_000_000
    assert form_d.accession_number == filing.accession_number


def test_recent_form_d_since_drops_older_filings():
    # The fixture has two April-2026 Form Ds and one from Sept 2025; a cutoff
    # between them keeps only the recent pair, so a live caller never fetches the
    # stale one's XML.
    client = EdgarClient(_fetcher)
    recent = client.recent_form_d(1950000, since=date(2026, 1, 1))
    assert len(recent) == 2
    assert all(f.filing_date >= date(2026, 1, 1) for f in recent)
    # Without the cutoff all three are returned (back-compatible default).
    assert len(client.recent_form_d(1950000)) == 3


def test_recent_form_d_since_keeps_undated_filing():
    # A filing with no parseable date can't be proven stale, so `since` keeps it
    # rather than dropping it. Filter a hand-built list (the `filings=` path) so
    # we can include an undated Form D.
    undated = Filing(
        cik="1950000",
        accession_number="0001950000-99-000099",
        form="D",
        filing_date=None,
        report_date=None,
        primary_document="primary_doc.xml",
    )
    dated_old = Filing(
        cik="1950000",
        accession_number="0001950000-20-000001",
        form="D",
        filing_date=date(2020, 1, 1),
        report_date=None,
        primary_document="primary_doc.xml",
    )
    result = EdgarClient(_fetcher).recent_form_d(
        1950000, filings=[undated, dated_old], since=date(2026, 1, 1)
    )
    assert undated in result  # kept despite the cutoff
    assert dated_old not in result  # genuinely stale, dropped
