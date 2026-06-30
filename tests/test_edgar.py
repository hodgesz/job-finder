"""Tests for the EDGAR client. Network is injected, so these run offline."""

from datetime import date
from pathlib import Path

import pytest

from jobfinder.sources.edgar import (
    EdgarClient,
    Filing,
    default_fetcher,
    parse_company_info,
    parse_submissions,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _submissions_fetcher(_url: str) -> str:
    return (FIXTURES / "submissions_sample.json").read_text()


def test_parse_submissions_reads_parallel_arrays():
    filings = parse_submissions((FIXTURES / "submissions_sample.json").read_text())
    assert len(filings) == 4
    first = filings[0]
    assert first.form == "8-K"
    assert first.accession_number == "0001140361-26-015711"
    assert first.filing_date == date(2026, 4, 20)
    assert first.report_date == date(2026, 4, 17)
    assert first.items == ["5.02"]


def test_filing_url_construction():
    filings = parse_submissions((FIXTURES / "submissions_sample.json").read_text())
    f = filings[0]
    assert f.accession_nodash == "000114036126015711"
    assert f.primary_document_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000114036126015711/ef20071035_8k.htm"
    )


def test_recent_8k_filters_by_item_without_fetching_docs():
    client = EdgarClient(_submissions_fetcher)
    all_8k = client.recent_8k(320193)
    assert len(all_8k) == 3  # two 8-K + one 8-K/A; excludes the 10-Q
    only_502 = client.recent_8k(320193, item="5.02")
    # The original 8-K and its 8-K/A amendment both disclose Item 5.02.
    assert {f.accession_number for f in only_502} == {
        "0001140361-26-015711",
        "0001140361-26-016000",
    }


def test_recent_8k_includes_amendments():
    client = EdgarClient(_submissions_fetcher)
    forms = {f.form for f in client.recent_8k(320193)}
    assert "8-K/A" in forms


def test_recent_8k_accepts_padded_and_prefixed_cik():
    client = EdgarClient(_submissions_fetcher)
    assert client.recent_8k("CIK0000320193", item="5.02")


def test_form_d_primary_data_url_strips_xsl_viewer_dir():
    # SEC advertises a Form D's primaryDocument as the XSL-styled HTML viewer
    # (xslFormDX08/primary_doc.xml); the raw XML our parser needs is the same
    # path with that styling directory removed.
    styled = Filing(
        cik="789019",
        accession_number="0001137638-20-000002",
        form="D",
        filing_date=date(2020, 1, 1),
        report_date=None,
        primary_document="xslFormDX08/primary_doc.xml",
    )
    assert styled.primary_data_url == (
        "https://www.sec.gov/Archives/edgar/data/789019/"
        "000113763820000002/primary_doc.xml"
    )
    # A plain document path is unchanged — primary_data_url == primary_document_url.
    plain = Filing(
        cik="320193",
        accession_number="0001140361-26-015711",
        form="8-K",
        filing_date=date(2026, 4, 20),
        report_date=None,
        primary_document="ef20071035_8k.htm",
    )
    assert plain.primary_data_url == plain.primary_document_url
    # Only a SOLE leading "xsl.../" segment is stripped: a genuine subdirectory
    # (even one nesting the viewer) is left intact rather than mis-rewritten.
    nested = Filing(
        cik="1",
        accession_number="0000000000-00-000000",
        form="D",
        filing_date=date(2020, 1, 1),
        report_date=None,
        primary_document="sub/xslFormDX08/primary_doc.xml",
    )
    assert nested.primary_data_url.endswith("/sub/xslFormDX08/primary_doc.xml")


def test_fetch_form_d_uses_raw_xml_url():
    # Reproduce the real-CIK bug: the styled viewer URL serves rendered HTML
    # (which crashes the XML parser), while the raw URL serves the XML. The
    # client must fetch the raw URL — so this fetcher returns HTML for the styled
    # path and would blow up parse_form_d if fetch_form_d picked the wrong one.
    fetched: list[str] = []

    filing = Filing(
        cik="789019",
        accession_number="0001137638-20-000002",
        form="D",
        filing_date=date(2020, 1, 1),
        report_date=None,
        primary_document="xslFormDX08/primary_doc.xml",
    )

    def fetcher(url: str) -> str:
        fetched.append(url)
        if "xslFormDX08" in url:
            return "<!DOCTYPE html><html><body>styled viewer</body></html>"
        return (FIXTURES / "form_d_sample.xml").read_text()

    form_d = EdgarClient(fetcher).fetch_form_d(filing)
    # It fetched the raw URL (not the styled viewer) and parsed real XML.
    assert fetched == [filing.primary_data_url]
    assert "xslFormDX08" not in fetched[0]
    assert form_d.issuer_name  # parsing succeeded on the XML, not the HTML


def test_parse_company_info_reads_sic_sector():
    info = parse_company_info((FIXTURES / "submissions_sample.json").read_text())
    assert info.cik == "0000320193"
    assert info.name == "Apple Inc."
    assert info.sic == "3571"
    assert info.sic_description == "Electronic Computers"
    # The header lists the stock exchange(s) the filer trades on -> a free,
    # honest "is exchange-listed" fact (Slice 17 derives a "public" stage from it).
    assert info.exchanges == ("Nasdaq",)


def test_parse_company_info_exchanges_default_and_blanks():
    # Missing or blank-only exchanges become an empty tuple (a private/unlisted
    # filer), so callers never read a stale or empty entry as a listing.
    no_exch = parse_company_info({"cik": "1", "name": "Priv Co"})
    assert no_exch.exchanges == ()
    blanks = parse_company_info(
        {"cik": "1", "name": "X", "exchanges": ["", "  ", "NYSE"]}
    )
    assert blanks.exchanges == ("NYSE",)


def test_parse_company_info_exchanges_tolerates_malformed_payload():
    # The submissions payload is external SEC JSON; guard its shape. A scalar
    # string must NOT be iterated character-by-character into bogus single-letter
    # exchanges, and a non-string element must not crash on `.strip()`.
    scalar = parse_company_info({"cik": "1", "name": "X", "exchanges": "Nasdaq"})
    assert scalar.exchanges == ()  # not ('N','a','s','d','a','q')
    mixed = parse_company_info(
        {"cik": "1", "name": "X", "exchanges": ["Nasdaq", 123, None]}
    )
    assert mixed.exchanges == ("Nasdaq",)  # non-strings skipped, no crash


def test_parse_company_info_missing_sic_is_none():
    # A filer with no assigned SIC (some private/foreign filers): blank fields
    # become None rather than empty strings, so the sector reads as unknown.
    info = parse_company_info(
        {"cik": "1", "name": " Acme Co ", "sic": "", "sicDescription": ""}
    )
    assert info.name == "Acme Co"
    assert info.sic is None
    assert info.sic_description is None


def test_company_info_reads_same_submissions_index():
    client = EdgarClient(_submissions_fetcher)
    info = client.company_info(320193)
    assert info.name == "Apple Inc."
    assert info.sic_description == "Electronic Computers"


def test_company_submissions_fetches_index_once():
    # company_info + recent_8k + recent_form_d all read the same submissions URL;
    # company_submissions fetches it once and the filings list is reused, so a
    # live CIK makes a single request to SEC's rate-limited endpoint.
    calls: list[str] = []

    def counting_fetcher(url: str) -> str:
        calls.append(url)
        return (FIXTURES / "submissions_sample.json").read_text()

    client = EdgarClient(counting_fetcher)
    info, filings = client.company_submissions(320193)
    eight_k = client.recent_8k(320193, item="5.02", filings=filings)
    form_d = client.recent_form_d(320193, filings=filings)

    assert len(calls) == 1
    assert info.name == "Apple Inc."
    assert len(eight_k) == 2  # the 8-K and its 8-K/A, both Item 5.02
    assert form_d == []  # this fixture has no Form D filings


def test_recent_8k_with_provided_filings_matches_a_fresh_fetch():
    # Passing a pre-fetched filings list filters identically to re-fetching.
    client = EdgarClient(_submissions_fetcher)
    _, filings = client.company_submissions(320193)
    assert client.recent_8k(320193, item="5.02", filings=filings) == client.recent_8k(
        320193, item="5.02"
    )


def test_default_fetcher_requires_contact_user_agent():
    with pytest.raises(ValueError, match="User-Agent"):
        default_fetcher("no-email-here")
    # A contact UA is accepted (builds the fetcher; no network call made here).
    assert callable(default_fetcher("job-finder research jane@example.com"))
