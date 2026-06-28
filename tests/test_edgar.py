"""Tests for the EDGAR client. Network is injected, so these run offline."""

from datetime import date
from pathlib import Path

import pytest

from jobfinder.sources.edgar import EdgarClient, default_fetcher, parse_submissions

FIXTURES = Path(__file__).parent / "fixtures"


def _submissions_fetcher(_url: str) -> str:
    return (FIXTURES / "submissions_sample.json").read_text()


def test_parse_submissions_reads_parallel_arrays():
    filings = parse_submissions((FIXTURES / "submissions_sample.json").read_text())
    assert len(filings) == 3
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
    assert len(all_8k) == 2  # excludes the 10-Q
    only_502 = client.recent_8k(320193, item="5.02")
    assert len(only_502) == 1
    assert only_502[0].accession_number == "0001140361-26-015711"


def test_recent_8k_accepts_padded_and_prefixed_cik():
    client = EdgarClient(_submissions_fetcher)
    assert client.recent_8k("CIK0000320193", item="5.02")


def test_default_fetcher_requires_contact_user_agent():
    with pytest.raises(ValueError, match="User-Agent"):
        default_fetcher("no-email-here")
    # A contact UA is accepted (builds the fetcher; no network call made here).
    assert callable(default_fetcher("job-finder research jane@example.com"))
