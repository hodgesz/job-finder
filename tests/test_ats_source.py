"""Tests for the ATS job-board client. Network is injected, so these run offline."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jobfinder.sources.ats import (
    PROVIDERS,
    AtsClient,
    default_fetcher,
    parse_ashby,
    parse_greenhouse,
    parse_lever,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fetcher(filename: str):
    def fetch(_url: str) -> str:
        return (FIXTURES / filename).read_text()

    return fetch


def test_parse_greenhouse_normalizes_postings():
    board = parse_greenhouse(
        (FIXTURES / "greenhouse_jobs.json").read_text(), token="acme"
    )
    assert board.provider == "greenhouse"
    assert board.token == "acme"
    assert board.url == "https://boards.greenhouse.io/acme"
    assert len(board.postings) == 5
    first = board.postings[0]
    assert first.id == "5501001"
    assert first.title == "Senior Software Engineer, Payments"
    assert first.department == "Engineering"
    assert first.location == "San Francisco, CA"
    assert first.url.endswith("/5501001")
    # Offset-aware timestamp normalized to UTC.
    assert first.updated_at == datetime(2026, 6, 20, 18, 3, tzinfo=timezone.utc)


def test_parse_lever_reads_categories_and_epoch_millis():
    board = parse_lever((FIXTURES / "lever_postings.json").read_text(), token="acme")
    assert board.provider == "lever"
    assert len(board.postings) == 3
    head = board.postings[1]
    assert head.title == "Head of Design"
    assert head.department == "Design"
    assert head.team == "Design Leadership"
    assert head.commitment == "Full-time"
    # createdAt is epoch milliseconds -> tz-aware UTC.
    assert head.updated_at is not None
    assert head.updated_at.tzinfo is not None
    assert head.updated_at.year == 2026


def test_parse_ashby_reads_flat_fields():
    board = parse_ashby((FIXTURES / "ashby_jobs.json").read_text(), token="acme")
    assert board.provider == "ashby"
    assert len(board.postings) == 3
    vp = board.postings[2]
    assert vp.title == "VP of Engineering"
    assert vp.department == "Engineering"
    assert vp.commitment == "FullTime"
    assert vp.updated_at == datetime(2026, 6, 10, 18, 0, tzinfo=timezone.utc)


def test_client_dispatches_by_provider():
    client = AtsClient(_fetcher("greenhouse_jobs.json"))
    board = client.fetch_board("greenhouse", "acme")
    assert board.provider == "greenhouse"
    assert len(board.postings) == 5


def test_client_is_case_insensitive_on_provider():
    client = AtsClient(_fetcher("lever_postings.json"))
    assert client.fetch_board("LEVER", "acme").provider == "lever"


def test_client_rejects_unknown_provider():
    client = AtsClient(_fetcher("greenhouse_jobs.json"))
    with pytest.raises(ValueError, match="Unknown ATS provider"):
        client.fetch_board("workday", "acme")


def test_default_fetcher_requires_user_agent():
    with pytest.raises(ValueError, match="User-Agent"):
        default_fetcher("")
    with pytest.raises(ValueError, match="User-Agent"):
        default_fetcher("   ")
    # A non-empty UA is accepted (no contact-email requirement, unlike SEC).
    assert callable(default_fetcher("job-finder research"))


def test_missing_and_placeholder_fields_become_none():
    payload = (
        '{"jobs": [{"id": 1, "title": "Engineer", '
        '"departments": [{"name": "No Department"}], "location": {}}]}'
    )
    board = parse_greenhouse(payload, token="x")
    posting = board.postings[0]
    assert posting.department is None
    assert posting.location is None
    assert posting.updated_at is None


def test_every_declared_provider_is_dispatchable():
    # Each provider in PROVIDERS must resolve to a parser (no KeyError). We feed
    # each its own correctly-shaped fixture.
    fixtures = {
        "greenhouse": "greenhouse_jobs.json",
        "lever": "lever_postings.json",
        "ashby": "ashby_jobs.json",
    }
    assert set(fixtures) == set(PROVIDERS)
    for provider, fixture in fixtures.items():
        board = AtsClient(_fetcher(fixture)).fetch_board(provider, "acme")
        assert board.provider == provider
        assert board.postings
