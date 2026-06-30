"""Tests for the LinkedIn job-alert email parser. Pure/offline against fixtures."""

from pathlib import Path

from jobfinder.jobsearch.models import Source
from jobfinder.jobsearch.sources.linkedin_email import parse_alert_email

FIXTURES = Path(__file__).parent / "fixtures"


def _eml(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_multi_job_html_alert_parses_each_job():
    postings = parse_alert_email(_eml("li_alert_multi.eml"))
    # Three distinct jobs (the duplicate "View job" anchors collapse).
    assert len(postings) == 3
    titles = [p.title for p in postings]
    assert "VP, AI & Data" in titles
    assert "Head of Machine Learning" in titles
    assert "Senior Machine Learning Engineer" in titles


def test_company_and_location_split_from_middot():
    postings = parse_alert_email(_eml("li_alert_multi.eml"))
    vp = next(p for p in postings if p.title == "VP, AI & Data")
    assert vp.company == "Acme Corp"
    assert vp.location == "Remote (United States)"
    assert vp.workplace_type == "remote"
    assert vp.source is Source.LINKEDIN_ALERT


def test_job_id_and_url_captured_not_fetched():
    postings = parse_alert_email(_eml("li_alert_multi.eml"))
    vp = next(p for p in postings if p.title == "VP, AI & Data")
    assert vp.source_job_id == "3811111111"
    assert vp.url is not None and "jobs/view/3811111111" in vp.url


def test_alert_keyword_from_quoted_subject():
    postings = parse_alert_email(_eml("li_alert_multi.eml"))
    assert all(p.alert_keyword == "VP of AI" for p in postings)


def test_alert_keyword_from_job_alert_for_subject():
    postings = parse_alert_email(_eml("li_alert_single.eml"))
    assert postings[0].alert_keyword == "vp of ai & analytics"


def test_date_header_becomes_posted_at():
    postings = parse_alert_email(_eml("li_alert_single.eml"))
    assert postings[0].posted_at is not None
    assert postings[0].posted_at.year == 2026


def test_single_job_html_alert():
    postings = parse_alert_email(_eml("li_alert_single.eml"))
    assert len(postings) == 1
    assert postings[0].title == "VP of AI & Analytics"
    assert postings[0].company == "Umbrella Inc"


def test_plaintext_alert_fallback():
    postings = parse_alert_email(_eml("li_alert_plaintext.eml"))
    titles = [p.title for p in postings]
    assert "Head of AI" in titles
    assert "VP Data Science" in titles
    head = next(p for p in postings if p.title == "Head of AI")
    assert head.company == "Stark Industries"
    assert head.location == "New York, NY"


def test_view_job_anchor_before_title_keeps_descriptive_title():
    # Same job id linked by a generic "View job" button BEFORE the descriptive
    # title anchor: the stored title must be the descriptive one, not "View job".
    from jobfinder.jobsearch.sources.linkedin_email import _assemble

    tokens = [
        ("job", "999", "https://www.linkedin.com/jobs/view/999/?trk=a", "View job"),
        ("text", "Acme Corp"),
        ("text", "Remote"),
        (
            "job",
            "999",
            "https://www.linkedin.com/jobs/view/999/?trk=b",
            "VP, AI & Data",
        ),
    ]
    postings = _assemble(tokens, keyword=None, posted_at=None)
    assert len(postings) == 1
    assert postings[0].title == "VP, AI & Data"
    assert postings[0].company == "Acme Corp"


def test_html_without_anchors_falls_back_to_plaintext():
    # A multipart alert whose HTML part carries no /jobs/view/ anchors (e.g. an
    # "open in browser" stub) must fall back to the text/plain alternative, which
    # often lists the same jobs.
    postings = parse_alert_email(_eml("li_alert_html_no_anchors.eml"))
    titles = [p.title for p in postings]
    assert "VP of AI" in titles
    assert "Head of Data Science" in titles
    hooli = next(p for p in postings if p.title == "VP of AI")
    assert hooli.company == "Hooli"


def test_non_job_email_yields_nothing():
    assert parse_alert_email(_eml("not_a_job.eml")) == []


def test_empty_input_yields_nothing():
    assert parse_alert_email("") == []


def test_accepts_str_and_bytes():
    as_bytes = parse_alert_email(_eml("li_alert_single.eml"))
    as_str = parse_alert_email((FIXTURES / "li_alert_single.eml").read_text())
    assert [p.title for p in as_bytes] == [p.title for p in as_str]
