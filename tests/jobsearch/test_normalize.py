"""Tests for canonicalization + cross-source de-duplication."""

from datetime import datetime, timezone

from jobfinder.jobsearch.models import RawPosting, Source
from jobfinder.jobsearch.normalize import (
    board_to_raw,
    canonicalize,
    normalize_company,
    normalize_title,
)
from jobfinder.sources.ats import JobBoard, JobPosting


def _li(title, company, *, url=None, job_id=None, location=None):
    return RawPosting(
        title=title,
        company=company,
        source=Source.LINKEDIN_ALERT,
        url=url,
        source_job_id=job_id,
        location=location,
    )


def test_normalize_title_flattens_formatting():
    assert normalize_title("VP, AI & Data") == "vp ai and data"
    assert normalize_title("VP  of AI/ML") == "vp of ai ml"


def test_normalize_company_strips_suffix():
    assert normalize_company("Acme Corp, Inc.") == "acme"
    assert normalize_company("Globex LLC") == "globex"


def test_distinct_jobs_stay_separate():
    jobs = canonicalize(
        [
            _li("VP, AI & Data", "Acme", job_id="1"),
            _li("Head of ML", "Globex", job_id="2"),
        ]
    )
    assert len(jobs) == 2


def test_soft_key_merges_linkedin_and_ats_same_role():
    li = _li("VP, AI & Data", "Acme", job_id="LI1", location="Remote")
    board = JobBoard(
        provider="greenhouse",
        token="Acme",
        url="https://boards.greenhouse.io/acme",
        postings=[
            JobPosting(
                id="GH9",
                title="VP AI and Data",
                location="Remote",
                url="https://boards.greenhouse.io/acme/jobs/GH9",
            )
        ],
    )
    jobs = canonicalize([li], [board])
    assert len(jobs) == 1
    job = jobs[0]
    # Both sources are preserved, and the ATS apply URL wins over the LI URL.
    assert set(job.source_kinds) == {"linkedin_alert", "greenhouse"}
    assert job.best_apply_url == "https://boards.greenhouse.io/acme/jobs/GH9"


def test_transitive_merge_across_hard_and_soft_keys():
    # A links to B via a shared URL hard key; C links to B via a shared
    # source_job_id; the three must collapse into ONE job (a one-pass group-id
    # lookup would leave A and C in separate groups — this needs real union-find).
    a = _li("VP AI", "Acme", url="https://x/1", location="Remote")
    b = RawPosting(
        "VP AI", "Acme", Source.GREENHOUSE, source_job_id="GH9", location="Austin"
    )
    c = RawPosting(
        "VP AI",
        "Acme",
        Source.GREENHOUSE,
        url="https://x/1",
        source_job_id="GH9",
        location="Austin",
    )
    jobs = canonicalize([a, b, c])
    assert len(jobs) == 1
    assert len(jobs[0].sources) == 3


def test_hard_key_url_merges_exact_duplicate():
    a = _li("VP AI", "Acme", url="https://x.co/jobs/1/")
    b = _li("VP AI", "Acme", url="https://x.co/jobs/1")  # trailing slash differs
    jobs = canonicalize([a, b])
    assert len(jobs) == 1


def test_empty_company_postings_do_not_false_merge():
    # Two distinct roles whose LinkedIn metadata didn't parse (empty company) and
    # which share a title+location must NOT soft-collapse into one job.
    a = RawPosting(
        "VP AI", "", Source.LINKEDIN_ALERT, source_job_id="A", location="Remote"
    )
    b = RawPosting(
        "VP AI", "", Source.LINKEDIN_ALERT, source_job_id="B", location="Remote"
    )
    assert len(canonicalize([a, b])) == 2


def test_empty_company_still_merges_via_hard_key():
    # The non-empty-company soft guard must not block a legitimate hard-key merge
    # (exact shared URL) when the company happens to be blank.
    c = RawPosting("VP AI", "", Source.LINKEDIN_ALERT, url="https://x/1")
    d = RawPosting("VP AI", "", Source.GREENHOUSE, url="https://x/1")
    assert len(canonicalize([c, d])) == 1


def test_different_titles_same_company_not_merged():
    jobs = canonicalize(
        [
            _li("VP AI", "Acme", job_id="1"),
            _li("VP Sales", "Acme", job_id="2"),
        ]
    )
    assert len(jobs) == 2


def test_board_to_raw_maps_provider_to_source():
    board = JobBoard(
        provider="lever",
        token="globex",
        url="https://jobs.lever.co/globex",
        postings=[JobPosting(id="L1", title="VP Analytics")],
    )
    raws = board_to_raw(board)
    assert raws[0].source is Source.LEVER
    assert raws[0].source_job_id == "L1"
    assert raws[0].company == "globex"


def test_most_recent_posted_at_wins_on_merge():
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 6, 1, tzinfo=timezone.utc)
    a = RawPosting(
        "VP AI", "Acme", Source.LINKEDIN_ALERT, source_job_id="1", posted_at=old
    )
    b = RawPosting("VP AI", "Acme", Source.GREENHOUSE, source_job_id="2", posted_at=new)
    jobs = canonicalize([a, b])
    assert len(jobs) == 1
    assert jobs[0].posted_at == new


def test_merge_tolerates_naive_and_aware_posted_at():
    # A LinkedIn Date header with the RFC 2822 "-0000" marker parses NAIVE; the
    # ATS timestamp is tz-aware. Merging both into one group must not crash on
    # max() comparing naive vs aware — this is the core LI↔ATS pairing.
    naive = datetime(2026, 6, 1)  # no tzinfo, as a "-0000" Date header yields
    aware = datetime(2026, 6, 2, tzinfo=timezone.utc)
    li = RawPosting(
        "VP AI", "Acme", Source.LINKEDIN_ALERT, source_job_id="LI1", posted_at=naive
    )
    board = JobBoard(
        provider="greenhouse",
        token="acme",
        url="https://boards.greenhouse.io/acme",
        postings=[JobPosting(id="GH1", title="VP AI", updated_at=aware)],
    )
    jobs = canonicalize([li], [board])
    assert len(jobs) == 1
    # Most-recent wins, and the result is tz-aware UTC.
    assert jobs[0].posted_at == aware
    assert jobs[0].posted_at.tzinfo is not None


def test_soft_key_merges_despite_remote_location_phrasing():
    # LinkedIn says "Remote (United States)"; the ATS board says just "Remote".
    # The coarse location bucket must still merge them into one job.
    li = _li("VP, AI & Data", "Acme", job_id="LI1", location="Remote (United States)")
    board = JobBoard(
        provider="greenhouse",
        token="acme",
        url="https://boards.greenhouse.io/acme",
        postings=[
            JobPosting(
                id="GH9",
                title="VP AI and Data",
                location="Remote",
                url="https://boards.greenhouse.io/acme/jobs/GH9",
            )
        ],
    )
    jobs = canonicalize([li], [board])
    assert len(jobs) == 1
    assert set(jobs[0].source_kinds) == {"linkedin_alert", "greenhouse"}


def test_merge_prefers_remote_location_over_ats_city():
    # An ATS board lists a HQ city ("Austin, TX") for a role a LinkedIn alert
    # flagged "Remote (United States)"; the merge must keep the remote location
    # (and derive workplace_type=remote) so the role isn't mis-scored on-site.
    url = "https://boards.greenhouse.io/acme/jobs/GH9"
    li = _li("VP, AI & Data", "Acme", url=url, location="Remote (United States)")
    board = JobBoard(
        provider="greenhouse",
        token="acme",
        url="https://boards.greenhouse.io/acme",
        postings=[
            JobPosting(id="GH9", title="VP AI and Data", location="Austin, TX", url=url)
        ],
    )
    jobs = canonicalize([li], [board])
    assert len(jobs) == 1
    assert jobs[0].location == "Remote (United States)"
    assert jobs[0].workplace_type == "remote"


def test_merge_keeps_ats_city_when_no_source_is_remote():
    # When neither source is remote, the ATS-first location is retained.
    url = "https://boards.greenhouse.io/acme/jobs/GH9"
    li = _li("VP AI", "Acme", url=url, location="New York")
    board = JobBoard(
        provider="greenhouse",
        token="acme",
        url="https://boards.greenhouse.io/acme",
        postings=[JobPosting(id="GH9", title="VP AI", location="Austin, TX", url=url)],
    )
    jobs = canonicalize([li], [board])
    assert len(jobs) == 1
    assert jobs[0].location == "Austin, TX"


def test_merge_prefers_linkedin_display_name_over_ats_slug():
    # board_to_raw puts the ATS slug ("stripe") in company; the merged job should
    # display the LinkedIn alert's real name ("Stripe, Inc."), not the slug.
    li = _li("VP, AI & Data", "Stripe, Inc.", job_id="LI1", location="Remote")
    board = JobBoard(
        provider="greenhouse",
        token="stripe",
        url="https://boards.greenhouse.io/stripe",
        postings=[
            JobPosting(
                id="GH9",
                title="VP AI and Data",
                location="Remote",
                url="https://boards.greenhouse.io/stripe/jobs/GH9",
            )
        ],
    )
    jobs = canonicalize([li], [board])
    assert len(jobs) == 1
    assert jobs[0].company == "Stripe, Inc."
    # ATS apply URL still wins for the application link.
    assert jobs[0].best_apply_url == "https://boards.greenhouse.io/stripe/jobs/GH9"
