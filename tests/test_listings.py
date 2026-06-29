"""Tests for listed-roles corroboration (jobfinder.listings).

Pure/offline: builds JobBoards and asserts the corroboration join is correct,
ordered, and capped — no network, injected `now`. The `target_persona` passed in
is the *authoritative* (signal-derived) function; the CLI computes it (and passes
None for a default-fallback persona) — see test_cli for that integration.
"""

from datetime import datetime, timedelta, timezone

from jobfinder.listings import (
    CorroboratingRole,
    RoleCorroboration,
    corroborate_roles,
    corroboration_lines,
)
from jobfinder.sources.ats import JobBoard, JobPosting

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _posting(
    pid: str,
    title: str,
    *,
    department: str | None = None,
    team: str | None = None,
    location: str | None = None,
    days_ago: int | None = 5,
    url: str | None = None,
) -> JobPosting:
    updated = None if days_ago is None else NOW - timedelta(days=days_ago)
    return JobPosting(
        id=pid,
        title=title,
        department=department,
        team=team,
        location=location,
        updated_at=updated,
        url=url or f"https://jobs.example.com/{pid}",
    )


def _board(postings: list[JobPosting], *, provider: str = "greenhouse") -> JobBoard:
    return JobBoard(
        provider=provider,
        token="acme",
        url=f"https://boards.{provider}.io/acme",
        postings=postings,
    )


def test_empty_when_no_boards():
    corro = corroborate_roles([], target_persona="CFO / VP Finance", now=NOW)
    assert corro.total == 0
    assert not corro.has_roles
    assert corro.sample == []
    # Nothing renders for a pure-SEC opportunity.
    assert corroboration_lines(corro) == []


def test_counts_total_recent_and_in_function():
    board = _board(
        [
            _posting("1", "Senior Accountant", department="Finance", days_ago=3),
            _posting("2", "FP&A Manager", department="Finance", days_ago=200),  # stale
            _posting("3", "Account Executive", department="Sales", days_ago=2),
        ]
    )
    corro = corroborate_roles([board], target_persona="CFO / VP Finance", now=NOW)
    assert corro.total == 3
    assert corro.recent == 2  # the 200-day-old req is not recent
    assert corro.in_function == 2  # both Finance reqs match the CFO persona


def test_none_target_flags_nothing_in_function():
    # The default-fallback guard: a funding-only opportunity passes target=None,
    # so even a board full of Finance reqs is NOT manufactured as corroboration.
    board = _board(
        [
            _posting("1", "Senior Accountant", department="Finance"),
            _posting("2", "Controller", department="Finance"),
        ]
    )
    corro = corroborate_roles([board], target_persona=None, now=NOW)
    assert corro.total == 2
    assert corro.recent == 2  # still counted/active
    assert corro.in_function == 0  # but never flagged in-function
    assert all(not r.in_function for r in corro.sample)


def test_in_function_matches_department_first_like_scorer():
    # A "Revenue Operations Lead" req sitting in a Finance department reads as
    # finance (CFO) by its DEPARTMENT, not sales by its title — the persona is
    # matched department-first, exactly as scoring._persona_fragments orders ATS
    # fragments. This keeps a posting that *drove* a department-surge opportunity
    # flagged in-function for that opportunity rather than diverging on its title.
    board = _board([_posting("1", "Revenue Operations Lead", department="Finance")])
    corro = corroborate_roles([board], target_persona="CFO / VP Finance", now=NOW)
    assert corro.in_function == 1
    corro_sales = corroborate_roles([board], target_persona="CRO / VP Sales", now=NOW)
    assert corro_sales.in_function == 0


def test_in_function_falls_back_to_title_when_no_department():
    # With no department/team, the title is the only fragment, so a "Revenue
    # Operations Lead" with no department reads as sales (CRO) by title.
    board = _board([_posting("1", "Revenue Operations Lead")])
    corro = corroborate_roles([board], target_persona="CRO / VP Sales", now=NOW)
    assert corro.in_function == 1


def test_sample_orders_in_function_then_recent_then_title():
    board = _board(
        [
            _posting(
                "z", "Zebra Analyst", department="Sales", days_ago=1
            ),  # off, recent
            _posting(
                "a", "Accountant", department="Finance", days_ago=300
            ),  # in-fn, stale
            _posting(
                "b", "Budget Analyst", department="Finance", days_ago=2
            ),  # in-fn, recent
            _posting(
                "c", "Comptroller Finance", department="Finance", days_ago=2
            ),  # in-fn, recent
        ]
    )
    corro = corroborate_roles(
        [board], target_persona="CFO / VP Finance", now=NOW, limit=10
    )
    titles = [r.title for r in corro.sample]
    # In-function first; within in-function, recent ahead of stale; ties by title.
    assert titles[0] == "Budget Analyst"
    assert titles[1] == "Comptroller Finance"
    assert titles[2] == "Accountant"
    assert titles[3] == "Zebra Analyst"  # off-function last


def test_sample_is_capped_at_limit():
    board = _board(
        [_posting(str(i), f"Finance Role {i}", department="Finance") for i in range(10)]
    )
    corro = corroborate_roles(
        [board], target_persona="CFO / VP Finance", now=NOW, limit=3
    )
    assert corro.total == 10
    assert len(corro.sample) == 3


def test_location_does_not_match_persona():
    # A posting whose only "Finance"-ish word is a place must not read in-function;
    # location is excluded from persona matching.
    board = _board([_posting("1", "Office Manager", location="Finance District, NY")])
    corro = corroborate_roles([board], target_persona="CFO / VP Finance", now=NOW)
    assert corro.in_function == 0
    # But location is still a usable "where" label when nothing better exists.
    assert corro.sample[0].where == "Finance District, NY"


def test_multiple_boards_are_aggregated():
    gh = _board(
        [_posting("1", "Accountant", department="Finance")], provider="greenhouse"
    )
    lever = _board(
        [_posting("2", "Controller", department="Finance")], provider="lever"
    )
    corro = corroborate_roles([gh, lever], target_persona="CFO / VP Finance", now=NOW)
    assert corro.total == 2
    assert len(corro.board_urls) == 2
    assert "https://boards.greenhouse.io/acme" in corro.board_urls
    assert "https://boards.lever.io/acme" in corro.board_urls


def test_undated_postings_count_but_are_not_recent():
    board = _board([_posting("1", "Accountant", department="Finance", days_ago=None)])
    corro = corroborate_roles([board], target_persona="CFO / VP Finance", now=NOW)
    assert corro.total == 1
    assert corro.recent == 0


def test_lines_render_headline_tags_and_board():
    role = CorroboratingRole(
        title="Controller",
        where="Finance",
        url="https://jobs.example.com/1",
        in_function=True,
        recent=True,
    )
    corro = RoleCorroboration(
        total=1,
        recent=1,
        in_function=1,
        sample=[role],
        board_urls=["https://boards.greenhouse.io/acme"],
    )
    lines = corroboration_lines(corro)
    text = "\n".join(lines)
    assert "Listed roles: 1 live (1 active in the last 30d, 1 in-function)" in text
    assert "Controller — Finance [in-function, recent]" in text
    assert "https://jobs.example.com/1" in text
    assert "Board: https://boards.greenhouse.io/acme" in text
