"""Tests for ATS hiring-pattern signal extraction (deterministic, offline)."""

from datetime import datetime, timedelta, timezone

from jobfinder.signals.ats_hiring import (
    MIN_RECENT_OPENINGS,
    signals_from_board,
)
from jobfinder.sources.ats import JobBoard, JobPosting

NOW = datetime(2026, 6, 28, tzinfo=timezone.utc)


def _posting(pid, title, *, department=None, team=None, days_ago=5) -> JobPosting:
    return JobPosting(
        id=pid,
        title=title,
        department=department,
        team=team,
        updated_at=NOW - timedelta(days=days_ago) if days_ago is not None else None,
        url=f"https://jobs.example.com/{pid}",
    )


def _board(postings, *, provider="greenhouse", token="acme") -> JobBoard:
    return JobBoard(
        provider=provider,
        token=token,
        url=f"https://boards.{provider}.io/{token}",
        postings=postings,
    )


def _by_type(signals):
    out = {}
    for s in signals:
        out.setdefault(s.signal_type, []).append(s)
    return out


def test_velocity_signal_fires_above_threshold_with_evidence():
    board = _board(
        [_posting(f"e{i}", f"Engineer {i}", department="Engineering") for i in range(6)]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    velocity = _by_type(signals)["ats_hiring_velocity"]
    assert len(velocity) == 1
    sig = velocity[0]
    # Citation rule holds (schema enforces >=1 evidence).
    assert sig.evidence[0].source == "greenhouse"
    assert sig.evidence[0].locator == "greenhouse:acme"
    assert sig.extracted_facts["recent_openings"] == 6
    assert 0.0 < sig.strength <= 1.0


def test_below_min_openings_yields_no_velocity_signal():
    board = _board(
        [
            _posting(f"e{i}", f"Engineer {i}", department="Engineering")
            for i in range(MIN_RECENT_OPENINGS - 1)
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "ats_hiring_velocity" not in _by_type(signals)


def test_stale_postings_are_not_counted_as_recent():
    # All openings are well outside the 30-day window -> no velocity signal.
    board = _board(
        [
            _posting(f"e{i}", f"Engineer {i}", department="Engineering", days_ago=120)
            for i in range(6)
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "ats_hiring_velocity" not in _by_type(signals)


def test_postings_without_timestamp_are_not_recent():
    board = _board(
        [
            _posting(f"e{i}", f"Engineer {i}", department="Engineering", days_ago=None)
            for i in range(6)
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "ats_hiring_velocity" not in _by_type(signals)


def test_department_surge_fires_for_concentrated_hiring():
    board = _board(
        [
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
            _posting("e3", "Platform Engineer", department="Engineering"),
            _posting("e4", "ML Engineer", department="Engineering"),
            _posting("s1", "Account Executive", department="Sales"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    surges = _by_type(signals)["department_surge"]
    # Engineering surges (4 reqs); Sales (1) does not.
    assert len(surges) == 1
    surge = surges[0]
    assert surge.extracted_facts["department"] == "Engineering"
    assert surge.extracted_facts["department_openings"] == 4


def test_greenfield_explicit_language_fires():
    board = _board(
        [
            _posting("g1", "Founding Engineer", department="Engineering"),
            _posting("e2", "Backend Engineer", department="Engineering"),
            _posting("e3", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    greenfield = _by_type(signals)["greenfield_team"]
    assert len(greenfield) == 1
    g = greenfield[0]
    assert g.extracted_facts["explicit_language"] is True
    assert g.strength == 0.8
    # Cites the specific posting URL, not just the board.
    assert g.evidence[0].url == "https://jobs.example.com/g1"


def test_greenfield_lone_leadership_req_fires():
    # A solitary leadership req in a department with no team beneath it.
    board = _board(
        [
            _posting("d1", "Head of Data", department="Data"),
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    greenfield = _by_type(signals)["greenfield_team"]
    assert len(greenfield) == 1
    g = greenfield[0]
    assert g.extracted_facts["explicit_language"] is False
    assert g.extracted_facts["department"] == "Data"
    assert g.strength == 0.6


def test_lone_head_of_support_function_is_not_greenfield():
    # Live finding D: "Head of Learning & Quality" is a routine support/ops org,
    # not a zero-to-one team — the lone-leadership heuristic must NOT fire even
    # though it is a solitary "Head of …" req with no team beneath it.
    board = _board(
        [
            _posting("s1", "Head of Learning & Quality", department="Delivery Center"),
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "greenfield_team" not in _by_type(signals)


def test_lone_head_of_it_support_title_is_not_greenfield():
    # A support function named in the TITLE (IT/service desk) is routine structure.
    board = _board(
        [
            _posting("s1", "Head of Service Desk", department="IT Operations"),
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "greenfield_team" not in _by_type(signals)


def test_support_word_in_department_does_not_suppress_core_title():
    # The denylist is matched against the TITLE only: a support word in the
    # DEPARTMENT label (e.g. "Workplace") must NOT veto a genuine core-function
    # leadership title — otherwise we trade finding D for dropping real seats.
    board = _board(
        [
            _posting("p1", "Head of Engineering", department="Workplace"),
            _posting("d1", "Data Analyst", department="Data"),
            _posting("d2", "Data Engineer", department="Data"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    greenfield = _by_type(signals)["greenfield_team"]
    assert len(greenfield) == 1
    assert greenfield[0].extracted_facts["department"] == "Workplace"


def test_reversed_l_and_d_phrasing_is_not_greenfield():
    # Bugbot #1: the L&D denylist must catch the support function in EITHER word
    # order — "Quality & Learning" is the same routine org as "Learning & Quality"
    # and must also be suppressed (a bare "quality" token was deliberately not
    # added, so order-independent L&D phrasing is what catches the reversed form).
    board = _board(
        [
            _posting("s1", "Head of Quality & Learning", department="Delivery Center"),
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "greenfield_team" not in _by_type(signals)


def test_lone_head_of_core_function_still_greenfield():
    # The denylist is permissive: a lone "Head of …" in a genuine product/GTM
    # function not on the support denylist still reads as a forming seat. (Data
    # is covered by test_greenfield_lone_leadership_req_fires; check another.)
    board = _board(
        [
            _posting("p1", "Head of Growth", department="Growth"),
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    greenfield = _by_type(signals)["greenfield_team"]
    assert len(greenfield) == 1
    assert greenfield[0].extracted_facts["department"] == "Growth"


def test_core_technical_titles_with_collision_words_still_greenfield():
    # Tokens were tightened so genuine technical/GTM seats whose titles merely
    # CONTAIN a support-ish word are NOT suppressed: "Machine Learning" (not L&D),
    # "Data Quality", "Developer Support" all read as forming functions.
    for title, dept in [
        ("Head of Machine Learning", "AI"),
        ("Head of AI & Machine Learning", "AI"),
        ("Head of Data Quality", "Data Platform"),
        ("Head of Developer Support", "Engineering"),
    ]:
        board = _board(
            [
                _posting("p1", title, department=dept),
                _posting("o1", "Backend Engineer", department="Other"),
                _posting("o2", "Frontend Engineer", department="Other"),
            ]
        )
        signals = signals_from_board(board, company_id="co-x", now=NOW)
        greenfield = _by_type(signals).get("greenfield_team", [])
        assert len(greenfield) == 1, f"{title!r} should fire greenfield"
        assert greenfield[0].extracted_facts["department"] == dept


def test_explicit_language_fires_even_for_support_function():
    # The denylist gates ONLY the lone-leadership heuristic. An EXPLICIT
    # founding/first-hire title in a support function still fires — explicit
    # zero-to-one language is authoritative regardless of the function.
    board = _board(
        [
            _posting("s1", "Founding Quality Engineer", department="Quality"),
            _posting("e1", "Backend Engineer", department="Engineering"),
            _posting("e2", "Frontend Engineer", department="Engineering"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    greenfield = _by_type(signals)["greenfield_team"]
    assert len(greenfield) == 1
    assert greenfield[0].extracted_facts["explicit_language"] is True


def test_leadership_req_with_a_team_is_not_greenfield():
    # Head of Data PLUS data ICs -> the function already has a team, not greenfield.
    board = _board(
        [
            _posting("d1", "Head of Data", department="Data"),
            _posting("d2", "Data Engineer", department="Data"),
            _posting("d3", "Data Analyst", department="Data"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "greenfield_team" not in _by_type(signals)


def test_empty_board_yields_no_signals():
    assert signals_from_board(_board([]), company_id="co-x", now=NOW) == []


def test_observed_at_defaults_now_for_recency_reference():
    # When `now` is omitted, `observed_at` is the recency reference.
    board = _board(
        [_posting(f"e{i}", f"Engineer {i}", department="Engineering") for i in range(6)]
    )
    signals = signals_from_board(board, company_id="co-x", observed_at=NOW)
    assert "ats_hiring_velocity" in _by_type(signals)


def test_far_future_timestamps_are_not_recent():
    # A misparsed/garbage far-future date must not be counted as recent and
    # silently inflate the velocity count.
    board = _board(
        [
            _posting(f"e{i}", f"Engineer {i}", department="Engineering", days_ago=-400)
            for i in range(6)
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "ats_hiring_velocity" not in _by_type(signals)


def test_small_clock_skew_still_counts_as_recent():
    # A posting dated a few hours in the future (clock skew) still counts.
    board = _board(
        [
            JobPosting(
                id=f"e{i}",
                title=f"Engineer {i}",
                department="Engineering",
                updated_at=NOW + timedelta(hours=6),
            )
            for i in range(6)
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    assert "ats_hiring_velocity" in _by_type(signals)


def test_slugifiable_collisions_get_distinct_signal_ids():
    # Two distinct departments that slugify identically must NOT share a Signal
    # id, or the store's id-keyed upsert would silently drop one surge.
    board = _board(
        [
            _posting("a1", "Eng A1", department="Sales/Ops"),
            _posting("a2", "Eng A2", department="Sales/Ops"),
            _posting("a3", "Eng A3", department="Sales/Ops"),
            _posting("b1", "Eng B1", department="Sales-Ops"),
            _posting("b2", "Eng B2", department="Sales-Ops"),
            _posting("b3", "Eng B3", department="Sales-Ops"),
        ]
    )
    surges = _by_type(signals_from_board(board, company_id="co-x", now=NOW))[
        "department_surge"
    ]
    ids = [s.id for s in surges]
    assert len(ids) == len(set(ids)) == 2


def test_non_ascii_departments_get_distinct_signal_ids():
    # All-non-ASCII names previously collapsed to the same slug ("unknown").
    board = _board(
        [
            _posting("a1", "Eng A1", department="技術"),
            _posting("a2", "Eng A2", department="技術"),
            _posting("a3", "Eng A3", department="技術"),
            _posting("b1", "Eng B1", department="технология"),
            _posting("b2", "Eng B2", department="технология"),
            _posting("b3", "Eng B3", department="технология"),
        ]
    )
    surges = _by_type(signals_from_board(board, company_id="co-x", now=NOW))[
        "department_surge"
    ]
    ids = [s.id for s in surges]
    assert len(ids) == len(set(ids)) == 2


def test_signal_ids_are_stable_across_runs():
    # The id-hash must be deterministic so the idempotent upsert re-saves the
    # same row rather than duplicating.
    board = _board(
        [_posting(f"e{i}", f"Engineer {i}", department="Engineering") for i in range(4)]
    )
    first = {s.id for s in signals_from_board(board, company_id="co-x", now=NOW)}
    second = {s.id for s in signals_from_board(board, company_id="co-x", now=NOW)}
    assert first == second


def test_greenfield_matches_bare_first_hire_and_comma_head():
    board = _board(
        [
            _posting("g1", "First Hire", department="Growth"),
            _posting("e1", "Engineer 1", department="Engineering"),
            _posting("e2", "Engineer 2", department="Engineering"),
        ]
    )
    titles = {
        s.extracted_facts["posting_title"]
        for s in _by_type(signals_from_board(board, company_id="co-x", now=NOW)).get(
            "greenfield_team", []
        )
    }
    assert "First Hire" in titles

    # "Head, Data" (comma + space) is a lone leadership req -> greenfield.
    board2 = _board(
        [
            _posting("d1", "Head, Data", department="Data"),
            _posting("e1", "Engineer 1", department="Engineering"),
            _posting("e2", "Engineer 2", department="Engineering"),
        ]
    )
    g2 = _by_type(signals_from_board(board2, company_id="co-x", now=NOW))[
        "greenfield_team"
    ]
    assert any(s.extracted_facts["department"] == "Data" for s in g2)


def test_team_used_when_department_absent():
    board = _board(
        [
            _posting("e1", "Backend Engineer", team="Platform"),
            _posting("e2", "Frontend Engineer", team="Platform"),
            _posting("e3", "Infra Engineer", team="Platform"),
        ]
    )
    signals = signals_from_board(board, company_id="co-x", now=NOW)
    surge = _by_type(signals)["department_surge"][0]
    assert surge.extracted_facts["department"] == "Platform"
