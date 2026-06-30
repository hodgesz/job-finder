"""Tests for the Layer-1 deterministic job scorer."""

from datetime import datetime, timedelta, timezone

from jobfinder.jobsearch.match import RECENCY_HORIZON_DAYS, rank_jobs, score_job
from jobfinder.jobsearch.models import CanonicalJob, Tier
from jobfinder.jobsearch.profile import LIVE_DIMENSIONS, VP_AI_PROFILE

NOW = datetime(2026, 6, 29, tzinfo=timezone.utc)


def _job(
    title, *, company="Acme", location=None, posted_at=NOW, apply_url="https://x/1"
):
    from jobfinder.jobsearch.normalize import normalize_title

    return CanonicalJob(
        company=company,
        title=title,
        normalized_title=normalize_title(title),
        location=location,
        workplace_type="remote" if location and "remote" in location.lower() else None,
        best_apply_url=apply_url,
        posted_at=posted_at,
    )


def test_primary_vp_ai_role_is_tier_a():
    m = score_job(_job("VP, AI & Data", location="Remote"), VP_AI_PROFILE, now=NOW)
    assert m.tier is Tier.A
    assert m.score >= 80
    assert not m.rejected


def test_ic_engineer_is_rejected():
    m = score_job(_job("Senior Machine Learning Engineer"), VP_AI_PROFILE, now=NOW)
    assert m.rejected
    assert m.tier is Tier.C
    assert "individual-contributor" in m.reason


def test_vp_engineering_title_survives_ic_noun():
    # A leadership word rescues a title that DOES contain an IC noun: "Engineer"
    # is a bare IC noun, but the "VP" leadership qualifier rescues it.
    m = score_job(_job("VP, Machine Learning Engineer"), VP_AI_PROFILE, now=NOW)
    assert not m.rejected


def test_vp_of_data_is_a_primary_title():
    # "VP of Data" is a bullseye data-leadership target and must score as a
    # primary title — but off-target "Data <X>" functions must not.
    primary = score_job(_job("VP of Data", location="Remote"), VP_AI_PROFILE, now=NOW)
    assert primary.tier is Tier.A
    for off_target in ("VP of Data Privacy", "VP of Data Center Operations"):
        m = score_job(_job(off_target, location="Remote"), VP_AI_PROFILE, now=NOW)
        ts = next(d for d in m.dimensions if d.name == "title_seniority")
        assert ts.raw == 0.3, off_target  # generic seniority, not a primary match


def test_principal_ic_is_rejected():
    # "Principal" is a seniority adjective, NOT a leadership noun — a Principal IC
    # must NOT be rescued (regression: "principal" used to live in the rescue set).
    for title in ("Principal ML Engineer", "Principal Data Scientist"):
        m = score_job(_job(title), VP_AI_PROFILE, now=NOW)
        assert m.rejected, title
        assert "individual-contributor" in m.reason


def test_intern_always_disqualified_even_with_leadership_word():
    m = score_job(_job("VP AI Intern"), VP_AI_PROFILE, now=NOW)
    assert m.rejected
    assert "disqualifying" in m.reason


def test_secondary_director_title_is_mid_tier():
    m = score_job(
        _job("Senior Director, Data Science", location="Remote"), VP_AI_PROFILE, now=NOW
    )
    assert not m.rejected
    # A secondary title is capped so it lands below a primary VP role.
    assert m.tier in (Tier.B, Tier.C)


def test_preferred_location_matches_whole_words_not_substrings():
    # "us"/"usa" must match as whole words ("United States"/"USA"), not as a
    # substring of unrelated city names (Austin, Houston) — which used to award a
    # full location score and inflate on-site roles.
    austin = score_job(_job("VP of AI", location="Austin, TX"), VP_AI_PROFILE, now=NOW)
    loc_dim = next(d for d in austin.dimensions if d.name == "location")
    assert loc_dim.raw == 0.3  # outside preferred, not a spurious 1.0

    usa = score_job(_job("VP of AI", location="USA"), VP_AI_PROFILE, now=NOW)
    usa_dim = next(d for d in usa.dimensions if d.name == "location")
    assert usa_dim.raw == 1.0  # genuine preferred location still matches


def test_non_ai_vp_role_scores_low():
    m = score_job(_job("VP of Sales", location="Remote"), VP_AI_PROFILE, now=NOW)
    assert not m.rejected
    # Leadership but no AI scope: shouldn't reach tier A.
    assert m.tier is not Tier.A


def test_recency_decays_score():
    fresh = score_job(
        _job("VP of AI", location="Remote", posted_at=NOW), VP_AI_PROFILE, now=NOW
    )
    stale = score_job(
        _job(
            "VP of AI",
            location="Remote",
            posted_at=NOW - timedelta(days=RECENCY_HORIZON_DAYS * 2),
        ),
        VP_AI_PROFILE,
        now=NOW,
    )
    assert fresh.score > stale.score


def test_preferred_location_has_no_relocation_risk():
    # A role in a preferred geography ("United States") scores a full location
    # fit, so it must NOT also carry a "may require relocation" risk.
    m = score_job(_job("VP of AI", location="United States"), VP_AI_PROFILE, now=NOW)
    loc = next(d for d in m.dimensions if d.name == "location")
    assert loc.raw == 1.0
    assert not any("relocation" in r for r in m.risks)

    # An off-target location still flags the relocation risk.
    off = score_job(_job("VP of AI", location="Austin, TX"), VP_AI_PROFILE, now=NOW)
    assert any("relocation" in r for r in off.risks)


def test_no_apply_url_surfaces_risk():
    m = score_job(
        _job("VP of AI", location="Remote", apply_url=None), VP_AI_PROFILE, now=NOW
    )
    assert any("apply URL" in r for r in m.risks)


def test_breakdown_is_explainable():
    m = score_job(_job("VP of AI", location="Remote"), VP_AI_PROFILE, now=NOW)
    names = {d.name for d in m.dimensions}
    assert {"title_seniority", "ai_scope", "location", "recency"} <= names
    # Score is the live dimensions' contribution renormalized over their weight
    # mass (neutral dimensions are declared but excluded from the score).
    live = [d for d in m.dimensions if d.name in LIVE_DIMENSIONS]
    live_weight = sum(d.weight for d in live)
    total = round(100 * sum(d.contribution for d in live) / live_weight, 1)
    assert abs(total - m.score) < 0.05


def test_perfect_live_job_reaches_100():
    # A perfect live-dimension job scores a true 100, not the old ~87.5 ceiling
    # (neutral, not-yet-evaluated dimensions no longer floor/cap the scale).
    m = score_job(
        _job("VP of AI & Machine Learning", location="Remote", posted_at=NOW),
        VP_AI_PROFILE,
        now=NOW,
    )
    assert m.score == 100.0
    assert m.tier is Tier.A


def test_rank_jobs_sorts_best_first_rejected_last():
    jobs = [
        _job("Senior ML Engineer"),  # rejected
        _job("VP of Sales", location="Onsite NYC"),  # low
        _job("VP, AI & Data", location="Remote"),  # high
    ]
    ranked = rank_jobs(jobs, VP_AI_PROFILE, now=NOW)
    assert ranked[0].job.title == "VP, AI & Data"
    assert ranked[-1].rejected
