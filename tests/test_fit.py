"""Tests for the candidate-vs-company fit model (Slice 8)."""

import pytest

from jobfinder.fit import (
    FIT_WEIGHTS,
    NEUTRAL,
    CandidateProfile,
    Firmographics,
    assess_fit,
)

# A representative profile: an early-growth robotics/industrial leader hunt.
PROFILE = CandidateProfile(
    target_sectors=("robotics", "industrial"),
    target_stages=("series_a", "series_b"),
    min_employees=50,
    max_employees=300,
)


def test_weights_sum_to_one():
    assert round(sum(FIT_WEIGHTS.values()), 6) == 1.0


def test_perfect_match_scores_one():
    firmo = Firmographics(
        sector="Robotics", funding_stage="Series B", employee_count=180
    )
    result = assess_fit(firmo, PROFILE)
    assert result.score == 1.0
    assert "matches target sector" in result.reason
    assert "in target stage" in result.reason
    assert "in target size" in result.reason


def test_off_profile_on_every_dimension_scores_low():
    firmo = Firmographics(
        sector="Biotechnology", funding_stage="public", employee_count=5000
    )
    result = assess_fit(firmo, PROFILE)
    # Sector miss (0) + far stage (0.2) + far size (0.2) -> well below neutral.
    assert result.score < 0.2
    assert "outside target sectors" in result.reason


def test_sector_substring_matches_either_direction():
    # Target "robotics" matches a more specific company sector...
    a = assess_fit(Firmographics(sector="Industrial Robotics"), PROFILE)
    assert a.fragments and "matches target sector" in a.fragments[0]
    # ...and a broader company sector that contains a target token.
    b = assess_fit(
        Firmographics(sector="Industrial"),
        CandidateProfile(target_sectors=("industrial automation",)),
    )
    assert "matches target sector" in b.fragments[0]


def test_short_sector_token_does_not_substring_match():
    # A short target token must not spuriously match an unrelated sector via
    # substring containment ("ai" is inside "maine"). Whole-word matching only.
    profile = CandidateProfile(target_sectors=("AI",))
    result = assess_fit(Firmographics(sector="Maine Logistics"), profile)
    assert "outside target sectors" in result.reason
    # But a real whole-word match still scores.
    assert (
        "matches target sector"
        in assess_fit(Firmographics(sector="AI Platform"), profile).reason
    )


def test_normalise_collapses_repeated_separators():
    # Scraped data with double spaces ("Series  B") must still match the
    # canonical "series_b" target exactly, not fall through to a far miss.
    profile = CandidateProfile(target_stages=("series_b",))
    result = assess_fit(Firmographics(funding_stage="Series  B"), profile)
    assert "in target stage" in result.reason


def test_adjacent_stage_earns_partial_credit():
    # Series C is one step past the Series B target -> partial, not far.
    adj = assess_fit(Firmographics(funding_stage="Series C"), PROFILE)
    far = assess_fit(Firmographics(funding_stage="public"), PROFILE)
    # Isolate the stage dimension by comparing scores (sector/size are neutral).
    assert adj.score > far.score
    assert "near target stage" in adj.reason
    assert "outside target stage" in far.reason


def test_unrecognised_stage_string_still_matches_target_exactly():
    # A target stage off the canonical progression still matches by equality.
    profile = CandidateProfile(target_stages=("bootstrapped",))
    result = assess_fit(Firmographics(funding_stage="Bootstrapped"), profile)
    assert "in target stage" in result.reason


def test_size_near_miss_vs_far_miss():
    # Just under the 50-person floor but within 2x -> near.
    near = assess_fit(Firmographics(employee_count=30), PROFILE)
    # An order of magnitude over the 300 ceiling -> far.
    far = assess_fit(Firmographics(employee_count=4000), PROFILE)
    assert near.score > far.score
    assert "near target size" in near.reason
    assert "outside target size" in far.reason


def test_missing_field_is_neutral_not_punitive():
    # No firmographics at all against a full profile: every dimension neutral, so
    # the blended score is exactly NEUTRAL (the historical 0.5 placeholder).
    result = assess_fit(Firmographics(), PROFILE)
    assert result.score == pytest.approx(NEUTRAL)
    assert "sector unknown" in result.reason


def test_empty_profile_expresses_no_preference():
    # A profile with no targets scores every company neutral with no fragments.
    firmo = Firmographics(sector="Robotics", funding_stage="Series B")
    result = assess_fit(firmo, CandidateProfile())
    assert result.score == pytest.approx(NEUTRAL)
    assert result.fragments == []
    # The firmographics ARE present here — it's the profile that has no criteria,
    # so the reason must not claim there was nothing to assess.
    assert result.reason == "no fit criteria specified"


def test_no_criteria_reason_does_not_imply_missing_firmographics():
    # Profile with no criteria but firmographics present: the reason describes
    # the missing *criteria*, not missing firmographics (Bugbot finding).
    firmo = Firmographics(sector="Robotics", employee_count=200)
    result = assess_fit(firmo, CandidateProfile())
    assert "firmographics" not in result.reason
    assert result.reason == "no fit criteria specified"


def test_score_is_deterministic_and_bounded():
    firmo = Firmographics(
        sector="Robotics", funding_stage="Series A", employee_count=120
    )
    first = assess_fit(firmo, PROFILE)
    second = assess_fit(firmo, PROFILE)
    assert first == second  # frozen dataclass equality, deterministic
    assert 0.0 <= first.score <= 1.0


def test_one_sided_size_band():
    # A floor with no ceiling: anything at/above the floor is in-band.
    profile = CandidateProfile(min_employees=100)
    assert assess_fit(Firmographics(employee_count=5000), profile).fragments[0]
    assert (
        "in target size"
        in assess_fit(Firmographics(employee_count=5000), profile).reason
    )
    # Below the floor but within 2x -> near.
    assert (
        "near target size"
        in assess_fit(Firmographics(employee_count=60), profile).reason
    )
