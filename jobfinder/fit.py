"""Candidate-vs-company fit model — the ``company_fit`` scorer component.

Until now ``company_fit`` was the one scorer component with no model behind it:
every company got a neutral, caller-supplied ``0.5`` placeholder. This module
fills that gap. Given a company's already-fetched firmographics (sector, funding
stage, headcount) and a ``CandidateProfile`` describing the kind of role the
person is hunting for, it derives an explainable [0, 1] fit score plus a short
human reason — so a high fit reads as *"Series B robotics, ~180 ppl matches
target sector & stage"*, not an opaque magic number.

Design constraints, kept consistent with the signal modules that feed the
scorer:

- **Pure, offline, deterministic.** No network, no clock, no LLM — fit is a
  table-driven weighted blend of three sub-scores (sector, stage, size). The
  same inputs always produce the same score and reason.
- **Decoupled from the scorer.** Like ``jobfinder.signals.*``, this module is an
  *input* to ``scoring`` (consumed by ``pipeline``), not imported by it. The
  scorer stays a pure weighted sum that takes a fit float + a note.
- **Missing data is neutral, never punitive.** Sources surface different
  firmographics; an absent field contributes the neutral 0.5 for its dimension
  rather than dragging the score to zero — so an unknown company lands near the
  old 0.5 placeholder rather than below it.
- **Wire contract untouched.** Firmographics ride on the pipeline's
  ``CompanyInputs``, never on ``schemas.Company`` (which stays the pure
  serialised contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sub-score for a dimension we can't evaluate (missing firmographic field, or a
# profile that expresses no preference on it). Neutral, not punitive: an unknown
# company should land near the historical 0.5 placeholder, not below it.
NEUTRAL = 0.5

# How the three dimensions blend into the overall fit score. Sector match is the
# strongest signal that a company is in the candidate's world; stage and size
# refine it. Sums to 1.0 so a perfect match on all three is exactly 1.0.
FIT_WEIGHTS: dict[str, float] = {
    "sector": 0.5,
    "stage": 0.3,
    "size": 0.2,
}

# Canonical funding-stage progression. Used so an *adjacent* stage (a Series A
# when the candidate targets Series B) earns partial credit rather than being
# treated as far off. Names are normalised (lowercased, spaces/hyphens -> "_")
# before lookup, so "Series B" and "series-b" both resolve to "series_b".
_STAGE_ORDER: tuple[str, ...] = (
    "pre_seed",
    "seed",
    "series_a",
    "series_b",
    "series_c",
    "series_d",
    "growth",
    "public",
)

# Partial-credit scores for a stage that is known but not an exact target.
_STAGE_ADJACENT = 0.6  # one step away on the progression
_STAGE_FAR = 0.2  # two or more steps away, or an unrecognised stage string

# Partial-credit score for a headcount that misses the target band but lands
# within a factor of two of the nearest bound (a near-miss on size).
_SIZE_NEAR = 0.5
_SIZE_FAR = 0.2
_SIZE_NEAR_FACTOR = 2.0


@dataclass(frozen=True)
class Firmographics:
    """Already-fetched structured facts about a company, used for fit scoring.

    Every field is optional because different enrichment sources surface
    different ones; an absent field contributes the neutral sub-score for its
    dimension rather than penalising the company. Fetching these is out of scope
    here (mirrors the SEC/ATS pattern: callers supply already-fetched inputs).
    """

    sector: str | None = None  # free text, e.g. "Robotics", "Industrial Robotics"
    funding_stage: str | None = None  # e.g. "seed", "Series B", "public"
    employee_count: int | None = None


@dataclass(frozen=True)
class CandidateProfile:
    """What the person we're hunting roles for is looking for.

    The fit model scores a company's firmographics against this. A profile that
    leaves a dimension empty (e.g. no ``target_sectors``) expresses no
    preference there, so that dimension scores neutral for every company.
    """

    target_sectors: tuple[str, ...] = ()
    target_stages: tuple[str, ...] = ()
    min_employees: int | None = None
    max_employees: int | None = None


@dataclass(frozen=True)
class FitAssessment:
    """A derived fit score in [0, 1] with a short, human-readable reason.

    ``reason`` is what makes the score explainable end to end: it flows into the
    ``company_fit`` ScoreComponent's note and the opportunity's ``why_now`` so a
    reader can see *why* a company fits, not just how much.
    """

    score: float
    reason: str
    fragments: list[str] = field(default_factory=list)


def _normalise(value: str) -> str:
    """Canonical key for a stage string: lowercased, with any run of spaces,
    hyphens or underscores collapsed to a single ``_``. So "Series  B",
    "Series-B" and "series_b" all map to "series_b"."""
    return "_".join(value.lower().replace("-", " ").replace("_", " ").split())


def _tokens(value: str) -> set[str]:
    """Lowercased word tokens of a free-text label, for whole-word matching."""
    return set(value.lower().replace("-", " ").replace("_", " ").split())


def _score_sector(
    sector: str | None, targets: tuple[str, ...]
) -> tuple[float, str | None]:
    """Sector sub-score. A target matches the company sector when one's word
    set is a subset of the other's (so "Robotics" matches "Industrial Robotics",
    but the short token "AI" does NOT spuriously match "Maine Logistics" — a
    substring test would have, since "ai" is inside "maine")."""
    if not targets:
        return NEUTRAL, None  # profile expresses no sector preference
    if not sector:
        return NEUTRAL, "sector unknown"
    sector_tokens = _tokens(sector)
    for target in targets:
        t = _tokens(target)
        if t and (t <= sector_tokens or sector_tokens <= t):
            return 1.0, f"{sector} matches target sector"
    return 0.0, f"{sector} outside target sectors"


def _stage_index(stage: str) -> int | None:
    try:
        return _STAGE_ORDER.index(_normalise(stage))
    except ValueError:
        return None


def _score_stage(
    stage: str | None, targets: tuple[str, ...]
) -> tuple[float, str | None]:
    """Stage sub-score: exact target = 1.0, one step away = partial, else far.

    An adjacent stage earns partial credit only when *both* the company stage
    and at least one target are on the known progression; an unrecognised stage
    string can still match a target exactly (string equality) but never counts
    as adjacent.
    """
    if not targets:
        return NEUTRAL, None
    if not stage:
        return NEUTRAL, "stage unknown"
    norm = _normalise(stage)
    target_norms = {_normalise(t) for t in targets}
    if norm in target_norms:
        return 1.0, f"{stage} in target stage"
    idx = _stage_index(stage)
    if idx is not None:
        target_indices = [_stage_index(t) for t in targets]
        if any(ti is not None and abs(ti - idx) == 1 for ti in target_indices):
            return _STAGE_ADJACENT, f"{stage} near target stage"
    return _STAGE_FAR, f"{stage} outside target stage"


def _score_size(
    employee_count: int | None, profile: CandidateProfile
) -> tuple[float, str | None]:
    """Size sub-score against the profile's headcount band.

    Inside the band = 1.0; within a factor of two of the nearest bound = near;
    further out = far. A profile with no bounds expresses no size preference.
    """
    low, high = profile.min_employees, profile.max_employees
    if low is None and high is None:
        return NEUTRAL, None
    if employee_count is None:
        return NEUTRAL, "headcount unknown"
    ppl = f"~{employee_count} ppl"
    if (low is None or employee_count >= low) and (
        high is None or employee_count <= high
    ):
        return 1.0, f"{ppl} in target size"
    if low is not None and employee_count < low:
        near = employee_count * _SIZE_NEAR_FACTOR >= low
    else:  # employee_count > high
        near = employee_count <= high * _SIZE_NEAR_FACTOR
    if near:
        return _SIZE_NEAR, f"{ppl} near target size"
    return _SIZE_FAR, f"{ppl} outside target size"


def assess_fit(
    firmographics: Firmographics, profile: CandidateProfile
) -> FitAssessment:
    """Derive a company_fit score in [0, 1] with an explainable reason.

    Blends sector, stage and size sub-scores by ``FIT_WEIGHTS``. Each dimension
    contributes a fragment to the reason; dimensions the profile or firmographics
    leave unevaluable contribute the neutral sub-score and no fragment, so the
    reason lists only what actually moved the score.
    """
    sector_score, sector_frag = _score_sector(
        firmographics.sector, profile.target_sectors
    )
    stage_score, stage_frag = _score_stage(
        firmographics.funding_stage, profile.target_stages
    )
    size_score, size_frag = _score_size(firmographics.employee_count, profile)

    score = (
        FIT_WEIGHTS["sector"] * sector_score
        + FIT_WEIGHTS["stage"] * stage_score
        + FIT_WEIGHTS["size"] * size_score
    )
    score = round(min(max(score, 0.0), 1.0), 3)

    # A fragment is omitted only when the profile expresses no preference on that
    # dimension; if all three are omitted the profile carries no criteria at all
    # (the firmographics may well be present), so say that rather than implying
    # the company had no firmographics to assess.
    fragments = [f for f in (sector_frag, stage_frag, size_frag) if f]
    reason = "; ".join(fragments) if fragments else "no fit criteria specified"
    return FitAssessment(score=score, reason=reason, fragments=fragments)
