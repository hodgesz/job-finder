"""Layer-1 deterministic scoring of a canonical job against the target profile.

This is the rules-based scorer (the product spec's "Layer 1"): pure, offline, no
LLM, no network. It produces an explainable 0-100 score with a per-dimension
breakdown and an A/B/C tier, so the ranked list reads like the core system's
``ScoreBreakdown`` — a human can see exactly *why* a job ranks where it does.

A richer LLM relevance re-rank (the spec's "Layer 2") is a later slice that
re-orders the Layer-1 top-N; it degrades to this scorer when no API key is set.

Dimensions scored here (others declared in ``profile`` score neutral until a
later slice can read the full job description):

- ``title_seniority`` — a primary VP/Head/Chief AI title scores full; a secondary
  (Sr Director) title scores partial; nothing matching scores low.
- ``ai_scope`` — presence of AI/data/analytics vocabulary in the title: any hit
  scores a strong base, additional distinct terms add up toward a full score.
- ``location`` — remote / preferred-geo fit.
- ``recency`` — newer postings score higher (with a horizon).

Negative filters short-circuit to a rejected, tier-C ``JobMatch``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from jobfinder.jobsearch.models import (
    CanonicalJob,
    DimensionScore,
    JobMatch,
    Tier,
)
from jobfinder.jobsearch.profile import (
    LIVE_DIMENSIONS,
    TIER_A_MIN,
    TIER_B_MIN,
    TargetProfile,
)

# A posting older than this contributes no recency (senior searches move on a
# months horizon). Matches the spirit of scoring.RECENCY_HORIZON_DAYS without
# importing the core scorer.
RECENCY_HORIZON_DAYS = 45.0

# Neutral sub-score for a dimension we can't evaluate yet (mirrors fit.NEUTRAL).
NEUTRAL = 0.5

# A primary title is a bullseye; a secondary one is capped so it rarely reaches
# tier A on its own (the spec: secondaries only when the rest of the fit is high).
_PRIMARY_TITLE_SCORE = 1.0
_SECONDARY_TITLE_SCORE = 0.55


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens (mirrors jobfinder.fit._tokens)."""
    return set(
        text.lower().replace("-", " ").replace("/", " ").replace("&", " ").split()
    )


def _score_title_seniority(job: CanonicalJob, profile: TargetProfile) -> DimensionScore:
    title = job.title
    if any(p.search(title) for p in profile.primary_title_patterns):
        raw, reason = _PRIMARY_TITLE_SCORE, "primary VP/Head/Chief AI-leadership title"
    elif any(p.search(title) for p in profile.secondary_title_patterns):
        raw, reason = (
            _SECONDARY_TITLE_SCORE,
            "secondary (director-tier) AI-leadership title",
        )
    elif any(p.search(title) for p in profile.seniority_patterns):
        raw, reason = 0.3, "senior title but not a clear AI-leadership match"
    else:
        raw, reason = 0.0, "no leadership-title match"
    return DimensionScore(
        "title_seniority", raw, profile.weights["title_seniority"], reason
    )


def _score_ai_scope(job: CanonicalJob, profile: TargetProfile) -> DimensionScore:
    """Presence of AI/data vocabulary in the title (+ department): a strong base
    for any hit, rising toward 1.0 as more distinct AI/data terms appear."""
    text = job.title
    if job.department:
        text = f"{text} {job.department}"
    toks = _tokens(text)
    hits = toks & profile.ai_keywords
    if not toks:
        return DimensionScore("ai_scope", 0.0, profile.weights["ai_scope"], "no title")
    # Presence-weighted: any AI keyword gives a strong base, more hits add up to 1.
    raw = min(1.0, 0.6 + 0.2 * (len(hits) - 1)) if hits else 0.0
    reason = (
        f"AI/data terms: {', '.join(sorted(hits))}"
        if hits
        else "no AI/data terms in title"
    )
    return DimensionScore("ai_scope", raw, profile.weights["ai_scope"], reason)


def _has_word(text: str, word: str) -> bool:
    """Whole-word (not substring) membership, so 'us' doesn't match 'Austin' and
    'remote' doesn't match inside another token."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _score_location(job: CanonicalJob, profile: TargetProfile) -> DimensionScore:
    weight = profile.weights["location"]
    loc = (job.location or "").lower()
    wt = (job.workplace_type or "").lower()
    if not loc and not wt:
        return DimensionScore("location", NEUTRAL, weight, "location unknown")
    if _has_word(loc, "remote") or wt == "remote":
        if profile.remote_ok:
            return DimensionScore("location", 1.0, weight, "remote")
    if any(_has_word(loc, pref) for pref in profile.preferred_locations):
        return DimensionScore("location", 1.0, weight, f"in preferred location ({loc})")
    if _has_word(loc, "hybrid") or wt == "hybrid":
        return DimensionScore("location", 0.6, weight, "hybrid")
    return DimensionScore(
        "location", 0.3, weight, f"outside preferred locations ({loc})"
    )


def _score_recency(
    job: CanonicalJob, *, now: datetime, weight: float
) -> DimensionScore:
    if job.posted_at is None:
        return DimensionScore("recency", NEUTRAL, weight, "post date unknown")
    posted = job.posted_at
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)
    age_days = (now - posted).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0  # future-dated; treat as brand new, don't reward over 1.0
    raw = max(0.0, 1.0 - age_days / RECENCY_HORIZON_DAYS)
    return DimensionScore("recency", raw, weight, f"posted ~{int(age_days)}d ago")


def _disqualified(job: CanonicalJob, profile: TargetProfile) -> str | None:
    """Return a rejection reason if a negative filter hits, else None.

    An always-disqualifying pattern rejects outright. An IC-role noun rejects
    only when no leadership qualifier rescues it (so "VP, ML Engineering"
    survives but "Senior ML Engineer" — which lacks a VP/Head/Chief/Director
    word — does not).
    """
    title = job.title
    for pat in profile.disqualifying_patterns:
        m = pat.search(title)
        if m:
            return f"disqualifying title token ({m.group(0)!r})"
    has_leadership = any(p.search(title) for p in profile.seniority_patterns)
    if not has_leadership:
        for pat in profile.ic_role_patterns:
            if pat.search(title):
                return "individual-contributor role (no leadership scope)"
    return None


def _tier(score: float) -> Tier:
    if score >= TIER_A_MIN:
        return Tier.A
    if score >= TIER_B_MIN:
        return Tier.B
    return Tier.C


def score_job(
    job: CanonicalJob, profile: TargetProfile, *, now: datetime | None = None
) -> JobMatch:
    """Score one canonical job in [0, 100] with an explainable breakdown.

    ``now`` is the recency reference (defaults to current UTC); injecting it keeps
    tests deterministic. A negative-filter hit short-circuits to a rejected,
    score-0 tier-C match so disqualified roles sort to the bottom but still carry
    a reason.
    """
    reference = now or datetime.now(timezone.utc)

    reject_reason = _disqualified(job, profile)
    if reject_reason is not None:
        return JobMatch(
            job=job,
            score=0.0,
            tier=Tier.C,
            reason=f"Rejected: {reject_reason}",
            dimensions=[],
            rejected=True,
        )

    live_dims = [
        _score_title_seniority(job, profile),
        _score_ai_scope(job, profile),
        _score_location(job, profile),
        _score_recency(job, now=reference, weight=profile.weights["recency"]),
    ]
    # Dimensions we can't judge yet are declared (for an honest, stable breakdown
    # + roadmap) but kept OUT of the score: the score is normalized over the
    # *evaluated* weight mass, so a perfect live-dimension job scores a true 100
    # rather than being capped at the live-weight share (~87.5). When a later
    # slice promotes one of these to a live dimension, it simply joins the
    # denominator — no rescaling of the thresholds.
    neutral_dims = [
        DimensionScore(name, NEUTRAL, weight, f"{name} not yet evaluated")
        for name, weight in profile.weights.items()
        if name not in LIVE_DIMENSIONS
    ]
    dims = live_dims + neutral_dims

    live_weight = sum(d.weight for d in live_dims) or 1.0
    score = round(100.0 * sum(d.contribution for d in live_dims) / live_weight, 1)
    tier = _tier(score)

    top = max(live_dims, key=lambda d: d.contribution)
    reason = f"{tier.value}-tier ({score:.0f}/100): {top.reason}"

    risks = _risks(job, profile)
    return JobMatch(
        job=job, score=score, tier=tier, reason=reason, dimensions=dims, risks=risks
    )


def _risks(job: CanonicalJob, profile: TargetProfile) -> list[str]:
    risks: list[str] = []
    loc = (job.location or "").lower()
    wt = (job.workplace_type or "").lower()
    # A relocation risk only applies when the role is neither remote NOR in a
    # preferred geography — using the same whole-word match as _score_location so
    # the risk can't contradict a full location score (e.g. "United States").
    is_remote = _has_word(loc, "remote") or wt == "remote"
    is_preferred = any(_has_word(loc, pref) for pref in profile.preferred_locations)
    if loc and not is_remote and not is_preferred:
        risks.append("may require relocation or on-site presence")
    if job.best_apply_url is None:
        risks.append("no apply URL captured — open the LinkedIn listing manually")
    return risks


def rank_jobs(
    jobs: list[CanonicalJob], profile: TargetProfile, *, now: datetime | None = None
) -> list[JobMatch]:
    """Score every job and return them sorted best-first (rejected last)."""
    matches = [score_job(j, profile, now=now) for j in jobs]
    return sorted(matches, key=lambda m: (m.rejected, -m.score))
