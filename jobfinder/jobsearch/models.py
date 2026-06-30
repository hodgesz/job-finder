"""Domain models for the job-search tool.

These are the tool's OWN dataclasses — deliberately separate from
``jobfinder.schemas`` (the core system's pydantic wire contract). The data here
describes *job postings and how well they fit the candidate*, which is a
different concept from the core system's company-level ``Signal``/``Opportunity``.

The flow is: raw postings from each source (``RawPosting``) →
de-duplicated/normalized (``CanonicalJob``) → scored against the target profile
(``JobMatch``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Source(str, Enum):
    """Where a posting was ingested from. ``str`` mixin so it serialises and
    compares as its value (e.g. ``"linkedin_alert"``)."""

    LINKEDIN_ALERT = "linkedin_alert"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    MANUAL = "manual"


@dataclass(frozen=True)
class RawPosting:
    """One job posting as ingested from a single source, before normalization.

    Provenance is preserved: ``source`` says where it came from and
    ``source_job_id`` is that source's own identifier (the LinkedIn job id, the
    ATS requisition id) when available, so the same role seen via two sources can
    be recognised as one. ``url`` is the posting's canonical link (the LinkedIn
    job URL for an alert — stored for *manual* review, never auto-fetched).
    """

    title: str
    company: str
    source: Source
    url: str | None = None
    source_job_id: str | None = None
    location: str | None = None
    workplace_type: str | None = None  # remote | hybrid | onsite, when stated
    department: str | None = None
    posted_at: datetime | None = None
    snippet: str | None = None
    alert_keyword: str | None = None  # the saved-search term that surfaced it


@dataclass(frozen=True)
class CanonicalJob:
    """One real job, after normalizing and de-duplicating across sources.

    ``sources`` keeps every ``RawPosting`` that mapped onto this job, so the
    provenance (and each source's URL/id) is never lost — a LinkedIn alert may
    carry recency while an ATS board carries the true application form.
    ``best_apply_url`` prefers a real ATS apply link over a LinkedIn job URL.
    """

    company: str
    title: str
    normalized_title: str
    location: str | None = None
    workplace_type: str | None = None
    department: str | None = None
    best_apply_url: str | None = None
    posted_at: datetime | None = None
    sources: list[RawPosting] = field(default_factory=list)

    @property
    def source_kinds(self) -> list[str]:
        """Distinct source kinds backing this job, in first-seen order."""
        seen: list[str] = []
        for raw in self.sources:
            if raw.source.value not in seen:
                seen.append(raw.source.value)
        return seen


class Tier(str, Enum):
    """Triage bucket for a scored job (thresholds in ``match``)."""

    A = "A"  # apply + outreach now
    B = "B"  # review
    C = "C"  # likely reject


class ApplicationStatus(str, Enum):
    """Where a job sits in the manual application pipeline (Slice D).

    User-driven: a job is born ``NEW`` on first ingest and only the user advances
    it. Transitions are free (any status to any other) — this is a single-user
    personal CRM, so there is no enforced ordering to fight. The status is
    persisted and is NEVER reset by a later re-ingest of the same job (re-seeing a
    job refreshes its posting/score fields but keeps the status the user set).
    ``str`` mixin so it serialises and compares as its value (e.g. ``"applied"``).
    """

    NEW = "new"  # just ingested, not yet triaged
    INTERESTED = "interested"  # worth pursuing
    APPLIED = "applied"  # application submitted
    INTERVIEWING = "interviewing"  # in the interview loop
    OFFER = "offer"  # offer received
    REJECTED = "rejected"  # declined / passed (either direction)
    ARCHIVED = "archived"  # hidden from the active list


@dataclass(frozen=True)
class DimensionScore:
    """One scored fit dimension: its raw [0,1] score, weight, and a reason.

    ``contribution`` (raw * weight) is what the dimension added to the 0-100
    total, so the breakdown reads like the core scorer's ``ScoreBreakdown`` — a
    human can see exactly why a job ranks where it does.
    """

    name: str
    raw: float
    weight: float
    reason: str

    @property
    def contribution(self) -> float:
        return self.raw * self.weight


@dataclass(frozen=True)
class LlmRerank:
    """An optional Layer-2 (LLM) judgement attached to a Layer-1 ``JobMatch``.

    The deterministic Layer-1 score/tier stay authoritative and unchanged; this
    is an *additive* annotation recording how a Gemini relevance pass re-ordered
    the top-N candidates and why. Surfacing it (rather than silently mutating the
    score) preserves the same explainability invariant as the dimension
    breakdown — a human can see the LLM's contribution and reasoning.

    ``rank`` is the LLM's 1-based ordering position within the candidate set;
    ``relevance`` is its coarse verdict (e.g. "strong"/"moderate"/"weak");
    ``rationale`` is its one-line justification.
    """

    rank: int
    relevance: str
    rationale: str


@dataclass(frozen=True)
class JobMatch:
    """A canonical job plus its deterministic fit assessment.

    ``score`` is 0-100; ``tier`` buckets it; ``reason`` is a one-line human
    summary; ``risks`` are caveats (e.g. "location may require relocation");
    ``dimensions`` is the per-dimension breakdown the score is built from.
    ``rejected`` marks a hard negative-filter hit (an IC role, an internship)
    that disqualifies the job regardless of other dimensions. ``llm`` is an
    optional Layer-2 re-rank annotation (``None`` for a pure Layer-1 result).
    """

    job: CanonicalJob
    score: float
    tier: Tier
    reason: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    rejected: bool = False
    llm: LlmRerank | None = None
