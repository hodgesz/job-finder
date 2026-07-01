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


class ContactRole(str, Enum):
    """How a person relates to a job, for ranking who to reach out to (Slice E).

    Ordering of *who matters most* is decided in ``contacts``, not here; this only
    classifies a person. ``str`` mixin so it serialises/compares as its value.
    """

    HIRING_MANAGER = "hiring_manager"  # owns the req / the role reports to them
    FUNCTION_LEADER = "function_leader"  # existing AI/data/analytics leadership
    EXECUTIVE = "executive"  # CEO / founder / C-suite sponsor
    RECRUITER = "recruiter"  # talent / recruiting owning the search
    TEAM = "team"  # a would-be peer on the team
    REFERRAL = "referral"  # a mutual connection who can refer
    OTHER = "other"


class ContactSource(str, Enum):
    """How a contact was discovered. LinkedIn is reached ONLY by manual review the
    user pastes back — never scraped — so a ``MANUAL`` contact is the norm."""

    LISTING = "listing"  # named in the job listing itself
    MANUAL = "manual"  # found via manual LinkedIn review and recorded by hand


@dataclass(frozen=True)
class ContactTarget:
    """A *role worth contacting* for a job — a "who to look for", not a person.

    The target list is generated deterministically from the canonical job (see
    ``contacts.target_contacts``) and rendered as a manual checklist. ``priority``
    is 1-based (1 = reach first). ``linkedin_search`` is plain text the user pastes
    into LinkedIn's search box *by hand* — the tool never queries or scrapes
    LinkedIn — and ``rationale`` explains why this target is worth the effort.
    """

    role: ContactRole
    priority: int
    label: str
    rationale: str
    linkedin_search: str


@dataclass(frozen=True)
class Contact:
    """A real named person to (maybe) reach out to about a job.

    Recorded by the user from a manual LinkedIn review (the checklist paste-back)
    or named in the listing. ``email_domain`` is the company's *business* email
    domain (e.g. "acme.com") used to infer a business email — never a personal
    domain (see ``email_format``). Email guesses are deliberately NOT stored on the
    contact: they are recomputed at display time so the always-honored
    do-not-contact list is applied against its current state, never a stale snapshot.
    """

    name: str
    company: str
    role: ContactRole = ContactRole.OTHER
    title: str | None = None
    linkedin_url: str | None = None
    email_domain: str | None = None
    source: ContactSource = ContactSource.MANUAL
    notes: str | None = None


class DraftStatus(str, Enum):
    """Where an assembled outreach email sits in the draft-and-approve flow (Slice F).

    An email is born ``DRAFTED`` when assembled — that is the *only* state in
    which it has not left the machine. It moves to ``SENT`` solely through the
    explicit, separate ``outreach send <id> --confirm`` step; nothing else
    advances it. There is no autonomous send: the gate between these two states
    IS the feature. ``str`` mixin so it serialises/compares as its value.
    """

    DRAFTED = "drafted"  # assembled + stored; nothing sent
    SENT = "sent"  # explicitly approved and sent via the gmail.send seam


@dataclass(frozen=True)
class OutreachEmail:
    """A tailored outreach email assembled for a contact about a job (Slice F).

    Assembled deterministically from data the tool already has (the
    ``CanonicalJob``, the ``Contact`` + role, the persona/match reason), optionally
    sharpened by an injected LLM tailoring seam. It is a *draft* — assembling one
    sends nothing. The two-step ``outreach send <id> --confirm`` gate is the only
    path that puts it on the wire.

    CAN-SPAM-style hygiene is structural, not decorative: ``from_name`` /
    ``from_email`` are the sender's real identity (a truthful "From"), ``subject``
    is a truthful, non-deceptive summary, and ``opt_out`` is a plain way for the
    recipient to ask not to be contacted again (the user records that back via
    ``dnc``). ``to_email`` is always a *business* address (personal domains are
    refused upstream) that is NOT on the do-not-contact list at assembly time —
    and the send gate re-checks it (defence in depth).

    ``tailoring`` records how the body was produced ("template" or
    "llm+template") so the output is explainable, mirroring how ``LlmRerank``
    surfaces the LLM's contribution rather than hiding it.
    """

    to_email: str
    to_name: str
    subject: str
    body: str
    from_name: str
    from_email: str
    company: str
    job_title: str
    opt_out: str
    tailoring: str = "template"


@dataclass(frozen=True)
class EmailGuess:
    """A confidence-scored *business* email constructed for a person at a domain.

    ``pattern`` names the construction rule (e.g. "first.last", "flast");
    ``confidence`` is 0-1; ``provenance`` records whether the confidence came from
    the heuristic prior ("heuristic") or was sharpened by an email-format lookup
    ("format-source"). The guess is always a business email — personal domains are
    refused upstream — and is suppressed entirely when on the do-not-contact list.
    """

    email: str
    pattern: str
    confidence: float
    provenance: str
    domain: str
