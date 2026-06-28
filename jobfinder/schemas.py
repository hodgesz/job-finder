"""Shared domain schema for job-finder.

These Pydantic models are the contract every module reads and writes:
signal-gathering modules emit `Signal`s, the scoring module turns
`Signal`s into `Opportunity`s. The central design rule is *evidence-backed
output*: an `Opportunity` may not exist without citing the `Signal`s that
produced it, and a `Signal` should carry the `Evidence` it was derived from.

This module is deliberately framework-free (pydantic only) so it can be
imported by in-process modules now and, later, by extracted A2A services
without dragging in ADK or LangGraph.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SignalType = Literal[
    "ats_hiring_velocity",
    "department_surge",
    "greenfield_team",
    "form_d_funding",
    "form_d_amendment",
    "8k_exec_departure",
    "8k_exec_appointment",
    "8k_distress_correlation",
    "news_funding",
    "new_exec_hire",
    "tech_stack_change",
]

OpportunityType = Literal[
    "hidden_role_likely",
    "warm_networking_target",
    "public_job_match",
    "board_level_intro",
    "consulting_project",
    "recruiter_outreach",
]

OpportunityStatus = Literal[
    "new",
    "reviewed",
    "approved",
    "contacted",
    "dismissed",
]


class Evidence(BaseModel):
    """A pointer to the raw source material backing a signal.

    Evidence is what lets a human (or an audit) re-trace why the system
    believed something. It should reference a retrievable source, not a
    paraphrase: a filing URL, an EDGAR accession number, a snapshot id.
    """

    source: str = Field(
        ..., description="Origin of the evidence, e.g. 'sec_edgar', 'greenhouse'."
    )
    url: str | None = Field(
        None, description="Canonical URL to the source document, if public."
    )
    locator: str | None = Field(
        None,
        description="Stable in-source pointer (EDGAR accession no., snapshot id, anchor).",
    )
    excerpt: str | None = Field(
        None, description="Short verbatim quote supporting the claim."
    )
    retrieved_at: datetime | None = Field(
        None, description="When the source was fetched."
    )


class Company(BaseModel):
    """Resolved company entity. Identity keys are optional because different
    sources surface different ones (EDGAR gives CIK; ATS gives a board URL)."""

    id: str = Field(..., description="Internal stable company id.")
    name: str
    domain: str | None = None
    cik: str | None = Field(None, description="SEC Central Index Key.")
    ticker: str | None = None
    ats_url: str | None = None
    aliases: list[str] = Field(default_factory=list)


class Signal(BaseModel):
    """An evidence-backed observation about a company at a point in time.

    Signals are the core stored artifact, NOT free-form "agent thoughts".
    Every signal should carry the evidence it was extracted from.
    """

    id: str
    company_id: str
    signal_type: SignalType
    source: str
    observed_at: datetime = Field(
        ..., description="When the system observed the signal."
    )
    effective_at: datetime | None = Field(
        None,
        description="When the underlying event took effect (e.g. exec departure date).",
    )
    title: str
    summary: str
    extracted_facts: dict = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="How sure we are the signal is real."
    )
    strength: float = Field(
        ..., ge=0.0, le=1.0, description="How strong the signal is, if real."
    )
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def _require_evidence(self) -> Signal:
        if not self.evidence:
            raise ValueError(
                f"Signal {self.id!r} ({self.signal_type}) must cite at least one Evidence; "
                "signals without sources are not allowed."
            )
        return self


class Opportunity(BaseModel):
    """A ranked, evidence-backed opportunity derived from one or more signals.

    Hard constraint: an opportunity must cite the signal ids that produced it.
    No 'the agent thinks this company might hire' without supporting signals.
    """

    id: str
    company_id: str
    target_persona: str = Field(
        ..., description="e.g. 'VP Sales', 'CRO', 'RevOps leader'."
    )
    opportunity_type: OpportunityType
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: float = Field(..., ge=0.0, le=1.0)
    fit_score: float = Field(..., ge=0.0, le=1.0)
    why_now: str
    recommended_next_action: str
    supporting_signal_ids: list[str] = Field(default_factory=list)
    status: OpportunityStatus = "new"

    @model_validator(mode="after")
    def _require_supporting_signals(self) -> Opportunity:
        if not self.supporting_signal_ids:
            raise ValueError(
                f"Opportunity {self.id!r} must cite at least one supporting signal id; "
                "opportunities without evidence are not allowed."
            )
        return self
