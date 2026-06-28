"""Signal -> Opportunity wiring.

This is the in-process pipeline the CLI drives: it takes raw filings (8-K
documents and parsed Form D records) grouped by company, runs each through its
signal extractor, and hands the combined per-company signal lists to the
weighted scorer. Keeping it here (rather than in the CLI) means the wiring is
unit-testable without argparse or stdout.

No A2A, no network of its own — callers supply already-fetched inputs, which
keeps the whole thing offline-testable against fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from jobfinder.fit import CandidateProfile, Firmographics, assess_fit
from jobfinder.schemas import Opportunity, Signal
from jobfinder.scoring import rank_opportunities
from jobfinder.signals.ats_hiring import signals_from_board
from jobfinder.signals.form_d import signals_from_form_d
from jobfinder.signals.sec_8k import signals_from_filing
from jobfinder.sources.ats import JobBoard
from jobfinder.sources.edgar import Filing, FormD


@dataclass
class CompanyInputs:
    """Raw, already-fetched filings for one company, keyed by internal id."""

    company_id: str
    name: str = ""
    # (filing, document text) pairs for 8-K filings.
    eight_k: list[tuple[Filing, str]] = field(default_factory=list)
    # (filing, parsed Form D) pairs.
    form_d: list[tuple[Filing, FormD]] = field(default_factory=list)
    # Public ATS job-board snapshots (Greenhouse/Lever/Ashby).
    ats_boards: list[JobBoard] = field(default_factory=list)
    # Already-fetched firmographics for the candidate-fit model. When a
    # CandidateProfile is supplied to the pipeline, these are scored into a
    # derived company_fit (see `jobfinder.fit`); absent firmographics or no
    # profile fall back to the literal `company_fit` float below.
    firmographics: Firmographics | None = None
    company_fit: float = 0.5


def signals_for_company(
    inputs: CompanyInputs,
    *,
    observed_at: datetime | None = None,
    now: datetime | None = None,
    extractor=None,
) -> list[Signal]:
    """Extract every signal for one company from its raw filings and boards.

    `now` is the reference point for ATS recency windows; it defaults inside the
    ATS stage to `observed_at`/utcnow so existing callers are unaffected.
    """
    signals: list[Signal] = []
    for filing, document in inputs.eight_k:
        signals.extend(
            signals_from_filing(
                filing,
                document,
                company_id=inputs.company_id,
                observed_at=observed_at,
                extractor=extractor,
            )
        )
    for filing, form_d in inputs.form_d:
        signals.extend(
            signals_from_form_d(
                filing,
                form_d,
                company_id=inputs.company_id,
                observed_at=observed_at,
            )
        )
    for board in inputs.ats_boards:
        signals.extend(
            signals_from_board(
                board,
                company_id=inputs.company_id,
                observed_at=observed_at,
                now=now,
            )
        )
    return signals


@dataclass
class PipelineResult:
    """Everything a run produced: the extracted signals and the ranked
    opportunities. The CLI prints the opportunities; the store persists both,
    because cross-run diffing (Pillar I) needs the raw signal history, not just
    the scored output."""

    signals: list[Signal] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)


def _fit_for_company(
    company: CompanyInputs, profile: CandidateProfile | None
) -> tuple[float, str | None]:
    """Resolve one company's (fit_score, fit_reason).

    With a `profile` and firmographics on hand, derive both from the fit model;
    otherwise fall back to the literal `company.company_fit` with no reason (the
    pre-Slice-8 behaviour), so callers that pass neither are unaffected.
    """
    if profile is not None and company.firmographics is not None:
        assessment = assess_fit(company.firmographics, profile)
        return assessment.score, assessment.reason
    return company.company_fit, None


def run_pipeline_detailed(
    companies: list[CompanyInputs],
    *,
    candidate_profile: CandidateProfile | None = None,
    observed_at: datetime | None = None,
    now: datetime | None = None,
    extractor=None,
) -> PipelineResult:
    """Run the pipeline and return both the signals and the ranked opportunities.

    `run_pipeline` is the thin opportunities-only wrapper; this is the variant
    callers use when they also need the signals (e.g. to persist them).

    When `candidate_profile` is given, each company's `company_fit` is *derived*
    from its firmographics against that profile (see `jobfinder.fit`) instead of
    the literal `CompanyInputs.company_fit`; the derived reason rides into the
    score breakdown and the opportunity's `why_now`.
    """
    signals_by_company: dict[str, list[Signal]] = defaultdict(list)
    fit_by_company: dict[str, float] = {}
    fit_reasons: dict[str, str] = {}
    for company in companies:
        signals_by_company[company.company_id].extend(
            signals_for_company(
                company, observed_at=observed_at, now=now, extractor=extractor
            )
        )
        fit_score, fit_reason = _fit_for_company(company, candidate_profile)
        fit_by_company[company.company_id] = fit_score
        if fit_reason is not None:
            fit_reasons[company.company_id] = fit_reason

    opportunities = rank_opportunities(
        dict(signals_by_company),
        company_fit=fit_by_company,
        company_fit_reasons=fit_reasons,
        now=now,
    )
    all_signals = [s for sigs in signals_by_company.values() for s in sigs]
    return PipelineResult(signals=all_signals, opportunities=opportunities)


def run_pipeline(
    companies: list[CompanyInputs],
    *,
    candidate_profile: CandidateProfile | None = None,
    observed_at: datetime | None = None,
    now: datetime | None = None,
    extractor=None,
) -> list[Opportunity]:
    """Full wiring: filings -> signals -> weighted score -> ranked Opportunities.

    `extractor` is forwarded to the 8-K signal stage; leaving it None keeps the
    run hermetic (regex fallback) unless a LLM is configured in the environment.
    """
    return run_pipeline_detailed(
        companies,
        candidate_profile=candidate_profile,
        observed_at=observed_at,
        now=now,
        extractor=extractor,
    ).opportunities
