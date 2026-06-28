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

from jobfinder.schemas import Opportunity, Signal
from jobfinder.scoring import rank_opportunities
from jobfinder.signals.form_d import signals_from_form_d
from jobfinder.signals.sec_8k import signals_from_filing
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
    company_fit: float = 0.5


def signals_for_company(
    inputs: CompanyInputs,
    *,
    observed_at: datetime | None = None,
    extractor=None,
) -> list[Signal]:
    """Extract every signal for one company from its raw filings."""
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
    return signals


def run_pipeline(
    companies: list[CompanyInputs],
    *,
    observed_at: datetime | None = None,
    now: datetime | None = None,
    extractor=None,
) -> list[Opportunity]:
    """Full wiring: filings -> signals -> weighted score -> ranked Opportunities.

    `extractor` is forwarded to the 8-K signal stage; leaving it None keeps the
    run hermetic (regex fallback) unless a LLM is configured in the environment.
    """
    signals_by_company: dict[str, list[Signal]] = defaultdict(list)
    fit_by_company: dict[str, float] = {}
    for company in companies:
        signals_by_company[company.company_id].extend(
            signals_for_company(company, observed_at=observed_at, extractor=extractor)
        )
        fit_by_company[company.company_id] = company.company_fit

    return rank_opportunities(
        dict(signals_by_company), company_fit=fit_by_company, now=now
    )
