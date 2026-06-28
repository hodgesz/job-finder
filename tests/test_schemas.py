"""Tests for the shared domain schema.

The most important invariants here are the evidence-required rules: they are
the architectural keystone of the system (no opportunity without cited
signals; no signal without cited evidence).
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from jobfinder.schemas import Evidence, Opportunity, Signal

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _evidence() -> Evidence:
    return Evidence(
        source="sec_edgar",
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123.txt",
        locator="0000320193-24-000123",
        excerpt="Item 5.02 ... resignation of the Chief Financial Officer ...",
        retrieved_at=NOW,
    )


def _signal(**overrides) -> Signal:
    base = dict(
        id="sig-1",
        company_id="co-1",
        signal_type="8k_exec_departure",
        source="sec_edgar",
        observed_at=NOW,
        title="CFO departure disclosed",
        summary="8-K Item 5.02(b) discloses CFO resignation with no named successor.",
        extracted_facts={"role": "CFO", "successor_present": False},
        evidence=[_evidence()],
        confidence=0.9,
        strength=0.7,
    )
    base.update(overrides)
    return Signal(**base)


def test_signal_with_evidence_is_valid():
    sig = _signal()
    assert sig.signal_type == "8k_exec_departure"
    assert sig.evidence[0].locator == "0000320193-24-000123"


def test_signal_without_evidence_rejected():
    with pytest.raises(ValidationError, match="must cite at least one Evidence"):
        _signal(evidence=[])


def test_signal_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        _signal(confidence=1.5)


def test_opportunity_with_supporting_signals_is_valid():
    opp = Opportunity(
        id="opp-1",
        company_id="co-1",
        target_persona="CFO",
        opportunity_type="hidden_role_likely",
        score=0.82,
        confidence=0.75,
        urgency=0.6,
        fit_score=0.7,
        why_now="CFO departed via 8-K with no successor; recent Form D funding.",
        recommended_next_action="Draft warm intro via board member (requires approval).",
        supporting_signal_ids=["sig-1"],
    )
    assert opp.status == "new"
    assert opp.supporting_signal_ids == ["sig-1"]


def test_opportunity_without_supporting_signals_rejected():
    with pytest.raises(
        ValidationError, match="must cite at least one supporting signal"
    ):
        Opportunity(
            id="opp-2",
            company_id="co-1",
            target_persona="CFO",
            opportunity_type="hidden_role_likely",
            score=0.5,
            confidence=0.5,
            urgency=0.5,
            fit_score=0.5,
            why_now="hunch",
            recommended_next_action="none",
            supporting_signal_ids=[],
        )
