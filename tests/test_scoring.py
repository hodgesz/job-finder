"""Tests for the weighted composite scorer (plan section 4)."""

from datetime import datetime, timezone

import pytest

from jobfinder.scoring import (
    WEIGHTS,
    build_opportunity,
    rank_opportunities,
    score_company,
)
from jobfinder.schemas import Evidence, Signal

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _evidence() -> list[Evidence]:
    return [Evidence(source="sec_edgar", locator="acc-1")]


def _signal(
    sid: str,
    signal_type: str,
    *,
    strength: float,
    confidence: float = 0.9,
    observed: datetime = NOW,
    effective: datetime | None = None,
    facts: dict | None = None,
) -> Signal:
    return Signal(
        id=sid,
        company_id="co-1",
        signal_type=signal_type,
        source="sec_edgar",
        observed_at=observed,
        effective_at=effective,
        title=f"{signal_type} signal",
        summary="...",
        extracted_facts=facts or {},
        evidence=_evidence(),
        confidence=confidence,
        strength=strength,
    )


def test_weights_sum_to_one():
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_funding_plus_vacuum_outscores_funding_alone():
    funding = _signal("s-fund", "form_d_funding", strength=0.9)
    vacuum = _signal(
        "s-vac",
        "8k_exec_departure",
        strength=0.75,
        facts={"leadership_vacuum": True, "roles": ["CFO"]},
    )
    both = score_company("co-1", [funding, vacuum], now=NOW).score
    only_funding = score_company("co-1", [funding], now=NOW).score
    assert both > only_funding  # concurrent signals rank higher


def test_liquidity_component_uses_funding_strength_and_weight():
    funding = _signal("s-fund", "form_d_funding", strength=0.8)
    breakdown = score_company("co-1", [funding], now=NOW)
    liq = breakdown.component("liquidity")
    assert liq.raw == 0.8
    assert liq.contribution == pytest.approx(0.30 * 0.8, abs=1e-6)
    assert liq.signal_ids == ["s-fund"]


def test_vacuum_with_named_successor_scores_lower_than_open_search():
    open_search = _signal(
        "s-open", "8k_exec_departure", strength=0.75, facts={"leadership_vacuum": True}
    )
    filled = _signal(
        "s-filled",
        "8k_exec_departure",
        strength=0.4,
        facts={"leadership_vacuum": False},
    )
    open_score = score_company("co-1", [open_search], now=NOW).component(
        "leadership_vacuum"
    )
    filled_score = score_company("co-1", [filled], now=NOW).component(
        "leadership_vacuum"
    )
    assert open_score.raw > filled_score.raw
    assert "open search" in open_score.note


def test_recency_decays_with_age():
    fresh = _signal("s-fresh", "form_d_funding", strength=0.9, effective=NOW)
    old = _signal(
        "s-old",
        "form_d_funding",
        strength=0.9,
        effective=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    fresh_r = score_company("co-1", [fresh], now=NOW).component("recency").raw
    old_r = score_company("co-1", [old], now=NOW).component("recency").raw
    assert fresh_r > old_r
    assert fresh_r == pytest.approx(1.0, abs=1e-3)


def test_future_dated_signal_does_not_overflow_score():
    # A signal dated after `now` (clock skew, or an event "effective [future
    # date]") must not push recency — or the composite score — above 1.0, which
    # would crash the Opportunity score validator.
    future = _signal(
        "s-future",
        "form_d_funding",
        strength=1.0,
        effective=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    breakdown = score_company("co-1", [future], company_fit=1.0, now=NOW)
    assert breakdown.component("recency").raw <= 1.0
    assert breakdown.score <= 1.0
    # Must not raise when building the Opportunity.
    opp = build_opportunity(breakdown, signals=[future])
    assert 0.0 <= opp.score <= 1.0


def test_unwired_components_score_zero():
    funding = _signal("s-fund", "form_d_funding", strength=0.9)
    breakdown = score_company("co-1", [funding], now=NOW)
    assert breakdown.component("hiring_velocity").raw == 0.0
    assert breakdown.component("strategic_language").raw == 0.0


def test_build_opportunity_cites_signals_and_sets_persona():
    funding = _signal("s-fund", "form_d_funding", strength=0.9)
    vacuum = _signal(
        "s-vac",
        "8k_exec_departure",
        strength=0.75,
        facts={"leadership_vacuum": True, "roles": ["CFO"]},
    )
    signals = [funding, vacuum]
    breakdown = score_company("co-1", signals, now=NOW)
    opp = build_opportunity(breakdown, signals=signals)
    assert set(opp.supporting_signal_ids) == {"s-fund", "s-vac"}
    assert opp.target_persona == "CFO / VP Finance"
    assert "Concurrent signals" in opp.why_now
    assert 0.0 <= opp.score <= 1.0
    assert opp.urgency > 0  # vacuum + fresh signal drives urgency


def test_rank_orders_companies_best_first_and_skips_empty():
    high = [
        _signal("a-fund", "form_d_funding", strength=0.9, facts={}),
        _signal(
            "a-vac",
            "8k_exec_departure",
            strength=0.75,
            facts={"leadership_vacuum": True, "roles": ["CFO"]},
        ),
    ]
    # Rebind company_id by constructing fresh signals for co-2.
    low = [
        Signal(
            id="b-fund",
            company_id="co-2",
            signal_type="form_d_funding",
            source="sec_edgar",
            observed_at=NOW,
            title="small raise",
            summary="...",
            evidence=_evidence(),
            confidence=0.95,
            strength=0.3,
        )
    ]
    ranked = rank_opportunities({"co-1": high, "co-2": low, "co-3": []}, now=NOW)
    assert [o.company_id for o in ranked] == ["co-1", "co-2"]  # co-3 skipped
    assert ranked[0].score > ranked[1].score
