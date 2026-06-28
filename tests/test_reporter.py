"""Tests for the Reporter (Slice 6) — the cross-run digest renderer.

Fully offline: builds a `StoreDiff` against an in-memory store and asserts the
rendered text. `now` is injected so relative ages are reproducible.
"""

from datetime import datetime, timezone

from jobfinder.reporter import render_digest
from jobfinder.schemas import Evidence, Opportunity, Signal
from jobfinder.store import Store

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
LATER = datetime(2026, 6, 8, tzinfo=timezone.utc)
# A fixed "render clock" a few days past LATER so ages are deterministic.
RENDER_NOW = datetime(2026, 6, 10, tzinfo=timezone.utc)


def _signal(
    sid: str = "acc-1:departure",
    *,
    company_id: str = "co-1",
    observed: datetime = NOW,
) -> Signal:
    return Signal(
        id=sid,
        company_id=company_id,
        signal_type="8k_exec_departure",
        source="sec_edgar",
        observed_at=observed,
        effective_at=observed,
        title="8-K Item 5.02 departure",
        summary="CFO resigned, no successor named.",
        evidence=[
            Evidence(source="sec_edgar", url="https://example.com/8k", locator="acc-1")
        ],
        confidence=0.9,
        strength=0.75,
    )


def _opportunity(
    oid: str = "opp:co-1",
    *,
    company_id: str = "co-1",
    score: float = 0.62,
    signal_ids: list[str] | None = None,
) -> Opportunity:
    return Opportunity(
        id=oid,
        company_id=company_id,
        target_persona="CFO / VP Finance",
        opportunity_type="hidden_role_likely",
        score=score,
        confidence=0.9,
        urgency=0.5,
        fit_score=0.8,
        why_now="Concurrent funding + vacuum.",
        recommended_next_action="Warm intro to the board.",
        supporting_signal_ids=signal_ids or ["acc-1:departure"],
    )


def test_empty_store_renders_no_opportunities():
    store = Store.in_memory()
    out = render_digest(store.diff(), now=RENDER_NOW)
    assert "No opportunities on file." in out


def test_standings_digest_has_no_since_header_and_no_new_tags():
    store = Store.in_memory()
    store.persist_run([_signal()], [_opportunity()], now=NOW)
    out = render_digest(store.diff(), now=RENDER_NOW)
    assert "current standings" in out
    assert "since" not in out.lower()
    # No baseline -> recurring, not NEW.
    assert "[recurring]" in out
    assert "[NEW]" not in out
    # The cited supporting signal id is traceable in the rendered line.
    assert "acc-1:departure" in out


def test_since_digest_flags_new_and_lists_new_signals():
    store = Store.in_memory()
    store.persist_run([_signal()], [_opportunity()], now=LATER)
    cutoff = datetime(2026, 6, 5, tzinfo=timezone.utc)
    out = render_digest(store.diff(since=cutoff), now=RENDER_NOW)
    assert "what changed since 2026-06-05" in out
    assert "[NEW]" in out
    assert "1 new this window" in out
    # Newly appeared signal cited to its evidence URL.
    assert "Newly appeared signals (1):" in out
    assert "https://example.com/8k" in out


def test_resaved_without_score_change_reads_updated_not_recurring():
    store = Store.in_memory()
    store.save_opportunity(_opportunity(score=0.50), now=NOW)
    cutoff = datetime(2026, 6, 5, tzinfo=timezone.utc)
    # Re-saved after the cutoff at the same score: touched this window, but no
    # movement -> [updated], not a stale arrow and not [recurring].
    store.save_opportunity(_opportunity(score=0.50), now=LATER)
    out = render_digest(store.diff(since=cutoff), now=RENDER_NOW)
    assert "[updated]" in out
    assert "↑" not in out and "↓" not in out


def test_sub_threshold_delta_does_not_render_contradictory_arrow():
    store = Store.in_memory()
    store.save_opportunity(_opportunity(score=0.500), now=NOW)
    # A movement smaller than display precision must not print "[↑ +0.00]".
    store.save_opportunity(_opportunity(score=0.502), now=LATER)
    out = render_digest(store.diff(), now=RENDER_NOW)
    assert "+0.00" not in out
    assert "↑" not in out


def test_window_digest_hides_movement_that_predates_the_cutoff():
    """A row whose score moved before the --since cutoff and was NOT re-saved in
    the window must read [recurring], not a stale [↑] — the digest title only
    promises changes since the cutoff."""
    store = Store.in_memory()
    # Move the score at NOW (0.40 -> 0.55), then never touch it again.
    store.save_opportunity(
        _opportunity(score=0.40), now=datetime(2026, 5, 20, tzinfo=timezone.utc)
    )
    store.save_opportunity(_opportunity(score=0.55), now=NOW)
    # Cutoff is AFTER the last save, so nothing changed within the window.
    cutoff = datetime(2026, 6, 5, tzinfo=timezone.utc)
    out = render_digest(store.diff(since=cutoff), now=RENDER_NOW)
    assert "[recurring]" in out
    assert "↑" not in out and "+0.15" not in out


def test_windowed_digest_lists_new_signals_even_with_no_opportunities():
    """A windowed report must still surface newly-appeared signals when the
    store has no opportunity rows — they are the 'what changed' the title
    promises, and a signal can appear before it scores into an opportunity."""
    store = Store.in_memory()
    # A signal but no opportunity (e.g. a lone departure not yet scored).
    store.persist_run([_signal()], [], now=LATER)
    cutoff = datetime(2026, 6, 5, tzinfo=timezone.utc)
    out = render_digest(store.diff(since=cutoff), now=RENDER_NOW)
    assert "No opportunities on file." in out
    assert "Newly appeared signals (1):" in out
    assert "https://example.com/8k" in out


def test_standings_digest_omits_window_wording_and_new_signals_section():
    """Without --since there is no window: no "(N new this window)" and no
    "Newly appeared signals" section."""
    store = Store.in_memory()
    store.persist_run([_signal()], [_opportunity()], now=NOW)
    out = render_digest(store.diff(), now=RENDER_NOW)
    assert "this window" not in out
    assert "Newly appeared signals" not in out


def test_standings_digest_shows_last_upsert_movement():
    """In the window-less standings view the last-upsert delta is shown directly
    (no cutoff to mislead about)."""
    store = Store.in_memory()
    store.save_opportunity(_opportunity(score=0.40), now=NOW)
    store.save_opportunity(_opportunity(score=0.55), now=LATER)
    out = render_digest(store.diff(), now=RENDER_NOW)
    assert "↑ +0.15" in out


def test_top_limits_opportunities_but_count_reflects_full_store():
    store = Store.in_memory()
    store.save_opportunity(_opportunity("opp:a", company_id="co-a", score=0.9), now=NOW)
    store.save_opportunity(_opportunity("opp:b", company_id="co-b", score=0.5), now=NOW)
    out = render_digest(store.diff(), now=RENDER_NOW, top=1)
    assert "2 opportunities on file" in out
    assert "showing 1." in out
    assert "co-a" in out
    assert "co-b" not in out


def test_evidence_and_why_now_survive_into_digest():
    store = Store.in_memory()
    store.persist_run([_signal()], [_opportunity()], now=NOW)
    out = render_digest(store.diff(), now=RENDER_NOW)
    assert "Why now: Concurrent funding + vacuum." in out
    assert "Evidence (supporting signals): acc-1:departure" in out
    assert "Target: CFO / VP Finance" in out
