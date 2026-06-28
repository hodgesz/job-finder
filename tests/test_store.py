"""Tests for the persistence layer (Slice 3).

Runs against an in-memory SQLite store — the same code path that backs Postgres
in production, exercised offline and hermetically here.
"""

from datetime import datetime, timezone

import pytest

from jobfinder.schemas import Evidence, Opportunity, Signal
from jobfinder.store import Store
from jobfinder.store.db import opportunities_table, signals_table

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
LATER = datetime(2026, 6, 8, tzinfo=timezone.utc)


def _signal(
    sid: str = "acc-1:departure",
    *,
    company_id: str = "co-1",
    signal_type: str = "8k_exec_departure",
    observed: datetime = NOW,
    effective: datetime | None = NOW,
    strength: float = 0.75,
) -> Signal:
    return Signal(
        id=sid,
        company_id=company_id,
        signal_type=signal_type,
        source="sec_edgar",
        observed_at=observed,
        effective_at=effective,
        title="8-K Item 5.02 departure",
        summary="CFO resigned, no successor named.",
        extracted_facts={"leadership_vacuum": True, "roles": ["CFO"]},
        evidence=[
            Evidence(
                source="sec_edgar",
                url="https://example.com/8k",
                locator="acc-1",
                excerpt="the Chief Financial Officer notified the Board",
                retrieved_at=observed,
            )
        ],
        confidence=0.9,
        strength=strength,
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


@pytest.fixture
def store() -> Store:
    return Store.in_memory()


def test_signal_round_trips_with_full_fidelity(store: Store):
    original = _signal()
    assert store.save_signal(original, now=NOW) is True
    loaded = store.get_signal(original.id)
    assert loaded is not None
    # Full model equality: nested Evidence, facts, tz-aware datetimes all survive.
    assert loaded == original
    assert loaded.observed_at.tzinfo is not None
    assert loaded.evidence[0].locator == "acc-1"


def test_opportunity_round_trips_with_full_fidelity(store: Store):
    original = _opportunity()
    assert store.save_opportunity(original, now=NOW) is True
    loaded = store.get_opportunity(original.id)
    assert loaded == original


def test_get_missing_returns_none(store: Store):
    assert store.get_signal("nope") is None
    assert store.get_opportunity("nope") is None


def test_resaving_same_id_updates_not_duplicates(store: Store):
    assert store.save_signal(_signal(strength=0.75), now=NOW) is True
    # Same id, changed payload, later run -> update, not a second row.
    assert store.save_signal(_signal(strength=0.4), now=LATER) is False
    with store.engine.connect() as conn:
        count = conn.execute(signals_table.select()).fetchall()
    assert len(count) == 1
    assert store.get_signal("acc-1:departure").strength == 0.4


def test_first_seen_is_preserved_across_resaves(store: Store):
    """The cross-run-diff hook: first_seen stays put, updated_at advances."""
    store.save_signal(_signal(), now=NOW)
    assert store.first_seen("signals", "acc-1:departure") == NOW
    store.save_signal(_signal(strength=0.4), now=LATER)
    # first_seen unchanged despite the re-save a week later.
    assert store.first_seen("signals", "acc-1:departure") == NOW
    # ...but updated_at moved to the later run.
    with store.engine.connect() as conn:
        row = (
            conn.execute(
                signals_table.select().where(signals_table.c.id == "acc-1:departure")
            )
            .mappings()
            .first()
        )
    assert row["updated_at"] == LATER.isoformat()
    assert row["first_seen_at"] == NOW.isoformat()


def test_persist_run_counts_inserts_and_updates(store: Store):
    sig = _signal()
    opp = _opportunity()
    first = store.persist_run([sig], [opp], now=NOW)
    assert (first.signals_inserted, first.opportunities_inserted) == (1, 1)
    assert (first.signals_updated, first.opportunities_updated) == (0, 0)
    assert first.total == 2

    # Re-running the same pipeline a week later: same ids -> all updates.
    second = store.persist_run([_signal(strength=0.5)], [opp], now=LATER)
    assert (second.signals_updated, second.opportunities_updated) == (1, 1)
    assert (second.signals_inserted, second.opportunities_inserted) == (0, 0)


def test_persist_run_is_atomic(store: Store):
    """A bad row in the batch rolls the whole run back — no partial writes."""
    good = _signal("acc-1:departure")

    class Boom(Signal):
        def model_dump(self, *a, **k):  # type: ignore[override]
            raise RuntimeError("serialization blew up")

    bad = Boom(**good.model_dump())
    with pytest.raises(RuntimeError):
        store.persist_run([good, bad], [], now=NOW)
    # The good signal must not have been committed.
    assert store.get_signal("acc-1:departure") is None


def test_signals_for_company_orders_newest_first(store: Store):
    older = _signal("acc-old:departure", observed=NOW)
    newer = _signal("acc-new:departure", observed=LATER)
    store.save_signal(older, now=NOW)
    store.save_signal(newer, now=LATER)
    # A different company's signal must not leak in.
    store.save_signal(_signal("other:departure", company_id="co-2"), now=NOW)

    ids = [s.id for s in store.signals_for_company("co-1")]
    assert ids == ["acc-new:departure", "acc-old:departure"]


def test_opportunities_for_company_orders_by_score(store: Store):
    store.save_opportunity(_opportunity("opp:lo", score=0.3), now=NOW)
    store.save_opportunity(_opportunity("opp:hi", score=0.9), now=NOW)
    ids = [o.id for o in store.opportunities_for_company("co-1")]
    assert ids == ["opp:hi", "opp:lo"]


def test_top_opportunities_ranks_globally_and_caps(store: Store):
    store.save_opportunity(_opportunity("opp:a", company_id="co-a", score=0.4), now=NOW)
    store.save_opportunity(_opportunity("opp:b", company_id="co-b", score=0.8), now=NOW)
    store.save_opportunity(_opportunity("opp:c", company_id="co-c", score=0.6), now=NOW)
    top2 = store.top_opportunities(limit=2)
    assert [o.id for o in top2] == ["opp:b", "opp:c"]
    assert len(store.top_opportunities()) == 3


def test_naive_datetime_is_stored_as_utc(store: Store):
    """Defensive: a naive observed_at is treated as UTC for the sort column."""
    naive = datetime(2026, 6, 1)  # no tzinfo
    store.save_signal(_signal(observed=naive, effective=None), now=NOW)
    with store.engine.connect() as conn:
        row = conn.execute(signals_table.select()).mappings().first()
    assert row["observed_at"] == "2026-06-01T00:00:00+00:00"
    assert row["effective_at"] is None


def test_create_false_skips_table_creation():
    store = Store(create=False)
    # Tables don't exist yet; creating them explicitly should then work.
    store.create_all()
    assert store.save_signal(_signal(), now=NOW) is True


def test_first_seen_rejects_unknown_table(store: Store):
    with pytest.raises(ValueError, match="unknown table"):
        store.first_seen("widgets", "x")


def test_save_dispatches_on_model_type(store: Store):
    # The unified save() routes each model to its own table.
    assert store.save(_signal()) is True
    assert store.save(_opportunity()) is True
    assert store.get_signal("acc-1:departure") is not None
    assert store.get_opportunity("opp:co-1") is not None


def test_tables_are_registered():
    # Guard against a silent rename breaking the schema contract.
    assert signals_table.name == "signals"
    assert opportunities_table.name == "opportunities"
