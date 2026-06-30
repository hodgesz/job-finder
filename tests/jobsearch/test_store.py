"""Tests for the job-search CRM persistence layer (Slice D).

Hermetic: runs against an in-memory SQLite store — the same code path that would
back a SQLite file or Postgres, exercised offline with no network or secrets.
Mirrors the core ``tests/test_store.py`` coverage (round-trip fidelity, idempotent
upsert, additive migration) plus the detour-specific crux: a user-set status
survives a re-ingest of the freshly-ranked job.
"""

from datetime import datetime, timezone

import pytest

from jobfinder.jobsearch.match import score_job
from jobfinder.jobsearch.models import (
    ApplicationStatus,
    CanonicalJob,
    JobMatch,
    LlmRerank,
    RawPosting,
    Source,
    Tier,
)
from jobfinder.jobsearch.normalize import job_key
from jobfinder.jobsearch.profile import VP_AI_PROFILE
from jobfinder.jobsearch.store import JobStore, jobs_table

NOW = datetime(2026, 6, 25, tzinfo=timezone.utc)
LATER = datetime(2026, 6, 27, tzinfo=timezone.utc)


def _raw(
    *,
    title: str = "VP of AI",
    company: str = "Acme, Inc.",
    source: Source = Source.LINKEDIN_ALERT,
    url: str | None = "https://www.linkedin.com/jobs/view/123",
    posted_at: datetime | None = datetime(2026, 6, 20, tzinfo=timezone.utc),
    location: str | None = "Remote (United States)",
) -> RawPosting:
    return RawPosting(
        title=title,
        company=company,
        source=source,
        url=url,
        source_job_id="123",
        location=location,
        department="Data & AI",
        posted_at=posted_at,
        snippet="Lead the AI org.",
        alert_keyword="VP of AI",
    )


def _job(
    *,
    title: str = "VP of AI",
    company: str = "Acme, Inc.",
    location: str | None = "Remote",
    posted_at: datetime | None = datetime(2026, 6, 20, tzinfo=timezone.utc),
    apply_url: str | None = "https://boards.greenhouse.io/acme/jobs/9",
) -> CanonicalJob:
    return CanonicalJob(
        company=company,
        title=title,
        normalized_title=title.lower(),
        location=location,
        department="Data & AI",
        best_apply_url=apply_url,
        posted_at=posted_at,
        sources=[_raw(title=title, company=company, posted_at=posted_at)],
    )


def _match(job: CanonicalJob | None = None, *, now: datetime = NOW) -> JobMatch:
    return score_job(job or _job(), VP_AI_PROFILE, now=now)


@pytest.fixture
def store() -> JobStore:
    return JobStore.in_memory()


# --------------------------------------------------------------------------- #
# Round-trip fidelity.
# --------------------------------------------------------------------------- #
def test_match_round_trips_with_full_fidelity(store: JobStore):
    original = _match()
    assert store.save_match(original, now=NOW) is True

    got = store.get(job_key(original.job))
    assert got is not None
    restored = got.match
    # Scalar fields.
    assert restored.score == original.score
    assert restored.tier is original.tier
    assert restored.reason == original.reason
    assert restored.rejected == original.rejected
    assert restored.risks == original.risks
    # Nested CanonicalJob + RawPosting.
    assert restored.job.company == original.job.company
    assert restored.job.title == original.job.title
    assert restored.job.best_apply_url == original.job.best_apply_url
    assert restored.job.sources[0].source is Source.LINKEDIN_ALERT
    assert restored.job.sources[0].alert_keyword == "VP of AI"
    # Dimension breakdown, including the derived contribution property.
    assert [d.name for d in restored.dimensions] == [
        d.name for d in original.dimensions
    ]
    for r, o in zip(restored.dimensions, original.dimensions):
        assert r.raw == o.raw and r.weight == o.weight and r.reason == o.reason
        assert r.contribution == pytest.approx(o.contribution)


def test_datetime_tz_awareness_survives_round_trip(store: JobStore):
    # An aware posting date must stay aware (and equal) after a DB round-trip —
    # the tz-naive/aware mismatch was the Slice-A headliner crash, so fidelity
    # here is load-bearing.
    aware = datetime(2026, 6, 20, 9, 30, tzinfo=timezone.utc)
    store.save_match(_match(_job(posted_at=aware)), now=NOW)
    got = store.get(job_key(_job(posted_at=aware)))
    assert got.match.job.posted_at == aware
    assert got.match.job.posted_at.tzinfo is not None


def test_naive_posting_date_round_trips_naive(store: JobStore):
    # A LinkedIn "-0000" Date header parses tz-naive; the payload must NOT silently
    # UTC-coerce it (that would change the value), so a naive in stays naive out.
    naive = datetime(2026, 6, 20, 9, 30)
    store.save_match(_match(_job(posted_at=naive)), now=NOW)
    got = store.get(job_key(_job(posted_at=naive)))
    assert got.match.job.posted_at == naive
    assert got.match.job.posted_at.tzinfo is None


def test_none_posting_date_round_trips_none(store: JobStore):
    store.save_match(_match(_job(posted_at=None)), now=NOW)
    got = store.get(job_key(_job(posted_at=None)))
    assert got.match.job.posted_at is None


def test_llm_rerank_annotation_round_trips(store: JobStore):
    from dataclasses import replace

    base = _match()
    annotated = replace(
        base, llm=LlmRerank(rank=1, relevance="strong", rationale="Bullseye VP-of-AI.")
    )
    store.save_match(annotated, now=NOW)
    got = store.get(job_key(base.job))
    assert got.match.llm == LlmRerank(
        rank=1, relevance="strong", rationale="Bullseye VP-of-AI."
    )


def test_get_unknown_id_returns_none(store: JobStore):
    assert store.get("nobody|here|remote") is None


# --------------------------------------------------------------------------- #
# Idempotent upsert.
# --------------------------------------------------------------------------- #
def test_resaving_same_job_is_one_row_first_seen_preserved(store: JobStore):
    m = _match()
    assert store.save_match(m, now=NOW) is True
    # Re-saving the same job returns False (update, not insert) and keeps one row.
    assert store.save_match(m, now=LATER) is False
    assert len(store.list_jobs()) == 1
    got = store.get(job_key(m.job))
    assert got.first_seen_at == NOW  # stamped once
    assert got.last_seen_at == LATER  # advances


def test_resave_refreshes_score_and_payload(store: JobStore):
    job = _job()
    store.save_match(score_job(job, VP_AI_PROFILE, now=NOW), now=NOW)
    # Same job_key (company/title/location unchanged) but a staler posting date →
    # lower recency → different score. The in-place update must reflect the new
    # score/payload.
    older = _job(posted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert job_key(older) == job_key(job)
    store.save_match(score_job(older, VP_AI_PROFILE, now=LATER), now=LATER)
    got = store.get(job_key(job))
    assert got.match.job.posted_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_save_matches_batch_counts(store: JobStore):
    a = _match(_job(title="VP of AI", company="Acme"))
    b = _match(_job(title="Head of AI", company="Globex"))
    result = store.save_matches([a, b], now=NOW)
    assert result.inserted == 2 and result.updated == 0 and result.total == 2
    # Re-running the same batch updates both.
    result2 = store.save_matches([a, b], now=LATER)
    assert result2.inserted == 0 and result2.updated == 2


def test_blank_company_jobs_get_distinct_keys_no_overwrite(store: JobStore):
    # canonicalize refuses to soft-merge blank-company postings (two blanks must
    # not collapse). job_key must mirror that: two DISTINCT under-parsed postings
    # (no company, but distinct hard keys) must persist as TWO rows, not silently
    # overwrite each other.
    raw_a = RawPosting(
        title="VP of AI",
        company="",
        source=Source.LINKEDIN_ALERT,
        url="https://www.linkedin.com/jobs/view/111",
        source_job_id="111",
    )
    raw_b = RawPosting(
        title="VP of AI",
        company="",
        source=Source.LINKEDIN_ALERT,
        url="https://www.linkedin.com/jobs/view/222",
        source_job_id="222",
    )
    job_a = CanonicalJob(
        company="", title="VP of AI", normalized_title="vp of ai", sources=[raw_a]
    )
    job_b = CanonicalJob(
        company="", title="VP of AI", normalized_title="vp of ai", sources=[raw_b]
    )
    assert job_key(job_a) != job_key(job_b)  # distinct hard-key fallbacks
    store.save_match(_match(job_a), now=NOW)
    store.save_match(_match(job_b), now=NOW)
    assert len(store.list_jobs()) == 2  # no silent overwrite


def test_job_key_fully_blank_posting_is_stable(store: JobStore):
    # A posting with no company, no title signature, and no hard key still gets a
    # stable hash key (so re-ingesting the same blank posting updates in place).
    blank = CanonicalJob(company="", title="", normalized_title="", sources=[])
    k1 = job_key(blank)
    k2 = job_key(CanonicalJob(company="", title="", normalized_title="", sources=[]))
    assert k1 == k2 and k1.startswith("raw:")


def test_job_key_identity_matches_in_run_dedupe(store: JobStore):
    # Two postings of the same role that the soft key would dedupe must resolve to
    # the same stored row (persistence identity == dedupe identity), even with a
    # differently-phrased remote location and a different apply URL.
    one = _job(location="Remote (United States)", apply_url="https://li/jobs/1")
    two = _job(location="Remote", apply_url="https://greenhouse/acme/2")
    assert job_key(one) == job_key(two)
    store.save_match(_match(one), now=NOW)
    store.save_match(_match(two), now=LATER)
    assert len(store.list_jobs()) == 1


# --------------------------------------------------------------------------- #
# Status pipeline — the crux: a re-ingest must NOT clobber a user-set status.
# --------------------------------------------------------------------------- #
def test_new_job_defaults_to_status_new(store: JobStore):
    store.save_match(_match(), now=NOW)
    got = store.get(job_key(_job()))
    assert got.status is ApplicationStatus.NEW
    assert got.status_updated_at is None  # untouched until the user advances it


def test_set_status_advances_and_stamps(store: JobStore):
    store.save_match(_match(), now=NOW)
    assert store.set_status(job_key(_job()), ApplicationStatus.APPLIED, now=LATER)
    got = store.get(job_key(_job()))
    assert got.status is ApplicationStatus.APPLIED
    assert got.status_updated_at == LATER
    assert got.last_seen_at == NOW  # status change does NOT touch ingestion time


def test_status_survives_reingest(store: JobStore):
    # THE crux of Slice D: mark a job APPLIED, then re-rank the mailbox (re-save
    # the freshly-scored job). Status, first_seen and status_updated_at are kept;
    # only the score/posting fields and last_seen refresh.
    job = _job()
    store.save_match(score_job(job, VP_AI_PROFILE, now=NOW), now=NOW)
    store.set_status(job_key(job), ApplicationStatus.APPLIED, now=NOW)

    # A later run re-sees the same job (perhaps with a fresher posting date).
    # The re-save is an update (returns False, not a fresh insert).
    refreshed = _job(posted_at=datetime(2026, 6, 26, tzinfo=timezone.utc))
    assert (
        store.save_match(score_job(refreshed, VP_AI_PROFILE, now=LATER), now=LATER)
        is False
    )
    got = store.get(job_key(job))
    assert got.status is ApplicationStatus.APPLIED  # NOT reset to NEW
    assert got.status_updated_at == NOW  # preserved
    assert got.first_seen_at == NOW  # preserved
    assert got.last_seen_at == LATER  # refreshed
    assert got.match.job.posted_at == datetime(2026, 6, 26, tzinfo=timezone.utc)


def test_set_status_on_missing_job_returns_false(store: JobStore):
    assert store.set_status("ghost|job|remote", ApplicationStatus.OFFER) is False


def test_free_transitions_any_to_any(store: JobStore):
    store.save_match(_match(), now=NOW)
    key = job_key(_job())
    # Jump straight to REJECTED, then back to INTERESTED — no enforced ordering.
    assert store.set_status(key, ApplicationStatus.REJECTED)
    assert store.get(key).status is ApplicationStatus.REJECTED
    assert store.set_status(key, ApplicationStatus.INTERESTED)
    assert store.get(key).status is ApplicationStatus.INTERESTED


# --------------------------------------------------------------------------- #
# Listing / filtering.
# --------------------------------------------------------------------------- #
def test_list_orders_by_score_desc(store: JobStore):
    low = _match(_job(title="Head of AI", company="Globex", location="New York"))
    high = _match(_job(title="VP of AI", company="Acme", location="Remote"))
    store.save_matches([low, high], now=NOW)
    listed = store.list_jobs()
    assert [sj.match.score for sj in listed] == sorted(
        [low.score, high.score], reverse=True
    )


def test_list_filters_by_status(store: JobStore):
    store.save_matches(
        [
            _match(_job(title="VP of AI", company="Acme")),
            _match(_job(title="Head of AI", company="Globex")),
        ],
        now=NOW,
    )
    store.set_status(
        job_key(_job(title="VP of AI", company="Acme")), ApplicationStatus.APPLIED
    )
    applied = store.list_jobs(status=ApplicationStatus.APPLIED)
    assert len(applied) == 1
    assert applied[0].match.job.company == "Acme"


def test_archived_hidden_by_default_shown_with_flag(store: JobStore):
    store.save_match(_match(), now=NOW)
    key = job_key(_job())
    store.set_status(key, ApplicationStatus.ARCHIVED)
    assert store.list_jobs() == []  # hidden
    assert len(store.list_jobs(include_archived=True)) == 1
    # Explicitly asking for the archived status still shows it.
    assert len(store.list_jobs(status=ApplicationStatus.ARCHIVED)) == 1


def test_list_min_tier_filters(store: JobStore):
    a_job = _job(title="VP of AI", company="Acme", location="Remote")  # tier A
    # A secondary (director-tier) title in a non-preferred geo lands in tier B.
    b_job = _job(title="Director of AI Strategy", company="Globex", location="New York")
    a_match, b_match = _match(a_job), _match(b_job)
    assert a_match.tier is Tier.A and b_match.tier is Tier.B  # guard the fixture
    store.save_matches([a_match, b_match], now=NOW)
    only_a = store.list_jobs(min_tier=Tier.A)
    assert all(sj.match.tier is Tier.A for sj in only_a)
    assert len(only_a) == 1


def test_list_limit_caps(store: JobStore):
    store.save_matches(
        [_match(_job(title=f"VP of AI {i}", company=f"Co{i}")) for i in range(5)],
        now=NOW,
    )
    assert len(store.list_jobs(limit=2)) == 2


def test_find_ids_by_prefix(store: JobStore):
    store.save_match(_match(), now=NOW)
    key = job_key(_job())
    assert store.find_ids(key[:4]) == [key]
    assert store.find_ids("zzz") == []


# --------------------------------------------------------------------------- #
# Migration — opening an older store file.
# --------------------------------------------------------------------------- #
def test_migrate_adds_missing_columns(tmp_path):
    # Build a jobs table missing the later status_updated_at column, then open it
    # with the current JobStore — _migrate must additively add the column rather
    # than crash with "no such column" (the Slice-6 lesson).
    from sqlalchemy import (
        JSON,
        Column,
        Float,
        MetaData,
        String,
        Table,
        create_engine,
    )

    db = tmp_path / "old.db"
    url = f"sqlite+pysqlite:///{db}"
    old_meta = MetaData()
    Table(
        "jobs",
        old_meta,
        Column("id", String, primary_key=True),
        Column("company", String, nullable=False),
        Column("normalized_title", String, nullable=False),
        Column("status", String, nullable=False),
        Column("score", Float, nullable=False),
        Column("tier", String, nullable=False),
        Column("location", String, nullable=True),
        Column("first_seen_at", String, nullable=False),
        Column("last_seen_at", String, nullable=False),
        Column("payload", JSON, nullable=False),
    )
    engine = create_engine(url)
    old_meta.create_all(engine)
    engine.dispose()

    store = JobStore(url)  # runs _migrate on construction
    cols = {c.name for c in jobs_table.columns}
    from sqlalchemy import inspect

    live = {c["name"] for c in inspect(store.engine).get_columns("jobs")}
    assert cols <= live  # every declared column now present
    # And the store works against the migrated file.
    store.save_match(_match(), now=NOW)
    assert store.get(job_key(_job())).status is ApplicationStatus.NEW


def test_dimensionscore_payload_omits_derived_contribution(store: JobStore):
    # contribution is a property; storing it would be redundant and could drift.
    store.save_match(_match(), now=NOW)
    with store.engine.connect() as conn:
        from sqlalchemy import select

        payload = conn.execute(select(jobs_table.c.payload)).first()[0]
    assert payload["dimensions"]
    assert all("contribution" not in d for d in payload["dimensions"])
