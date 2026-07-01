"""SQLAlchemy-Core persistence for the job-search CRM (Slice D).

Makes ``rank`` runs *stateful*: canonical jobs and their latest match assessment
are saved to a local DB so the tool remembers what it has seen across runs, and
each job can be tracked through a manual application pipeline
(``ApplicationStatus``: NEW → INTERESTED → APPLIED → INTERVIEWING → OFFER /
REJECTED / ARCHIVED). Re-seeing a job updates it *in place* — its posting/score
fields refresh while the user-set status and ``first_seen_at`` are preserved — so
re-ranking a mailbox never resets a job already marked APPLIED.

This deliberately reuses the *pattern* of the core ``jobfinder.store.db.Store``
(SQLAlchemy-Core, one table of indexed scalar columns plus a full JSON
``payload``, an idempotent select-then-write upsert that runs identically on
SQLite and Postgres, ISO-8601 UTC text timestamps that sort chronologically,
``StaticPool`` for in-memory SQLite tests). It does NOT import or share the core
store: this is the detour's own table in the jobsearch namespace, fully
decoupled — the only core coupling in the whole detour stays ``sources/ats.py``.

The one real difference from core: the core store round-trips *pydantic* models,
so the payload re-parses itself on read. The detour's domain models are
*dataclasses* (decoupled from core pydantic), so this module supplies its own
typed (de)serialization (:func:`_match_to_payload` / :func:`_match_from_payload`)
that restores tz-aware datetimes, the ``Source``/``Tier`` enums, and the nested
``RawPosting``/``DimensionScore``/``LlmRerank`` objects faithfully.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from jobfinder.jobsearch.models import (
    ApplicationStatus,
    CanonicalJob,
    Contact,
    ContactRole,
    ContactSource,
    DimensionScore,
    DraftStatus,
    JobMatch,
    LlmRerank,
    OutreachEmail,
    RawPosting,
    Source,
    Tier,
)
from jobfinder.jobsearch.email_format import normalize_domain, parent_domains
from jobfinder.jobsearch.normalize import job_key

# In-memory SQLite needs a single shared connection or each checkout sees a
# fresh, empty database; StaticPool pins one connection for the engine's life.
IN_MEMORY_URL = "sqlite+pysqlite:///:memory:"

_metadata = MetaData()

# One table for the CRM. Indexed scalar columns exist only to filter/order
# without deserialising every row; the JSON ``payload`` is the source of truth on
# read (it round-trips the full JobMatch dataclass, nested job included).
#
# Timestamps are ISO-8601 UTC *text* (sorts lexically in true chronological order
# and stays identical across SQLite and Postgres without wading into dialect tz
# handling). ``first_seen_at`` is stamped once on insert and preserved;
# ``last_seen_at`` advances on every re-ingest; ``status_updated_at`` is set only
# when the user changes status (NULL while a job is still its born-NEW state).
jobs_table = Table(
    "jobs",
    _metadata,
    Column("id", String, primary_key=True),  # the stable job_key
    Column("company", String, nullable=False, index=True),
    Column("normalized_title", String, nullable=False, index=True),
    Column("status", String, nullable=False, index=True),
    Column("score", Float, nullable=False, index=True),
    Column("tier", String, nullable=False, index=True),
    Column("location", String, nullable=True),
    Column("first_seen_at", String, nullable=False),
    Column("last_seen_at", String, nullable=False, index=True),
    Column("status_updated_at", String, nullable=True),
    Column("payload", JSON, nullable=False),
)

# Contacts discovered for a job (Slice E). Decoupled from the ``jobs`` table
# schema (it is NOT entangled): a contact references a job by its stable
# ``job_key`` in ``job_id`` but the two tables are independent, so a contact can
# exist for a job that was never persisted (the user may record a contact from a
# manual review before saving the rank run). The composite primary key
# (job_id, contact_key) makes recording the same person twice idempotent; the
# full ``Contact`` dataclass round-trips through the JSON ``payload``.
contacts_table = Table(
    "contacts",
    _metadata,
    Column("job_id", String, primary_key=True),
    Column("contact_key", String, primary_key=True),  # normalized name|domain
    Column("name", String, nullable=False),
    Column("company", String, nullable=False, index=True),
    Column("role", String, nullable=False, index=True),
    Column("email_domain", String, nullable=True),
    Column("first_seen_at", String, nullable=False),
    Column("last_seen_at", String, nullable=False),
    Column("payload", JSON, nullable=False),
)

# The do-not-contact list (Slice E) — ALWAYS honored wherever a contact or email
# guess could surface. An entry is either a full email address or a bare domain
# (``kind`` = "email" | "domain"); a domain entry suppresses every address at that
# domain. The normalized ``value`` is the primary key so adding the same entry
# twice is idempotent.
do_not_contact_table = Table(
    "do_not_contact",
    _metadata,
    Column("value", String, primary_key=True),  # normalized email or domain
    Column("kind", String, nullable=False),  # "email" | "domain"
    Column("reason", String, nullable=True),
    Column("added_at", String, nullable=False),
)

# Assembled outreach drafts (Slice F) — the draft-and-approve store. A draft is
# the assembled email (subject + body + truthful sender identity + opt-out) for a
# job + recipient; it is born ``DRAFTED`` (nothing sent) and only the explicit
# ``outreach send <id> --confirm`` path flips it to ``SENT``. Decoupled from the
# ``jobs``/``contacts`` tables (NOT entangled): a draft references a job by its
# stable ``job_id`` but stands alone. The primary key ``id`` is a stable
# :func:`draft_key` over (job_id, recipient) so re-assembling the same outreach
# updates in place rather than piling up duplicates. ``sent_at`` is NULL until
# the draft is sent. The full ``OutreachEmail`` round-trips through ``payload``.
drafts_table = Table(
    "drafts",
    _metadata,
    Column("id", String, primary_key=True),  # stable draft_key(job_id, to_email)
    Column("job_id", String, nullable=False, index=True),
    Column("to_email", String, nullable=False, index=True),
    Column("status", String, nullable=False, index=True),  # drafted | sent
    Column("subject", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("sent_at", String, nullable=True),
    Column("payload", JSON, nullable=False),
)


@dataclass(frozen=True)
class StoredContact:
    """A persisted contact plus its bookkeeping timestamps."""

    contact: Contact
    first_seen_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True)
class DoNotContactEntry:
    """One do-not-contact suppression: an email or a whole domain."""

    value: str
    kind: str  # "email" | "domain"
    reason: str | None = None


@dataclass(frozen=True)
class StoredDraft:
    """A persisted outreach draft plus its draft-and-approve bookkeeping.

    ``email`` is the full assembled ``OutreachEmail``; ``status`` is DRAFTED until
    the explicit send step flips it to SENT (stamping ``sent_at``). ``id`` is the
    stable :func:`draft_key`, used by the CLI to reference a draft for send.
    """

    id: str
    job_id: str
    email: OutreachEmail
    status: DraftStatus
    created_at: datetime
    sent_at: datetime | None


@dataclass(frozen=True)
class StoredJob:
    """A persisted job: its latest assessment plus the CRM bookkeeping.

    ``match`` is the full ``JobMatch`` (with its nested ``CanonicalJob``) as last
    ranked; ``status`` is the user-driven pipeline state (preserved across
    re-ingests); the timestamps come straight from the store's own columns.
    """

    match: JobMatch
    status: ApplicationStatus
    first_seen_at: datetime
    last_seen_at: datetime
    status_updated_at: datetime | None


@dataclass(frozen=True)
class SaveResult:
    """How a save changed the store: jobs newly inserted vs. updated in place."""

    inserted: int = 0
    updated: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated


# Lowest-to-highest tier ordering so a ``min_tier`` filter is a simple comparison.
_TIER_RANK = {Tier.C: 0, Tier.B: 1, Tier.A: 2}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    """Normalise to a UTC ISO-8601 string for the *scalar* timestamp columns.

    Used only for the bookkeeping columns (first_seen/last_seen/status_updated)
    so they sort uniformly. A naive datetime is assumed already-UTC; an aware one
    is converted. (Datetimes *inside* the payload are stored verbatim by
    :func:`_iso_opt` to preserve their original tz-awareness on round-trip.)
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _iso_opt(value: datetime | None) -> str | None:
    """Serialise a payload datetime *verbatim* (faithful round-trip), or None.

    ``datetime.isoformat()`` + ``datetime.fromisoformat()`` round-trips both naive
    and tz-aware values losslessly — a posting date that arrived tz-naive
    (a LinkedIn ``-0000`` Date header) stays naive, an aware ATS timestamp stays
    aware — so we do NOT UTC-normalise here (unlike :func:`_iso`)."""
    return value.isoformat() if value is not None else None


def _dt_opt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _escape_like(fragment: str) -> str:
    """Escape SQL LIKE metacharacters (``\\``, ``%``, ``_``) for a literal match.

    Used by :meth:`JobStore.find_ids` so a user-supplied id fragment is matched
    verbatim rather than as a wildcard pattern. The backslash escape char must be
    escaped first so it doesn't double-escape the ``%``/``_`` we add after it.
    """
    return fragment.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# --------------------------------------------------------------------------- #
# Dataclass (de)serialization for the JSON payload.
#
# The detour's models are dataclasses, not pydantic, so unlike the core store we
# can't lean on ``model_dump``/``model_validate``. These typed converters restore
# the enums, the tz-awareness of every datetime, and the nested objects exactly.
# DimensionScore.contribution / CanonicalJob.source_kinds are derived properties,
# so they are recomputed on read, never stored.
# --------------------------------------------------------------------------- #
def _raw_to_dict(raw: RawPosting) -> dict:
    return {
        "title": raw.title,
        "company": raw.company,
        "source": raw.source.value,
        "url": raw.url,
        "source_job_id": raw.source_job_id,
        "location": raw.location,
        "workplace_type": raw.workplace_type,
        "department": raw.department,
        "posted_at": _iso_opt(raw.posted_at),
        "snippet": raw.snippet,
        "alert_keyword": raw.alert_keyword,
    }


def _raw_from_dict(d: dict) -> RawPosting:
    return RawPosting(
        title=d["title"],
        company=d["company"],
        source=Source(d["source"]),
        url=d.get("url"),
        source_job_id=d.get("source_job_id"),
        location=d.get("location"),
        workplace_type=d.get("workplace_type"),
        department=d.get("department"),
        posted_at=_dt_opt(d.get("posted_at")),
        snippet=d.get("snippet"),
        alert_keyword=d.get("alert_keyword"),
    )


def _job_to_dict(job: CanonicalJob) -> dict:
    return {
        "company": job.company,
        "title": job.title,
        "normalized_title": job.normalized_title,
        "location": job.location,
        "workplace_type": job.workplace_type,
        "department": job.department,
        "best_apply_url": job.best_apply_url,
        "posted_at": _iso_opt(job.posted_at),
        "sources": [_raw_to_dict(r) for r in job.sources],
    }


def _job_from_dict(d: dict) -> CanonicalJob:
    return CanonicalJob(
        company=d["company"],
        title=d["title"],
        normalized_title=d["normalized_title"],
        location=d.get("location"),
        workplace_type=d.get("workplace_type"),
        department=d.get("department"),
        best_apply_url=d.get("best_apply_url"),
        posted_at=_dt_opt(d.get("posted_at")),
        sources=[_raw_from_dict(r) for r in d.get("sources", [])],
    )


def _dim_to_dict(dim: DimensionScore) -> dict:
    # contribution is a property (raw * weight) — recomputed on read, not stored.
    return {
        "name": dim.name,
        "raw": dim.raw,
        "weight": dim.weight,
        "reason": dim.reason,
    }


def _dim_from_dict(d: dict) -> DimensionScore:
    return DimensionScore(
        name=d["name"], raw=d["raw"], weight=d["weight"], reason=d["reason"]
    )


def _llm_to_dict(llm: LlmRerank | None) -> dict | None:
    if llm is None:
        return None
    return {"rank": llm.rank, "relevance": llm.relevance, "rationale": llm.rationale}


def _llm_from_dict(d: dict | None) -> LlmRerank | None:
    if d is None:
        return None
    return LlmRerank(rank=d["rank"], relevance=d["relevance"], rationale=d["rationale"])


def _match_to_payload(match: JobMatch) -> dict:
    return {
        "job": _job_to_dict(match.job),
        "score": match.score,
        "tier": match.tier.value,
        "reason": match.reason,
        "dimensions": [_dim_to_dict(d) for d in match.dimensions],
        "risks": list(match.risks),
        "rejected": match.rejected,
        "llm": _llm_to_dict(match.llm),
    }


def _match_from_payload(d: dict) -> JobMatch:
    return JobMatch(
        job=_job_from_dict(d["job"]),
        score=d["score"],
        tier=Tier(d["tier"]),
        reason=d["reason"],
        dimensions=[_dim_from_dict(x) for x in d.get("dimensions", [])],
        risks=list(d.get("risks", [])),
        rejected=d.get("rejected", False),
        llm=_llm_from_dict(d.get("llm")),
    )


def _contact_to_payload(contact: Contact) -> dict:
    return {
        "name": contact.name,
        "company": contact.company,
        "role": contact.role.value,
        "title": contact.title,
        "linkedin_url": contact.linkedin_url,
        "email_domain": contact.email_domain,
        "source": contact.source.value,
        "notes": contact.notes,
    }


def _contact_from_payload(d: dict) -> Contact:
    return Contact(
        name=d["name"],
        company=d["company"],
        role=ContactRole(d.get("role", ContactRole.OTHER.value)),
        title=d.get("title"),
        linkedin_url=d.get("linkedin_url"),
        email_domain=d.get("email_domain"),
        source=ContactSource(d.get("source", ContactSource.MANUAL.value)),
        notes=d.get("notes"),
    )


def _email_to_payload(email: OutreachEmail) -> dict:
    return {
        "to_email": email.to_email,
        "to_name": email.to_name,
        "subject": email.subject,
        "body": email.body,
        "from_name": email.from_name,
        "from_email": email.from_email,
        "company": email.company,
        "job_title": email.job_title,
        "opt_out": email.opt_out,
        "tailoring": email.tailoring,
    }


def _email_from_payload(d: dict) -> OutreachEmail:
    return OutreachEmail(
        to_email=d["to_email"],
        to_name=d["to_name"],
        subject=d["subject"],
        body=d["body"],
        from_name=d["from_name"],
        from_email=d["from_email"],
        company=d["company"],
        job_title=d["job_title"],
        opt_out=d["opt_out"],
        tailoring=d.get("tailoring", "template"),
    )


def draft_key(job_id: str, to_email: str) -> str:
    """Stable per-(job, recipient) identity for an outreach draft.

    Re-assembling outreach for the same job + recipient updates the stored draft
    in place rather than duplicating. The recipient address is normalised
    (lower-cased, domain-normalised) so ``Jane@Acme.com`` and ``jane@acme.com``
    key identically. A short BLAKE2b digest keeps the id compact and free of
    characters that would complicate CLI fragment matching, while the job_id +
    recipient pair guarantees no cross-job collision. Falls back to the stripped
    lower-case address if it can't be normalised."""
    local, _, dom = to_email.strip().lower().partition("@")
    norm_dom = normalize_domain(dom) or dom
    recipient = f"{local}@{norm_dom}" if local else to_email.strip().lower()
    digest = hashlib.blake2b(
        f"{job_id}\x00{recipient}".encode(), digest_size=10
    ).hexdigest()
    return f"d_{digest}"


def contact_key(contact: Contact) -> str:
    """Stable per-job identity for a contact: normalized name + email domain.

    Recording the same person (same name, same domain) twice updates in place
    rather than inserting a duplicate. The domain is included so two distinct
    people who happen to share a name at different companies don't collide. Both
    halves use the SAME normalization the rest of the tool uses — the name is
    whitespace-collapsed lower-case, the domain runs through
    :func:`normalize_domain` — so a contact recorded as ``acme.com`` and again as
    ``https://acme.com/careers`` keys identically and updates in place rather than
    duplicating (an unparseable domain falls back to its stripped lower-case form).
    """
    name = " ".join(contact.name.lower().split())
    raw_domain = (contact.email_domain or "").strip().lower()
    domain = normalize_domain(raw_domain) or raw_domain
    return f"{name}|{domain}"


def _dnc_normalize(value: str) -> tuple[str, str] | None:
    """Normalize a do-not-contact entry to ``(normalized_value, kind)`` or None.

    An entry containing ``@`` is treated as a full email (normalized to its
    lower-cased local@domain); anything else is treated as a bare domain
    (normalized via :func:`normalize_domain`). Returns ``None`` for an
    unparseable/empty value so the caller can reject it cleanly.
    """
    text = value.strip().lower()
    if not text:
        return None
    if "@" in text:
        local, _, dom = text.partition("@")
        norm_dom = normalize_domain(dom)
        if not local or norm_dom is None:
            return None
        return f"{local}@{norm_dom}", "email"
    norm_dom = normalize_domain(text)
    return (norm_dom, "domain") if norm_dom else None


class JobStore:
    """Durable home for ranked jobs and their application-pipeline status.

    Construct directly from a SQLAlchemy URL::

        JobStore("sqlite+pysqlite:///jobsearch.db")        # local file
        JobStore("postgresql+psycopg://user:pw@host/db")   # if ever needed

    or use :meth:`in_memory` for an ephemeral test store. Tables are created on
    construction when ``create=True`` (the default). Mirrors the core ``Store``'s
    engine selection so the identical code path runs on SQLite and Postgres.
    """

    def __init__(self, url: str = IN_MEMORY_URL, *, create: bool = True):
        connect_args: dict = {}
        engine_kwargs: dict = {}
        if url.startswith("sqlite"):
            # pysqlite forbids using a connection from another thread; relax that
            # for every SQLite URL (matches the core store).
            connect_args["check_same_thread"] = False
            if ":memory:" in url:
                # An in-memory DB lives only as long as its connection, so all
                # checkouts must share one. Keyed on ":memory:" (not the exact
                # constant) so a differently-spelled in-memory URL still works.
                engine_kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(
            url, connect_args=connect_args, **engine_kwargs
        )
        if create:
            self.create_all()

    @classmethod
    def in_memory(cls) -> JobStore:
        """An ephemeral SQLite store living entirely in process memory."""
        return cls(IN_MEMORY_URL)

    def create_all(self) -> None:
        _metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """Additively add columns introduced after a store file was created.

        ``create_all`` only issues ``CREATE TABLE IF NOT EXISTS`` — it never
        alters an existing table — so a DB written by an earlier build keeps its
        old column set. Any new indexed/scalar column needs this or opening an old
        file raises ``OperationalError: no such column`` (the lesson the core
        store learned in Slice 6). Portable across SQLite/Postgres, idempotent.
        """
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        with self.engine.begin() as conn:
            for table in _metadata.tables.values():
                if table.name not in existing_tables:
                    continue  # create_all already made it with every column.
                present = {c["name"] for c in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name in present:
                        continue
                    coltype = column.type.compile(self.engine.dialect)
                    conn.execute(
                        text(
                            f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {coltype}'
                        )
                    )

    # ------------------------------------------------------------------ #
    # Writes.
    # ------------------------------------------------------------------ #
    def save_match(self, match: JobMatch, *, now: datetime | None = None) -> bool:
        """Upsert one ranked job. Returns True if it was newly inserted.

        Keyed on the stable :func:`job_key`. On re-ingest the posting/score fields
        refresh and ``last_seen_at`` advances, but the user-set ``status``,
        ``status_updated_at`` and ``first_seen_at`` are preserved — a job marked
        APPLIED stays APPLIED when its mailbox is re-ranked.
        """
        now = now or _utcnow()
        with self.engine.begin() as conn:
            return self._write(conn, match, now)

    def save_matches(
        self, matches: list[JobMatch], *, now: datetime | None = None
    ) -> SaveResult:
        """Persist a whole ranked run in one transaction (atomic).

        A shared ``now`` stamps every row in the run identically.
        """
        now = now or _utcnow()
        inserted = updated = 0
        with self.engine.begin() as conn:
            for match in matches:
                if self._write(conn, match, now):
                    inserted += 1
                else:
                    updated += 1
        return SaveResult(inserted=inserted, updated=updated)

    def set_status(
        self,
        job_id: str,
        status: ApplicationStatus,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Set a job's pipeline status. Returns True if a row was updated.

        Free transitions: any status to any other (single-user personal CRM, no
        enforced ordering). Stamps ``status_updated_at``; does NOT touch
        ``last_seen_at`` (that records ingestion, not triage). Returns False if no
        job with that id exists, so the caller can report a clear error.
        """
        now = now or _utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(
                jobs_table.update()
                .where(jobs_table.c.id == job_id)
                .values(status=status.value, status_updated_at=_iso(now))
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------ #
    # Reads.
    # ------------------------------------------------------------------ #
    def get(self, job_id: str) -> StoredJob | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    jobs_table.c.payload,
                    jobs_table.c.status,
                    jobs_table.c.first_seen_at,
                    jobs_table.c.last_seen_at,
                    jobs_table.c.status_updated_at,
                ).where(jobs_table.c.id == job_id)
            ).first()
        return self._to_stored(row) if row else None

    def list_jobs(
        self,
        *,
        status: ApplicationStatus | None = None,
        min_tier: Tier | None = None,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[StoredJob]:
        """Persisted jobs, highest score first.

        ``status`` filters to exactly that pipeline state. With no status filter,
        ARCHIVED jobs are hidden unless ``include_archived`` is set (ARCHIVED is
        the "hide from the active list" state). ``min_tier`` keeps only jobs at or
        above the given tier.
        """
        stmt = select(
            jobs_table.c.payload,
            jobs_table.c.status,
            jobs_table.c.first_seen_at,
            jobs_table.c.last_seen_at,
            jobs_table.c.status_updated_at,
        ).order_by(jobs_table.c.score.desc())
        if status is not None:
            stmt = stmt.where(jobs_table.c.status == status.value)
        elif not include_archived:
            stmt = stmt.where(jobs_table.c.status != ApplicationStatus.ARCHIVED.value)
        if min_tier is not None:
            allowed = [
                t.value for t, r in _TIER_RANK.items() if r >= _TIER_RANK[min_tier]
            ]
            stmt = stmt.where(jobs_table.c.tier.in_(allowed))
        if limit is not None:
            stmt = stmt.limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [self._to_stored(r) for r in rows]

    def find_ids(self, prefix: str) -> list[str]:
        """All stored job ids starting with ``prefix`` (for CLI id resolution).

        The job_key is a composite string; this lets the ``status`` subcommand
        accept an unambiguous leading fragment instead of the whole quoted key.

        The fragment is matched as a LITERAL prefix: ``%`` and ``_`` (SQL LIKE
        wildcards) are escaped so a fragment like ``%`` or one containing ``_``
        can't match rows it shouldn't (which, with a single stored job, would let
        ``status`` mutate it despite no real prefix being supplied).
        """
        pattern = _escape_like(prefix) + "%"
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(jobs_table.c.id)
                .where(jobs_table.c.id.like(pattern, escape="\\"))
                .order_by(jobs_table.c.id)
            ).all()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ #
    # Contacts (Slice E).
    # ------------------------------------------------------------------ #
    def save_contact(
        self, job_id: str, contact: Contact, *, now: datetime | None = None
    ) -> bool:
        """Upsert one contact for a job. Returns True if newly inserted.

        Keyed on ``(job_id, contact_key)`` so re-recording the same person updates
        their details in place rather than duplicating. ``first_seen_at`` is
        stamped once; ``last_seen_at`` advances each time.
        """
        now = now or _utcnow()
        now_iso = _iso(now)
        key = contact_key(contact)
        refreshed = {
            "name": contact.name,
            "company": contact.company,
            "role": contact.role.value,
            "email_domain": contact.email_domain,
            "last_seen_at": now_iso,
            "payload": _contact_to_payload(contact),
        }
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(contacts_table.c.job_id).where(
                    (contacts_table.c.job_id == job_id)
                    & (contacts_table.c.contact_key == key)
                )
            ).first()
            if existing is None:
                conn.execute(
                    contacts_table.insert().values(
                        job_id=job_id,
                        contact_key=key,
                        first_seen_at=now_iso,
                        **refreshed,
                    )
                )
                return True
            conn.execute(
                contacts_table.update()
                .where(
                    (contacts_table.c.job_id == job_id)
                    & (contacts_table.c.contact_key == key)
                )
                .values(**refreshed)
            )
            return False

    def list_contacts(self, job_id: str) -> list[StoredContact]:
        """All contacts recorded for a job, in first-seen order."""
        stmt = (
            select(
                contacts_table.c.payload,
                contacts_table.c.first_seen_at,
                contacts_table.c.last_seen_at,
            )
            .where(contacts_table.c.job_id == job_id)
            .order_by(contacts_table.c.first_seen_at, contacts_table.c.contact_key)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [
            StoredContact(
                contact=_contact_from_payload(payload),
                first_seen_at=datetime.fromisoformat(first_seen),
                last_seen_at=datetime.fromisoformat(last_seen),
            )
            for payload, first_seen, last_seen in rows
        ]

    # ------------------------------------------------------------------ #
    # Do-not-contact list (always honored).
    # ------------------------------------------------------------------ #
    def add_do_not_contact(
        self, value: str, *, reason: str | None = None, now: datetime | None = None
    ) -> DoNotContactEntry | None:
        """Add an email or domain to the do-not-contact list (idempotent).

        Returns the normalized :class:`DoNotContactEntry` as actually persisted, or
        ``None`` if ``value`` couldn't be parsed as an email or a domain. Re-adding
        an existing entry with a new ``reason`` updates it (and keeps the original
        ``added_at``); re-adding with no reason leaves the stored reason intact —
        and the returned entry reflects that stored reason, not the (absent) passed
        one, so the return value never disagrees with what ``list_do_not_contact``
        reports.
        """
        normalized = _dnc_normalize(value)
        if normalized is None:
            return None
        norm_value, kind = normalized
        now_iso = _iso(now or _utcnow())
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(do_not_contact_table.c.reason).where(
                    do_not_contact_table.c.value == norm_value
                )
            ).first()
            if existing is None:
                conn.execute(
                    do_not_contact_table.insert().values(
                        value=norm_value, kind=kind, reason=reason, added_at=now_iso
                    )
                )
                stored_reason = reason
            elif reason is not None:
                conn.execute(
                    do_not_contact_table.update()
                    .where(do_not_contact_table.c.value == norm_value)
                    .values(reason=reason)
                )
                stored_reason = reason
            else:
                # No new reason supplied — keep (and report) the existing one.
                stored_reason = existing[0]
        return DoNotContactEntry(value=norm_value, kind=kind, reason=stored_reason)

    def list_do_not_contact(self) -> list[DoNotContactEntry]:
        """Every do-not-contact entry, value-sorted."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    do_not_contact_table.c.value,
                    do_not_contact_table.c.kind,
                    do_not_contact_table.c.reason,
                ).order_by(do_not_contact_table.c.value)
            ).all()
        return [DoNotContactEntry(value=v, kind=k, reason=r) for v, k, r in rows]

    def is_suppressed(self, email: str) -> bool:
        """True if ``email`` (or its domain / a parent domain) is do-not-contact.

        A full-email entry suppresses exactly that address; a domain entry
        suppresses every address at that domain AND its sub-domains (blocking
        "acme.com" blocks "careers.acme.com"). This is the always-honored guard
        every contact/email surface applies before emitting a guess — and
        ``guess_emails`` itself applies it at the producer, so a suppressed address
        is never constructed in the first place.
        """
        normalized = _dnc_normalize(email)
        if normalized is None:
            return False
        norm_value, kind = normalized
        # The set of stored values that would suppress this address: the address
        # itself (when it's an email) plus its domain and every parent domain
        # (so a domain entry covers sub-domains). One query covers all of them.
        domain = norm_value.split("@", 1)[1] if kind == "email" else norm_value
        candidates = {norm_value, *parent_domains(domain)}
        with self.engine.connect() as conn:
            row = conn.execute(
                select(do_not_contact_table.c.value).where(
                    do_not_contact_table.c.value.in_(candidates)
                )
            ).first()
        return row is not None

    # ------------------------------------------------------------------ #
    # Outreach drafts (Slice F — draft-and-approve).
    # ------------------------------------------------------------------ #
    def save_draft(
        self, job_id: str, email: OutreachEmail, *, now: datetime | None = None
    ) -> StoredDraft:
        """Upsert an outreach draft for a job + recipient. Returns the StoredDraft.

        Keyed on :func:`draft_key` ``(job_id, to_email)`` so re-assembling outreach
        for the same recipient refreshes the subject/body in place rather than
        duplicating.

        CRUCIAL: a draft that was ALREADY SENT is NOT overwritten. Re-running
        ``outreach draft`` for a recipient you already emailed returns the existing
        SENT record UNCHANGED (its ``status``/``sent_at``/body preserved) so the
        caller can see it was already contacted and warn. This protects two things
        at once — the "already contacted" record is never erased, and the send
        gate's ``status is SENT`` re-send guard can't be silently re-armed by a
        re-draft. A DRAFTED (not-yet-sent) draft refreshes in place as before.
        """
        now = now or _utcnow()
        now_iso = _iso(now)
        row_id = draft_key(job_id, email.to_email)
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(drafts_table.c.status, drafts_table.c.created_at).where(
                    drafts_table.c.id == row_id
                )
            ).first()
            if existing is not None and existing[0] == DraftStatus.SENT.value:
                # Already sent — do not overwrite. Return the sent record as-is.
                sent_row = conn.execute(
                    select(
                        drafts_table.c.id,
                        drafts_table.c.job_id,
                        drafts_table.c.payload,
                        drafts_table.c.status,
                        drafts_table.c.created_at,
                        drafts_table.c.sent_at,
                    ).where(drafts_table.c.id == row_id)
                ).first()
                return self._to_stored_draft(sent_row)
            values = {
                "job_id": job_id,
                "to_email": email.to_email,
                "status": DraftStatus.DRAFTED.value,
                "subject": email.subject,
                "sent_at": None,
                "payload": _email_to_payload(email),
            }
            if existing is None:
                conn.execute(
                    drafts_table.insert().values(
                        id=row_id, created_at=now_iso, **values
                    )
                )
                created_iso = now_iso
            else:
                # DRAFTED → refresh in place; preserve the original created_at.
                conn.execute(
                    drafts_table.update()
                    .where(drafts_table.c.id == row_id)
                    .values(**values)
                )
                created_iso = existing[1]
        return StoredDraft(
            id=row_id,
            job_id=job_id,
            email=email,
            status=DraftStatus.DRAFTED,
            created_at=datetime.fromisoformat(created_iso),
            sent_at=None,
        )

    def get_draft(self, draft_id: str) -> StoredDraft | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    drafts_table.c.id,
                    drafts_table.c.job_id,
                    drafts_table.c.payload,
                    drafts_table.c.status,
                    drafts_table.c.created_at,
                    drafts_table.c.sent_at,
                ).where(drafts_table.c.id == draft_id)
            ).first()
        return self._to_stored_draft(row) if row else None

    def list_drafts(self, *, status: DraftStatus | None = None) -> list[StoredDraft]:
        """All outreach drafts, newest first; optionally filtered by status."""
        stmt = select(
            drafts_table.c.id,
            drafts_table.c.job_id,
            drafts_table.c.payload,
            drafts_table.c.status,
            drafts_table.c.created_at,
            drafts_table.c.sent_at,
        ).order_by(drafts_table.c.created_at.desc(), drafts_table.c.id)
        if status is not None:
            stmt = stmt.where(drafts_table.c.status == status.value)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [self._to_stored_draft(r) for r in rows]

    def find_draft_ids(self, prefix: str) -> list[str]:
        """All draft ids starting with ``prefix`` (for CLI id resolution).

        Mirrors :meth:`find_ids`: the fragment is matched as a LITERAL prefix with
        SQL LIKE wildcards escaped, so a ``%``/``_`` fragment can't match drafts it
        shouldn't (which, with a single stored draft, would let ``send`` target it
        despite no real prefix being supplied)."""
        pattern = _escape_like(prefix) + "%"
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(drafts_table.c.id)
                .where(drafts_table.c.id.like(pattern, escape="\\"))
                .order_by(drafts_table.c.id)
            ).all()
        return [r[0] for r in rows]

    def mark_sent(self, draft_id: str, *, now: datetime | None = None) -> bool:
        """Flip a draft to SENT, stamping ``sent_at``. Returns True if a row changed.

        Called by the CLI *after* the gmail.send seam confirms the send, so the
        store records what actually went out. Returns False if the draft id no
        longer exists, so the caller can report it rather than claim a false send.
        """
        now = now or _utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(
                drafts_table.update()
                .where(drafts_table.c.id == draft_id)
                .values(status=DraftStatus.SENT.value, sent_at=_iso(now))
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------ #
    # Internals.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_stored_draft(row) -> StoredDraft:
        draft_id, job_id, payload, status, created_at, sent_at = row
        return StoredDraft(
            id=draft_id,
            job_id=job_id,
            email=_email_from_payload(payload),
            status=DraftStatus(status),
            created_at=datetime.fromisoformat(created_at),
            sent_at=_dt_opt(sent_at),
        )

    @staticmethod
    def _to_stored(row) -> StoredJob:
        payload, status, first_seen, last_seen, status_updated = row
        return StoredJob(
            match=_match_from_payload(payload),
            status=ApplicationStatus(status),
            first_seen_at=datetime.fromisoformat(first_seen),
            last_seen_at=datetime.fromisoformat(last_seen),
            status_updated_at=_dt_opt(status_updated),
        )

    @staticmethod
    def _write(conn, match: JobMatch, now: datetime) -> bool:
        """Insert or update one match on ``conn``. Returns True if newly inserted.

        Select-then-write (not a dialect-specific ON CONFLICT) so one code path
        covers SQLite and Postgres. On update the user-set ``status`` /
        ``status_updated_at`` / ``first_seen_at`` are left untouched — only the
        posting/score fields and ``last_seen_at`` refresh.
        """
        job = match.job
        row_id = job_key(job)
        now_iso = _iso(now)
        # Fields that always refresh from the latest ranking.
        refreshed = {
            "company": job.company,
            "normalized_title": job.normalized_title,
            "score": match.score,
            "tier": match.tier.value,
            "location": job.location,
            "last_seen_at": now_iso,
            "payload": _match_to_payload(match),
        }
        existing = conn.execute(
            select(jobs_table.c.id).where(jobs_table.c.id == row_id)
        ).first()
        if existing is None:
            conn.execute(
                jobs_table.insert().values(
                    id=row_id,
                    status=ApplicationStatus.NEW.value,
                    first_seen_at=now_iso,
                    status_updated_at=None,
                    **refreshed,
                )
            )
            return True
        # Update in place: status / status_updated_at / first_seen_at preserved.
        conn.execute(
            jobs_table.update().where(jobs_table.c.id == row_id).values(**refreshed)
        )
        return False
