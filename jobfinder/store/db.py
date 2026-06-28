"""SQLAlchemy-Core backed store for Signals and Opportunities.

The store keeps two tables — ``signals`` and ``opportunities`` — each row
holding indexed scalar columns for querying plus a full JSON ``payload`` that
round-trips the pydantic model losslessly. The payload is the source of truth
on read (pydantic re-parses it, restoring tz-aware datetimes and nested
Evidence); the scalar columns exist only to filter and order without
deserialising every row.

Writes are an idempotent upsert keyed on the model id: a row's ``first_seen_at``
is stamped once on first insert and preserved on every later save, while
``updated_at`` advances. The upsert is a portable select-then-write rather than
a dialect-specific ``ON CONFLICT`` so the identical code path runs on SQLite
(tests, offline) and Postgres (production) — the engine is chosen purely by URL.
"""

from __future__ import annotations

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

from jobfinder.schemas import Opportunity, Signal

# In-memory SQLite needs a single shared connection or each checkout sees a
# fresh, empty database; StaticPool pins one connection for the engine's life.
IN_MEMORY_URL = "sqlite+pysqlite:///:memory:"

_metadata = MetaData()

# An ISO-8601 UTC string sorts lexically in true chronological order, so storing
# timestamps as text keeps ordering correct and identical across SQLite and
# Postgres without wading into each dialect's tz handling. Reconstruction of the
# real datetime happens from the JSON payload, not these columns.
signals_table = Table(
    "signals",
    _metadata,
    Column("id", String, primary_key=True),
    Column("company_id", String, nullable=False, index=True),
    Column("signal_type", String, nullable=False, index=True),
    Column("source", String, nullable=False),
    Column("observed_at", String, nullable=False, index=True),
    Column("effective_at", String, nullable=True),
    Column("first_seen_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("payload", JSON, nullable=False),
)

opportunities_table = Table(
    "opportunities",
    _metadata,
    Column("id", String, primary_key=True),
    Column("company_id", String, nullable=False, index=True),
    Column("opportunity_type", String, nullable=False),
    Column("score", Float, nullable=False, index=True),
    Column("status", String, nullable=False),
    # The score this row held *before* its most recent upsert. NULL on first
    # insert (nothing to compare to). It lets the Reporter show rank/score
    # movement across runs without snapshotting full history: the upsert keeps
    # one row per opportunity, so without carrying the prior score forward the
    # old value would be overwritten and lost. This is an audit scalar, not part
    # of the pydantic Opportunity (schemas.py stays pure).
    Column("previous_score", Float, nullable=True),
    Column("first_seen_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("payload", JSON, nullable=False),
)


@dataclass(frozen=True)
class OpportunityChange:
    """One opportunity in the current standings, annotated with cross-run facts.

    The annotations come straight from the store's own bookkeeping —
    ``first_seen_at`` (stamped once), ``updated_at`` (advances each save) and the
    carried-forward ``previous_score`` — so a Reporter can tell new from
    recurring and show score movement without ever re-deriving history from raw
    rows.
    """

    opportunity: Opportunity
    first_seen_at: datetime
    updated_at: datetime
    previous_score: float | None
    # first_seen_at >= the diff cutoff: this opportunity appeared this window.
    is_new: bool
    # updated_at >= the diff cutoff: this opportunity was (re)saved this window.
    changed_in_window: bool

    @property
    def score_delta(self) -> float | None:
        """How much the score moved on its last upsert, if there is a prior."""
        if self.previous_score is None:
            return None
        return round(self.opportunity.score - self.previous_score, 4)


@dataclass(frozen=True)
class SignalChange:
    """A signal that first appeared in the diff window, with its appearance time.

    Every entry in a ``StoreDiff.new_signals`` list is by construction new (the
    query only selects signals whose ``first_seen_at`` is on/after the cutoff),
    so there is no ``is_new`` flag — the list itself carries that meaning."""

    signal: Signal
    first_seen_at: datetime


@dataclass(frozen=True)
class StoreDiff:
    """A cross-run view of the store relative to a ``since`` cutoff.

    ``opportunities`` is the full current standings ranked best-first (each tagged
    new/recurring); ``new_signals`` are the signals that first appeared in the
    window. When ``since`` is None there is no baseline to diff against, so
    nothing is tagged new and ``new_signals`` is empty — a plain ranked digest.
    """

    since: datetime | None
    opportunities: list[OpportunityChange]
    new_signals: list[SignalChange]


@dataclass(frozen=True)
class PersistResult:
    """How a save changed the store, split by table and insert vs. update.

    ``updated`` rows are the ones a cross-run diff cares about: a signal seen
    again on a later run is recurring, not new.
    """

    signals_inserted: int = 0
    signals_updated: int = 0
    opportunities_inserted: int = 0
    opportunities_updated: int = 0

    @property
    def total(self) -> int:
        return (
            self.signals_inserted
            + self.signals_updated
            + self.opportunities_inserted
            + self.opportunities_updated
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    """Normalise to a UTC ISO-8601 string so stored timestamps sort uniformly.

    A naive datetime is assumed to already be UTC (the codebase produces
    tz-aware UTC everywhere, but be defensive); an aware one is converted.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


# The single source of truth for how each model maps onto its table and its
# indexed scalar columns. Both the single-item (`save_*`) and batch
# (`persist_run`) write paths resolve through `_row_spec`, so adding a queryable
# column means editing exactly one place.
_TABLES_BY_NAME = {
    "signals": signals_table,
    "opportunities": opportunities_table,
}


def _signal_scalars(signal: Signal) -> dict:
    return {
        "company_id": signal.company_id,
        "signal_type": signal.signal_type,
        "source": signal.source,
        "observed_at": _iso(signal.observed_at),
        "effective_at": _iso(signal.effective_at) if signal.effective_at else None,
    }


def _opportunity_scalars(opportunity: Opportunity) -> dict:
    return {
        "company_id": opportunity.company_id,
        "opportunity_type": opportunity.opportunity_type,
        "score": opportunity.score,
        "status": opportunity.status,
    }


def _row_spec(model: Signal | Opportunity) -> tuple[Table, dict]:
    """Resolve a model to (its table, its indexed scalar columns)."""
    if isinstance(model, Signal):
        return signals_table, _signal_scalars(model)
    return opportunities_table, _opportunity_scalars(model)


class Store:
    """Durable home for Signals and Opportunities.

    Construct directly from a SQLAlchemy URL::

        Store("sqlite+pysqlite:///jobfinder.db")          # offline / local file
        Store("postgresql+psycopg://user:pw@host/db")     # production

    or use :meth:`in_memory` for an ephemeral test store. Tables are created on
    construction when ``create=True`` (the default).
    """

    def __init__(self, url: str = IN_MEMORY_URL, *, create: bool = True):
        connect_args: dict = {}
        engine_kwargs: dict = {}
        if url.startswith("sqlite"):
            # pysqlite forbids using a connection from a thread other than the
            # one that created it; relax that for every SQLite URL so the store
            # works from a thread pool / web worker, not just in-memory.
            connect_args["check_same_thread"] = False
            if ":memory:" in url:
                # An in-memory DB lives only as long as its connection, so all
                # checkouts must share one connection or each sees an empty DB.
                # Keyed on ":memory:" (not the exact constant) so a differently
                # spelled in-memory URL doesn't silently lose writes.
                engine_kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(
            url, connect_args=connect_args, **engine_kwargs
        )
        if create:
            self.create_all()

    @classmethod
    def in_memory(cls) -> Store:
        """An ephemeral SQLite store living entirely in process memory."""
        return cls(IN_MEMORY_URL)

    def create_all(self) -> None:
        _metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a store was first created.

        ``create_all`` only issues ``CREATE TABLE IF NOT EXISTS`` — it never
        alters a table that already exists, so a DB file written by an earlier
        slice keeps its old column set. Without this, opening a pre-Slice-6
        store and calling :meth:`diff` (or re-persisting, which hits the
        ``previous_score`` UPDATE) raises ``OperationalError: no such column``.
        We additively ``ALTER TABLE ... ADD COLUMN`` any declared column the
        live table is missing — portable across SQLite and Postgres and safe to
        run every construction (we only add what's absent).
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
    def save_signal(self, signal: Signal, *, now: datetime | None = None) -> bool:
        """Upsert one signal. Returns True if it was newly inserted."""
        return self.save(signal, now=now)

    def save_opportunity(
        self, opportunity: Opportunity, *, now: datetime | None = None
    ) -> bool:
        """Upsert one opportunity. Returns True if it was newly inserted."""
        return self.save(opportunity, now=now)

    def save(self, model: Signal | Opportunity, *, now: datetime | None = None) -> bool:
        """Upsert one Signal or Opportunity. Returns True if newly inserted."""
        now = now or _utcnow()
        with self.engine.begin() as conn:
            return self._write(conn, model, now)

    def persist_run(
        self,
        signals: list[Signal],
        opportunities: list[Opportunity],
        *,
        now: datetime | None = None,
    ) -> PersistResult:
        """Persist a whole pipeline run (signals + ranked opportunities).

        Everything lands in a single transaction so a half-written run can't
        leave the store inconsistent. A shared ``now`` stamps every row in the
        run identically.
        """
        now = now or _utcnow()
        s_ins = s_upd = o_ins = o_upd = 0
        with self.engine.begin() as conn:
            for signal in signals:
                if self._write(conn, signal, now):
                    s_ins += 1
                else:
                    s_upd += 1
            for opp in opportunities:
                if self._write(conn, opp, now):
                    o_ins += 1
                else:
                    o_upd += 1
        return PersistResult(
            signals_inserted=s_ins,
            signals_updated=s_upd,
            opportunities_inserted=o_ins,
            opportunities_updated=o_upd,
        )

    # ------------------------------------------------------------------ #
    # Reads.
    # ------------------------------------------------------------------ #
    def get_signal(self, signal_id: str) -> Signal | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(signals_table.c.payload).where(signals_table.c.id == signal_id)
            ).first()
        return Signal.model_validate(row[0]) if row else None

    def get_opportunity(self, opportunity_id: str) -> Opportunity | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(opportunities_table.c.payload).where(
                    opportunities_table.c.id == opportunity_id
                )
            ).first()
        return Opportunity.model_validate(row[0]) if row else None

    def signals_for_company(self, company_id: str) -> list[Signal]:
        """All signals for a company, newest observation first."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(signals_table.c.payload)
                .where(signals_table.c.company_id == company_id)
                .order_by(signals_table.c.observed_at.desc())
            ).all()
        return [Signal.model_validate(r[0]) for r in rows]

    def opportunities_for_company(self, company_id: str) -> list[Opportunity]:
        """All opportunities for a company, highest score first."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(opportunities_table.c.payload)
                .where(opportunities_table.c.company_id == company_id)
                .order_by(opportunities_table.c.score.desc())
            ).all()
        return [Opportunity.model_validate(r[0]) for r in rows]

    def top_opportunities(self, limit: int | None = None) -> list[Opportunity]:
        """Every stored opportunity ranked best-first (optionally capped)."""
        stmt = select(opportunities_table.c.payload).order_by(
            opportunities_table.c.score.desc()
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return [Opportunity.model_validate(r[0]) for r in rows]

    def diff(self, *, since: datetime | None = None) -> StoreDiff:
        """Cross-run diff of the whole store relative to ``since``.

        Returns the current opportunity standings (ranked best-first, each tagged
        new vs recurring and carrying its prior score) plus the signals that
        first appeared on or after ``since``. With ``since=None`` there is no
        baseline, so nothing is flagged new — the caller gets a plain ranked
        digest. This is the one query a Reporter needs; it keeps all the
        timestamp/score bookkeeping in the store rather than the Reporter.
        """
        since_iso = _iso(since) if since is not None else None
        opp_changes: list[OpportunityChange] = []
        sig_changes: list[SignalChange] = []
        with self.engine.connect() as conn:
            opp_rows = conn.execute(
                select(
                    opportunities_table.c.payload,
                    opportunities_table.c.previous_score,
                    opportunities_table.c.first_seen_at,
                    opportunities_table.c.updated_at,
                ).order_by(opportunities_table.c.score.desc())
            ).all()
            for payload, prev_score, first_seen_at, updated_at in opp_rows:
                opp_changes.append(
                    OpportunityChange(
                        opportunity=Opportunity.model_validate(payload),
                        first_seen_at=datetime.fromisoformat(first_seen_at),
                        updated_at=datetime.fromisoformat(updated_at),
                        previous_score=prev_score,
                        is_new=since_iso is not None and first_seen_at >= since_iso,
                        changed_in_window=since_iso is not None
                        and updated_at >= since_iso,
                    )
                )

            sig_stmt = select(
                signals_table.c.payload, signals_table.c.first_seen_at
            ).order_by(signals_table.c.observed_at.desc())
            # Only the genuinely new signals are interesting in a "what changed"
            # view; with no baseline there is nothing new to report.
            if since_iso is not None:
                sig_stmt = sig_stmt.where(signals_table.c.first_seen_at >= since_iso)
                sig_rows = conn.execute(sig_stmt).all()
                for payload, first_seen_at in sig_rows:
                    sig_changes.append(
                        SignalChange(
                            signal=Signal.model_validate(payload),
                            first_seen_at=datetime.fromisoformat(first_seen_at),
                        )
                    )
        return StoreDiff(
            since=since, opportunities=opp_changes, new_signals=sig_changes
        )

    def first_seen(self, table: str, row_id: str) -> datetime | None:
        """When a signal/opportunity id was first persisted (for cross-run diffs)."""
        try:
            tbl = _TABLES_BY_NAME[table]
        except KeyError:
            raise ValueError(
                f"unknown table {table!r}; expected one of {sorted(_TABLES_BY_NAME)}"
            ) from None
        with self.engine.connect() as conn:
            row = conn.execute(
                select(tbl.c.first_seen_at).where(tbl.c.id == row_id)
            ).first()
        return datetime.fromisoformat(row[0]) if row else None

    # ------------------------------------------------------------------ #
    # Internals.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write(conn, model: Signal | Opportunity, now: datetime) -> bool:
        """Insert or update one model on `conn`, preserving first_seen_at.

        Returns True if the row was newly inserted. A select-then-write upsert
        (rather than a dialect-specific ON CONFLICT) keeps one code path across
        SQLite and Postgres.
        """
        table, scalars = _row_spec(model)
        now_iso = _iso(now)
        is_opp = table is opportunities_table
        # For opportunities, also read the current score so we can carry it into
        # `previous_score` on update — that is what makes rank/score movement
        # derivable across runs from a single upserted row.
        cols = [table.c.first_seen_at]
        if is_opp:
            cols.append(table.c.score)
        existing = conn.execute(select(*cols).where(table.c.id == model.id)).first()
        values = {
            **scalars,
            "payload": model.model_dump(mode="json"),
            "updated_at": now_iso,
        }
        if existing is None:
            conn.execute(
                table.insert().values(id=model.id, first_seen_at=now_iso, **values)
            )
            return True
        if is_opp:
            # existing == (first_seen_at, score); preserve the pre-update score.
            values["previous_score"] = existing[1]
        conn.execute(table.update().where(table.c.id == model.id).values(**values))
        return False
