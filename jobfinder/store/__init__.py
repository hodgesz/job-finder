"""Persistence for job-finder (Slice 3).

A thin durable store for `Signal`s and `Opportunity`s so that runs *accumulate*
rather than print-and-forget. Accumulation is the prerequisite for the
cross-run diffing the ATS hiring-velocity pillar (Pillar I) needs later: to say
"this company opened five new junior reqs since last week" you must have last
week's snapshot on disk.

Design choices that mirror the rest of the codebase:

- **Schema-first, framework-light.** Built on SQLAlchemy *Core* (not the ORM)
  so the table definitions read like the `@dataclass` records elsewhere and the
  domain models stay pure pydantic — `jobfinder.schemas` does not import this
  module.
- **One backend abstraction, two homes.** The same `Store` runs against SQLite
  (the stdlib driver — tests and offline use) and Postgres (production) chosen
  purely by connection URL, keeping the injectable/offline-testable pattern the
  EDGAR client established.
- **Idempotent upsert keyed on the model id.** Signal/Opportunity ids are
  deterministic (`<accession>:departure`, `opp:<company_id>`), so re-running a
  pipeline re-saves the *same* rows instead of duplicating them. Each row also
  carries a `first_seen_at` that is set once and preserved across re-saves —
  the concrete hook cross-run diffing will read to tell new from recurring.
"""

from __future__ import annotations

from jobfinder.store.db import (
    OpportunityChange,
    PersistResult,
    SignalChange,
    Store,
    StoreDiff,
)

__all__ = [
    "OpportunityChange",
    "PersistResult",
    "SignalChange",
    "Store",
    "StoreDiff",
]
