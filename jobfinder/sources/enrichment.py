"""Firmographic enrichment seam — funding stage & headcount from outside SEC.

The fit model (``jobfinder.fit``) scores three dimensions: sector, funding
stage and headcount. SEC filings give us the sector (SIC) for free, and being
exchange-listed tells us the stage is "public" (see ``firmographics``). But the
*private*-company funding stage (seed / Series A / …) and the headcount are not
in any free structured feed we already fetch — they live behind commercial
enrichment APIs (Clearbit, PeopleDataLabs, Crunchbase, …).

Rather than bind one of those vendors today — and add a paid dependency + an API
key + a network call to a code path that is otherwise free and hermetic — this
module ships the *seam*: an injectable ``EnrichmentClient`` that the live
assembler consults to fill the two SEC-blind dimensions. The default
``NullEnrichmentClient`` returns nothing, so:

- **CI stays hermetic.** No client is bound by default, so a live run (and every
  test) makes no enrichment network call and behaves exactly as it did before
  this slice — sector-only firmographics, stage/size neutral (unless the company
  is exchange-listed, which is derived for free upstream).
- **A real vendor is a drop-in.** A future ``ClearbitEnrichmentClient`` (or
  similar) implements the same ``enrich`` protocol behind its own injected
  ``Fetcher`` and degrades to ``None`` without an API key — mirroring how
  ``GeminiExtractor`` is strictly opt-in. Nothing else has to change: the merge
  precedence and the wiring already exist.

Design, consistent with ``EdgarClient`` / ``AtsClient``:

- **Network access is injected** (a ``Fetcher`` callable), so a concrete client
  is fully unit-testable offline against a fake fetcher.
- **Decoupled.** Like the other source modules, this is an *input* to the
  pipeline (consumed by the CLI assembler), not imported by ``scoring``,
  ``fit`` or ``schemas`` (which stays the pure wire contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Enrichment:
    """The two firmographic dimensions SEC filings don't disclose.

    Both are optional: an enrichment source may surface one, both, or neither,
    and an absent field is simply not merged (the fit model scores a missing
    dimension as neutral, never punitive). Sector is deliberately absent here —
    SEC's SIC description already covers it and is more authoritative for a
    registrant than a third-party guess.
    """

    funding_stage: str | None = None
    employee_count: int | None = None


@runtime_checkable
class EnrichmentClient(Protocol):
    """Looks up a company's funding stage and headcount.

    Implementations take whatever identifiers they need (CIK and/or name) and
    return an ``Enrichment``; they should return an empty ``Enrichment`` (rather
    than raise) when they have no data or aren't configured, so the caller can
    treat "no enrichment" uniformly.
    """

    def enrich(self, *, cik: str | None, name: str) -> Enrichment: ...


class NullEnrichmentClient:
    """The default: no enrichment source bound, so nothing is ever filled.

    Keeps live runs free and hermetic (no network call, no vendor, no key). Used
    until a concrete vendor client is wired in; with it, ``firmographics_from_sec``
    behaves exactly as it did before this slice.
    """

    def enrich(self, *, cik: str | None, name: str) -> Enrichment:
        return Enrichment()
