"""Derive a company's firmographics from data we already fetch.

Slice 8 added the firmographic fit model (``jobfinder.fit``) but wired it only
into the demo path: live runs supplied no ``Firmographics``, so every real
company fell back to the neutral ``0.5`` ``company_fit`` placeholder. This module
closes that gap *without a new network dependency or API key* — it derives the
firmographics from structured facts the live SEC fetch already returns:

- **Sector** from the filer's SIC description in the submissions index header
  (``CompanyInfo.sic_description``, e.g. "Industrial Instruments For
  Measurement"), falling back to the coarser Form D ``industry_group`` when no
  SIC is on file. Sector is the heaviest fit dimension (weight 0.5), so deriving
  it alone meaningfully lifts a live run off the placeholder.
- **Funding stage** and **headcount** are *not* present in SEC filings, so they
  stay ``None``. The fit model already scores a missing dimension as neutral
  (never punitive), so a derived firmographic with sector-only data is honest:
  it improves the dimension we can know and leaves the rest untouched. A richer
  enrichment source (a network client behind its own injectable fetcher) could
  populate stage/size later — deliberately out of scope here, mirroring how SEC
  came before any enrichment API.

Design, consistent with the rest of the codebase:

- **Pure and offline.** This module does no fetching of its own. The caller
  hands it the already-fetched ``CompanyInfo`` / ``FormD`` records (themselves
  produced behind ``EdgarClient``'s injected fetcher), exactly as the signal
  modules take already-fetched filings. So it is fully unit-testable with no
  network and no clock.
- **Decoupled.** Like the signal modules, this is an *input* to the pipeline,
  not imported by ``scoring`` or ``schemas`` (which stays the pure wire
  contract — firmographics ride on ``CompanyInputs``, never on ``Company``).
"""

from __future__ import annotations

from collections.abc import Iterable

from jobfinder.fit import Firmographics
from jobfinder.sources.edgar import CompanyInfo, FormD


def _clean(value: str | None) -> str | None:
    """Trim a free-text label; map empty/whitespace to None."""
    if value is None:
        return None
    text = value.strip()
    return text or None


def firmographics_from_sec(
    company_info: CompanyInfo,
    form_d: Iterable[FormD] = (),
) -> Firmographics | None:
    """Derive ``Firmographics`` from already-fetched SEC records.

    The sector is taken from the filer's SIC description when available, else
    the (coarser) ``industry_group`` of the first Form D that carries one — a
    filer may have several Form D filings and only some disclose an industry, so
    we scan rather than trusting the first blindly. Funding stage and headcount
    are not disclosed in SEC filings, so they remain ``None`` (scored neutral by
    the fit model). Returns ``None`` when neither source yields a sector — there
    is nothing to assess, so the caller keeps the literal ``company_fit``
    fallback rather than constructing an all-empty firmographic.
    """
    sector = _clean(company_info.sic_description)
    if sector is None:
        sector = next(
            (group for fd in form_d if (group := _clean(fd.industry_group))), None
        )
    if sector is None:
        return None
    return Firmographics(sector=sector)
