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
- **Funding stage** is partly free: the submissions header lists the stock
  ``exchanges`` the filer is listed on, and a non-empty list means the company
  is *exchange-listed* — i.e. its stage is effectively "public", the last step
  of the fit model's stage progression. So a live SEC registrant gets an honest,
  free stage with no extra call or key. (A private SEC filer — many Form D
  issuers — carries no exchange, so we leave the stage unknown rather than
  guess.) The private-company stage (seed / Series A / …) still isn't in any
  free feed.
- **Headcount** is *not* in SEC filings at all.

The two SEC-blind dimensions (a *private* company's funding stage, and every
company's headcount) are filled by an optional, injectable
``EnrichmentClient`` (``jobfinder.sources.enrichment``). The default is a
no-op, so absent a bound vendor this module behaves exactly as it did before:
sector (+ a free "public" stage for listed filers), everything else neutral.
The fit model already scores a missing dimension as neutral (never punitive),
so partial data is always honest.

**Merge precedence.** The SEC-derived facts are authoritative for a registrant
and take precedence; enrichment only *fills the gaps* it leaves:

- *Sector* — SEC SIC description, else the first Form D ``industry_group``.
  Enrichment never supplies sector (SEC's classification is more authoritative
  for a registrant than a third-party guess).
- *Funding stage* — "public" when the filer is exchange-listed (SEC wins), else
  whatever enrichment supplies.
- *Headcount* — enrichment only (SEC discloses none).

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
from jobfinder.sources.enrichment import Enrichment

# Funding stage assigned to an exchange-listed SEC filer. Matches the last step
# of the fit model's ``_STAGE_ORDER`` so a candidate targeting "public" scores it
# exactly and one targeting an earlier round scores it as far (not neutral).
PUBLIC_STAGE = "public"


def _clean(value: str | None) -> str | None:
    """Trim a free-text label; map empty/whitespace to None."""
    if value is None:
        return None
    text = value.strip()
    return text or None


def firmographics_from_sec(
    company_info: CompanyInfo,
    form_d: Iterable[FormD] = (),
    *,
    enrichment: Enrichment = Enrichment(),
) -> Firmographics | None:
    """Derive ``Firmographics`` from already-fetched SEC records + enrichment.

    Sector is the SIC description when available, else the (coarser)
    ``industry_group`` of the first Form D that carries one — a filer may have
    several Form D filings and only some disclose an industry, so we scan rather
    than trusting the first blindly. Funding stage is "public" when the filer is
    exchange-listed (free, from the submissions header); otherwise it, and the
    headcount, come from the optional ``enrichment`` (empty by default → both
    stay ``None``, scored neutral by the fit model). The SEC-derived facts take
    precedence; enrichment only fills the gaps they leave.

    Returns ``None`` only when *no* dimension could be derived from any source —
    there is nothing to assess, so the caller keeps the literal ``company_fit``
    fallback rather than constructing an all-empty firmographic.
    """
    sector = _clean(company_info.sic_description)
    if sector is None:
        sector = next(
            (group for fd in form_d if (group := _clean(fd.industry_group))), None
        )

    # Exchange-listed => "public" stage (SEC wins); else fall back to enrichment.
    funding_stage = PUBLIC_STAGE if company_info.exchanges else enrichment.funding_stage
    # SEC discloses no headcount, so this is enrichment-only.
    employee_count = enrichment.employee_count

    if sector is None and funding_stage is None and employee_count is None:
        return None
    return Firmographics(
        sector=sector,
        funding_stage=funding_stage,
        employee_count=employee_count,
    )
