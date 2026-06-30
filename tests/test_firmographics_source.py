"""Tests for deriving firmographics from already-fetched SEC records (offline)."""

from jobfinder.fit import Firmographics
from jobfinder.sources.edgar import CompanyInfo, FormD
from jobfinder.sources.enrichment import Enrichment
from jobfinder.sources.firmographics import firmographics_from_sec


def test_sector_from_sic_description():
    info = CompanyInfo(
        cik="0000320193",
        name="Apple Inc.",
        sic="3571",
        sic_description="Electronic Computers",
    )
    firmo = firmographics_from_sec(info)
    assert firmo == Firmographics(sector="Electronic Computers")
    # SEC discloses no stage/headcount, so those stay neutral-by-omission.
    assert firmo.funding_stage is None
    assert firmo.employee_count is None


def _form_d(industry_group: str | None = None, acc: str = "acc-1") -> FormD:
    return FormD(
        issuer_cik="1",
        issuer_name="X",
        accession_number=acc,
        industry_group=industry_group,
    )


def test_form_d_industry_group_is_fallback_when_no_sic():
    info = CompanyInfo(cik="1", name="Northwind", sic=None, sic_description=None)
    firmo = firmographics_from_sec(info, form_d=[_form_d("Manufacturing")])
    assert firmo == Firmographics(sector="Manufacturing")


def test_form_d_fallback_scans_for_first_with_industry_group():
    # A filer with no SIC and several Form D filings, only a later one carrying
    # an industry group: scanning the list (not just the first) finds it, instead
    # of falsely reporting no sector.
    info = CompanyInfo(cik="1", name="X", sic=None, sic_description=None)
    form_d = [_form_d(None, acc="a"), _form_d("Pooled Investment Fund", acc="b")]
    firmo = firmographics_from_sec(info, form_d=form_d)
    assert firmo is not None
    assert firmo.sector == "Pooled Investment Fund"


def test_sic_description_wins_over_form_d_industry_group():
    # The SIC sector is more specific than Form D's coarse industry group, so it
    # takes precedence when both are present.
    info = CompanyInfo(
        cik="1", name="X", sic="3559", sic_description="Special Industry Machinery"
    )
    firmo = firmographics_from_sec(info, form_d=[_form_d("Manufacturing")])
    assert firmo.sector == "Special Industry Machinery"


def test_no_sector_anywhere_returns_none():
    # Neither source yields a sector -> None, so the caller keeps the literal
    # company_fit fallback instead of building an all-empty firmographic.
    info = CompanyInfo(cik="1", name="X")
    assert firmographics_from_sec(info) is None
    # A Form D with no industry group is no better than nothing.
    assert firmographics_from_sec(info, form_d=[_form_d(None)]) is None


def test_blank_sector_strings_are_ignored():
    # Whitespace-only labels are not a real sector.
    info = CompanyInfo(cik="1", name="X", sic_description="   ")
    assert firmographics_from_sec(info, form_d=[_form_d("  ")]) is None


# --------------------------------------------------------------------------- #
# Slice 17: free "public" stage for exchange-listed filers + enrichment merge.
# --------------------------------------------------------------------------- #
def test_exchange_listed_filer_is_public_stage():
    # A non-empty exchanges list means the filer is exchange-listed -> its stage
    # is "public" (the last step of the fit model's progression), derived for free
    # from the submissions header.
    info = CompanyInfo(
        cik="320193",
        name="Apple Inc.",
        sic_description="Electronic Computers",
        exchanges=("Nasdaq",),
    )
    firmo = firmographics_from_sec(info)
    assert firmo == Firmographics(sector="Electronic Computers", funding_stage="public")


def test_private_filer_has_no_public_stage():
    # A filer with no exchange listing (e.g. a Form D private issuer) leaves the
    # stage unknown -> None, never assumed. Sector still derives from Form D.
    info = CompanyInfo(cik="1", name="Northwind", exchanges=())
    firmo = firmographics_from_sec(info, form_d=[_form_d("Manufacturing")])
    assert firmo == Firmographics(sector="Manufacturing")
    assert firmo.funding_stage is None


def test_enrichment_fills_stage_and_size_when_not_exchange_listed():
    # A private filer's stage and headcount come from enrichment (the SEC-blind
    # dimensions); the SEC sector is still authoritative.
    info = CompanyInfo(cik="1", name="Acme", sic_description="Special Machinery")
    firmo = firmographics_from_sec(
        info,
        enrichment=Enrichment(funding_stage="series_b", employee_count=180),
    )
    assert firmo == Firmographics(
        sector="Special Machinery", funding_stage="series_b", employee_count=180
    )


def test_exchange_public_stage_wins_over_enrichment_stage():
    # Merge precedence: the SEC-derived "public" stage is authoritative for a
    # registrant, so it wins even if enrichment guesses a (stale) private stage.
    # Enrichment's headcount still fills the dimension SEC can't supply.
    info = CompanyInfo(
        cik="320193",
        name="Apple Inc.",
        sic_description="Electronic Computers",
        exchanges=("Nasdaq",),
    )
    firmo = firmographics_from_sec(
        info,
        enrichment=Enrichment(funding_stage="series_c", employee_count=164000),
    )
    assert firmo.funding_stage == "public"  # SEC wins
    assert firmo.employee_count == 164000  # enrichment fills the gap


def test_enrichment_only_firmographic_with_no_sector_is_returned():
    # No sector anywhere, but enrichment supplies stage/size: there IS something
    # to assess, so we return a (sector-less) firmographic rather than None.
    info = CompanyInfo(cik="1", name="X")
    firmo = firmographics_from_sec(
        info, enrichment=Enrichment(funding_stage="seed", employee_count=12)
    )
    assert firmo is not None
    assert firmo.sector is None
    assert firmo.funding_stage == "seed"
    assert firmo.employee_count == 12


def test_all_sources_empty_still_returns_none():
    # Nothing from SEC, no exchange, empty enrichment -> nothing to assess.
    info = CompanyInfo(cik="1", name="X")
    assert firmographics_from_sec(info, enrichment=Enrichment()) is None
