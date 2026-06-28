"""Tests for deriving firmographics from already-fetched SEC records (offline)."""

from jobfinder.fit import Firmographics
from jobfinder.sources.edgar import CompanyInfo, FormD
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
