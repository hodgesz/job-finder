"""ATS wiring through the pipeline and CLI live mode (offline)."""

from datetime import datetime, timedelta, timezone

import pytest

from jobfinder.cli import _ats_companies, _parse_ats_spec, build_parser, main
from jobfinder.pipeline import CompanyInputs, run_pipeline_detailed
from jobfinder.scoring import score_company
from jobfinder.sources.ats import AtsClient, JobBoard, JobPosting

NOW = datetime(2026, 6, 28, tzinfo=timezone.utc)


def _board() -> JobBoard:
    postings = [
        JobPosting(
            id=f"e{i}",
            title=("Founding Engineer" if i == 0 else f"Engineer {i}"),
            department="Engineering",
            updated_at=NOW - timedelta(days=3),
            url=f"https://jobs.example.com/e{i}",
        )
        for i in range(5)
    ]
    return JobBoard(
        provider="greenhouse",
        token="acme",
        url="https://boards.greenhouse.io/acme",
        postings=postings,
    )


def test_pipeline_activates_hiring_and_strategic_components():
    company = CompanyInputs(company_id="co-ats", name="ATS Co", ats_boards=[_board()])
    result = run_pipeline_detailed([company], observed_at=NOW, now=NOW)
    assert result.opportunities
    signals = result.signals
    # The board produced ATS signals feeding both previously-dormant components.
    breakdown = score_company("co-ats", signals, now=NOW)
    assert breakdown.component("hiring_velocity").raw > 0
    assert breakdown.component("strategic_language").raw > 0
    # And those components actually contribute to the composite score.
    assert breakdown.component("hiring_velocity").contribution > 0


def test_parse_ats_spec_valid_and_invalid():
    assert _parse_ats_spec("greenhouse:stripe") == ("greenhouse", "stripe")
    assert _parse_ats_spec("LEVER:Netflix") == ("lever", "Netflix")
    with pytest.raises(ValueError, match="provider:token"):
        _parse_ats_spec("greenhouse")
    with pytest.raises(ValueError, match="provider:token"):
        _parse_ats_spec("workday:acme")
    with pytest.raises(ValueError, match="provider:token"):
        _parse_ats_spec("greenhouse:")


def test_ats_companies_builds_from_injected_client():
    client = AtsClient(lambda _url: '{"jobs": []}')
    companies = _ats_companies(client, ["greenhouse:acme"])
    assert len(companies) == 1
    assert companies[0].company_id == "ats-greenhouse-acme"
    assert companies[0].ats_boards[0].provider == "greenhouse"


def test_live_requires_at_least_one_source(capsys):
    code = main(["live", "--user-agent", "job-finder test@example.com"])
    assert code == 2
    err = capsys.readouterr().err
    assert "at least one --cik or --ats" in err


def test_live_accepts_ats_without_cik():
    # --cik is no longer required; --ats alone is a valid source.
    parser = build_parser()
    args = parser.parse_args(
        ["live", "--ats", "greenhouse:acme", "--user-agent", "x@example.com"]
    )
    assert args.cik == []
    assert args.ats == ["greenhouse:acme"]
