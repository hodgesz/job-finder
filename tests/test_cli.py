"""Tests for the pipeline wiring and the CLI demo (offline)."""

from datetime import date, datetime, timezone

from jobfinder.cli import _demo_companies, build_parser, main, render
from jobfinder.pipeline import CompanyInputs, run_pipeline
from jobfinder.signals.extraction import RegexExtractor
from jobfinder.sources.edgar import Filing, FormD

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _run(companies):
    # Force the deterministic regex extractor so tests never touch a live LLM,
    # mirroring what the `demo` CLI command does.
    return run_pipeline(companies, observed_at=NOW, now=NOW, extractor=RegexExtractor())


def test_run_pipeline_ranks_funding_plus_vacuum_first():
    companies = _demo_companies()
    opps = _run(companies)
    # Northwind has both a funding signal and a CFO vacuum -> ranks first.
    assert opps[0].company_id == "co-northwind"
    assert opps[0].score == max(o.score for o in opps)
    # Every opportunity cites at least one supporting signal (schema rule).
    assert all(o.supporting_signal_ids for o in opps)


def test_pipeline_company_with_no_signals_is_dropped():
    empty = CompanyInputs(company_id="co-empty", name="Empty Co")
    funded_filing = Filing(
        cik="1",
        accession_number="acc-1",
        form="D",
        filing_date=date(2026, 5, 1),
        report_date=None,
        items=[],
        primary_document="primary_doc.xml",
    )
    funded = CompanyInputs(
        company_id="co-funded",
        name="Funded Co",
        form_d=[
            (
                funded_filing,
                FormD(
                    issuer_cik="1",
                    issuer_name="Funded Co",
                    accession_number="acc-1",
                    total_amount_sold=10_000_000.0,
                    total_remaining=0.0,
                ),
            )
        ],
    )
    opps = _run([empty, funded])
    assert [o.company_id for o in opps] == ["co-funded"]


def test_demo_dataset_produces_three_ranked_opportunities():
    opps = _run(_demo_companies())
    ids = [o.company_id for o in opps]
    assert ids == ["co-northwind", "co-lumen", "co-atlas"]


def test_demo_northwind_is_funding_plus_vacuum():
    # The flagship demo case: a fresh raise AND a CFO departure with no named
    # successor. Under the deterministic regex extractor this is reproducible.
    opps = _run(_demo_companies())
    northwind = next(o for o in opps if o.company_id == "co-northwind")
    assert "open search" in northwind.why_now
    assert "Form D capital raised" in northwind.why_now


def test_render_includes_evidence_and_why_now():
    opps = _run(_demo_companies())
    out = render(opps, _demo_companies(), top=5)
    assert "Northwind Robotics Inc." in out
    assert "Why now:" in out
    assert "Evidence (supporting signals):" in out
    # The funding accession appears as cited evidence.
    assert "0001950000-26-000003:form_d" in out


def test_render_respects_top_n():
    opps = _run(_demo_companies())
    out = render(opps, _demo_companies(), top=1)
    assert "1. Northwind" in out
    assert "Lumen Bio Corp." not in out


def test_render_handles_no_opportunities():
    out = render([], [], top=5)
    assert "No qualifying opportunities found." in out


def test_main_demo_prints_report(capsys):
    code = main(["demo"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Top 3 companies" in out
    assert "Northwind Robotics Inc." in out


def test_main_demo_top_flag(capsys):
    code = main(["--top", "1", "demo"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Northwind" in out
    assert "Atlas Freight" not in out


def test_live_requires_user_agent():
    parser = build_parser()
    # argparse exits with code 2 when a required arg is missing.
    try:
        parser.parse_args(["live", "--cik", "320193"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for missing --user-agent")
