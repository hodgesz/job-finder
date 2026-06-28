"""Tests for the pipeline wiring and the CLI demo (offline)."""

from datetime import date, datetime, timezone

from jobfinder.cli import (
    _demo_companies,
    _parse_since,
    _store_url,
    build_parser,
    main,
    render,
)
from jobfinder.pipeline import CompanyInputs, run_pipeline, run_pipeline_detailed
from jobfinder.signals.extraction import RegexExtractor
from jobfinder.sources.edgar import Filing, FormD
from jobfinder.store import Store

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


def test_demo_dataset_produces_ranked_opportunities():
    opps = _run(_demo_companies())
    ids = [o.company_id for o in opps]
    # Northwind (all four pillars) leads; Helix is the ATS-only company.
    assert ids == ["co-northwind", "co-lumen", "co-helix", "co-atlas"]


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
    assert "Top 4 companies" in out
    assert "Northwind Robotics Inc." in out


def test_main_demo_top_flag(capsys):
    code = main(["--top", "1", "demo"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Northwind" in out
    assert "Atlas Freight" not in out


def test_run_pipeline_detailed_returns_signals_and_opportunities():
    result = run_pipeline_detailed(
        _demo_companies(), observed_at=NOW, now=NOW, extractor=RegexExtractor()
    )
    # Same ranking as run_pipeline, plus the raw signals that produced it.
    assert [o.company_id for o in result.opportunities] == [
        "co-northwind",
        "co-lumen",
        "co-helix",
        "co-atlas",
    ]
    assert result.signals
    # Every cited supporting signal is present in the returned signal list.
    cited = {sid for o in result.opportunities for sid in o.supporting_signal_ids}
    assert cited <= {s.id for s in result.signals}


def test_store_url_maps_path_vs_scheme():
    assert _store_url("runs.db") == "sqlite+pysqlite:///runs.db"
    assert _store_url("/tmp/jf.db") == "sqlite+pysqlite:////tmp/jf.db"
    # An explicit URL passes through untouched.
    url = "postgresql+psycopg://u:p@host/db"
    assert _store_url(url) == url


def test_main_demo_persists_to_sqlite(tmp_path, capsys):
    db = tmp_path / "runs.db"
    code = main(["demo", "--db", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Persisted to {db}" in out
    assert "signals" in out and "opportunities" in out

    # Re-open the same file and confirm the run actually landed.
    store = Store(_store_url(str(db)), create=False)
    opps = store.top_opportunities()
    assert [o.company_id for o in opps] == [
        "co-northwind",
        "co-lumen",
        "co-helix",
        "co-atlas",
    ]
    assert store.signals_for_company("co-northwind")


def test_main_demo_rerun_updates_not_duplicates(tmp_path, capsys):
    db = tmp_path / "runs.db"
    main(["demo", "--db", str(db)])
    capsys.readouterr()
    # Second identical run: everything upserts, nothing duplicates.
    main(["demo", "--db", str(db)])
    out = capsys.readouterr().out
    assert "updated" in out
    store = Store(_store_url(str(db)), create=False)
    # Four companies -> exactly four opportunities, even after two runs.
    assert len(store.top_opportunities()) == 4


def test_main_demo_without_db_does_not_persist(capsys):
    code = main(["demo"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Persisted to" not in out


def test_parse_since_handles_bare_date_and_naive_datetime():
    # A bare date -> midnight UTC.
    assert _parse_since("2026-06-01") == datetime(2026, 6, 1, tzinfo=timezone.utc)
    # A naive datetime is assumed UTC.
    assert _parse_since("2026-06-01T12:00:00") == datetime(
        2026, 6, 1, 12, tzinfo=timezone.utc
    )


def test_main_report_reads_persisted_db(tmp_path, capsys):
    db = tmp_path / "runs.db"
    main(["demo", "--db", str(db)])
    capsys.readouterr()
    code = main(["report", "--db", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Opportunity digest — current standings" in out
    assert "co-northwind" in out
    # Evidence citations survive from persistence into the digest.
    assert "0001950000-26-000003:form_d" in out


def test_main_report_since_flags_new(tmp_path, capsys):
    db = tmp_path / "runs.db"
    main(["demo", "--db", str(db)])
    capsys.readouterr()
    # Everything was just written, so a cutoff in the past flags all as new.
    code = main(["report", "--db", str(db), "--since", "2026-01-01"])
    assert code == 0
    out = capsys.readouterr().out
    assert "what changed since 2026-01-01" in out
    assert "[NEW]" in out
    assert "Newly appeared signals" in out


def test_main_report_respects_top(tmp_path, capsys):
    db = tmp_path / "runs.db"
    main(["demo", "--db", str(db)])
    capsys.readouterr()
    code = main(["--top", "1", "report", "--db", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert "showing 1." in out
    assert "co-northwind" in out
    assert "co-atlas" not in out


def test_main_report_empty_db_is_graceful(tmp_path, capsys):
    db = tmp_path / "empty.db"
    code = main(["report", "--db", str(db)])
    assert code == 0
    out = capsys.readouterr().out
    assert "No opportunities on file." in out


def test_report_requires_db():
    parser = build_parser()
    try:
        parser.parse_args(["report"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for missing --db")


def test_live_requires_user_agent():
    parser = build_parser()
    # argparse exits with code 2 when a required arg is missing.
    try:
        parser.parse_args(["live", "--cik", "320193"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for missing --user-agent")
