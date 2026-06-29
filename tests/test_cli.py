"""Tests for the pipeline wiring and the CLI demo (offline)."""

from datetime import date, datetime, timezone
from pathlib import Path

from jobfinder.cli import (
    _DEMO_PROFILE,
    _demo_companies,
    _live_companies,
    _parse_since,
    _profile_from_args,
    _store_url,
    build_parser,
    main,
    render,
)
from jobfinder.fit import CandidateProfile
from jobfinder.pipeline import CompanyInputs, run_pipeline, run_pipeline_detailed
from jobfinder.signals.extraction import RegexExtractor
from jobfinder.sources.ats import JobBoard, JobPosting
from jobfinder.sources.edgar import EdgarClient, Filing, FormD
from jobfinder.store import Store

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"


def _run(companies):
    # Force the deterministic regex extractor so tests never touch a live LLM,
    # and apply the demo candidate profile so fit is *derived* — exactly what the
    # `demo` CLI command does, so tests exercise the one canonical demo ranking.
    return run_pipeline(
        companies,
        candidate_profile=_DEMO_PROFILE,
        observed_at=NOW,
        now=NOW,
        extractor=RegexExtractor(),
    )


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
    # Northwind (all four pillars + a perfect firmographic fit) leads. Atlas now
    # outranks Helix: a CFO vacuum plus a perfect Logistics/Series A fit beats
    # Helix's hiring-only signals on an off-profile (AI/seed) firmographic.
    assert ids == ["co-northwind", "co-lumen", "co-atlas", "co-helix"]


def test_demo_northwind_is_funding_plus_vacuum():
    # The flagship demo case: a fresh raise AND a CFO departure with no named
    # successor. Under the deterministic regex extractor this is reproducible.
    opps = _run(_demo_companies())
    northwind = next(o for o in opps if o.company_id == "co-northwind")
    assert "open search" in northwind.why_now
    assert "Form D capital raised" in northwind.why_now


def test_demo_personas_differ_by_signal():
    # The headline proof of the persona-driven scorer: Northwind's CFO departure
    # targets a finance leader, while Helix's Engineering surge (no SEC filing)
    # targets an engineering leader — not the old hardcoded finance persona.
    opps = _run(_demo_companies())
    by_id = {o.company_id: o for o in opps}
    assert by_id["co-northwind"].target_persona == "CFO / VP Finance"
    assert by_id["co-helix"].target_persona == "VP Engineering / Engineering leader"
    # Atlas (CFO stepped down) is also finance; Lumen (funding only) falls back.
    assert by_id["co-atlas"].target_persona == "CFO / VP Finance"
    assert by_id["co-lumen"].target_persona == "CFO / VP Finance"


def test_demo_render_shows_distinct_personas():
    out = render(_run(_demo_companies()), _demo_companies(), top=5)
    assert "Target: VP Engineering / Engineering leader" in out
    assert "Target: CFO / VP Finance" in out
    # Header no longer claims a single fixed persona.
    assert "a senior role may be forming" in out


def test_render_includes_evidence_and_why_now():
    opps = _run(_demo_companies())
    out = render(opps, _demo_companies(), top=5)
    assert "Northwind Robotics Inc." in out
    assert "Why now:" in out
    assert "Evidence (supporting signals):" in out
    # The funding accession appears as cited evidence.
    assert "0001950000-26-000003:form_d" in out


def _detailed(companies):
    return run_pipeline_detailed(
        companies,
        candidate_profile=_DEMO_PROFILE,
        observed_at=NOW,
        now=NOW,
        extractor=RegexExtractor(),
    )


def test_render_shows_listed_roles_corroboration():
    # Slice 11: each opportunity surfaces the company's live ATS reqs, with the
    # in-function ones flagged — the hidden seat corroborated by listed roles.
    result = _detailed(_demo_companies())
    out = render(
        result.opportunities,
        _demo_companies(),
        top=5,
        signals=result.signals,
        now=NOW,
    )
    # Northwind's CFO opportunity lists its Finance reqs, in-function flagged.
    assert "Listed roles:" in out
    assert "in-function" in out
    assert "Controller" in out
    assert "Board: https://boards.greenhouse.io/northwind" in out


def test_render_no_listed_roles_for_pure_sec_opportunity():
    # Atlas has only an 8-K (no ATS board) -> no "Listed roles" block for it.
    result = _detailed(_demo_companies())
    out = render(
        result.opportunities,
        _demo_companies(),
        top=5,
        signals=result.signals,
        now=NOW,
    )
    atlas_block = out.split("Atlas Freight Inc.")[1].split("Helix Labs")[0]
    assert "Listed roles:" not in atlas_block


def test_render_funding_only_opp_does_not_fake_in_function_roles():
    # The code-review catch: a funding-only opportunity falls back to the scorer's
    # DEFAULT_PERSONA ('CFO / VP Finance') with no signal behind it. Its company's
    # routine Finance reqs must NOT be flagged in-function — that would manufacture
    # the backfill-vs-real-role corroboration this slice exists to provide.
    fd_filing = Filing(
        cik="9",
        accession_number="acc-9",
        form="D",
        filing_date=date(2026, 5, 1),
        report_date=None,
        items=[],
        primary_document="primary_doc.xml",
    )
    fd = FormD(
        issuer_cik="9",
        issuer_name="Quattro Capital",
        accession_number="acc-9",
        total_amount_sold=20_000_000.0,
        total_remaining=0.0,
    )
    # Only two Finance reqs -> below the department-surge threshold, so no signal
    # derives a finance persona; the opp's persona is the default fallback.
    board = JobBoard(
        provider="greenhouse",
        token="quattro",
        url="https://boards.greenhouse.io/quattro",
        postings=[
            JobPosting(id="1", title="Senior Accountant", department="Finance"),
            JobPosting(id="2", title="Controller", department="Finance"),
        ],
    )
    company = CompanyInputs(
        company_id="co-q",
        name="Quattro Capital",
        form_d=[(fd_filing, fd)],
        ats_boards=[board],
    )
    result = _detailed([company])
    opp = result.opportunities[0]
    # Persona is the default fallback (not signal-derived).
    assert opp.target_persona == "CFO / VP Finance"
    assert "inferred from signal" not in opp.why_now
    out = render(
        result.opportunities, [company], top=5, signals=result.signals, now=NOW
    )
    # The reqs are still listed (2 live) but none flagged in-function.
    assert "Listed roles: 2 live" in out
    assert "0 in-function" in out
    assert "[in-function" not in out


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
        _demo_companies(),
        candidate_profile=_DEMO_PROFILE,
        observed_at=NOW,
        now=NOW,
        extractor=RegexExtractor(),
    )
    # Same ranking as run_pipeline, plus the raw signals that produced it.
    assert [o.company_id for o in result.opportunities] == [
        "co-northwind",
        "co-lumen",
        "co-atlas",
        "co-helix",
    ]
    assert result.signals
    # Every cited supporting signal is present in the returned signal list.
    cited = {sid for o in result.opportunities for sid in o.supporting_signal_ids}
    assert cited <= {s.id for s in result.signals}


def test_demo_fit_is_derived_not_hardcoded():
    # Slice 8: the demo companies' fit is derived from firmographics, not the old
    # hand-set magic numbers. Northwind (Robotics/Series B/180) is a perfect
    # match; Helix (AI/seed/35) is off-profile and lower.
    opps = _run(_demo_companies())
    by_id = {o.company_id: o for o in opps}
    assert by_id["co-northwind"].fit_score == 1.0
    assert by_id["co-helix"].fit_score < 0.5
    # And the reason is visible in the why_now (explainable end to end).
    assert "Robotics matches target sector" in by_id["co-northwind"].why_now


def test_demo_atlas_outranks_helix_on_fit():
    # A perfect firmographic fit (Logistics/Series A) plus a CFO vacuum lifts
    # Atlas above hiring-only Helix on an off-profile firmographic — the headline
    # effect of wiring company_fit.
    opps = _run(_demo_companies())
    ids = [o.company_id for o in opps]
    assert ids.index("co-atlas") < ids.index("co-helix")


def test_pipeline_without_profile_uses_literal_company_fit():
    # No candidate_profile -> the literal CompanyInputs.company_fit is used and
    # no firmographic reason appears (pre-Slice-8 behaviour preserved).
    company = CompanyInputs(
        company_id="co-x",
        name="X",
        form_d=[
            (
                Filing(
                    cik="1",
                    accession_number="acc-x",
                    form="D",
                    filing_date=date(2026, 5, 1),
                    report_date=None,
                    items=[],
                    primary_document="primary_doc.xml",
                ),
                FormD(
                    issuer_cik="1",
                    issuer_name="X",
                    accession_number="acc-x",
                    total_amount_sold=10_000_000.0,
                    total_remaining=0.0,
                ),
            )
        ],
        company_fit=0.77,
    )
    opps = run_pipeline([company], observed_at=NOW, now=NOW, extractor=RegexExtractor())
    assert opps[0].fit_score == 0.77
    assert "Fit " not in opps[0].why_now


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
        "co-atlas",
        "co-helix",
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


# --------------------------------------------------------------------------- #
# Slice 9: live firmographics derivation + candidate-profile flags.
# --------------------------------------------------------------------------- #
def _edgar_fetcher(url: str) -> str:
    """Route the two URL shapes the live SEC path hits to fixtures, fully
    offline: the submissions index (used for both filings and company_info) and
    any primary document."""
    if "/submissions/" in url:
        return (FIXTURES / "submissions_sample.json").read_text()
    return (FIXTURES / "8k_item502_apple.txt").read_text()


def test_live_company_derives_firmographics_and_real_name():
    companies = _live_companies(EdgarClient(_edgar_fetcher), ["320193"], now=NOW)
    assert len(companies) == 1
    company = companies[0]
    # The real entity name from the index replaces the "CIK <n>" stub.
    assert company.name == "Apple Inc."
    # Firmographics are derived from the filer's SIC sector (no extra fetch).
    assert company.firmographics is not None
    assert company.firmographics.sector == "Electronic Computers"
    # SEC discloses neither, so they stay neutral-by-omission.
    assert company.firmographics.funding_stage is None
    assert company.firmographics.employee_count is None


def test_live_company_skips_fetching_stale_form_d():
    # End-to-end recency floor: the fixture has two Apr-2026 Form Ds and one from
    # Sep 2025. With `now` well past a year after the stale one, _live_companies
    # must filter it out *before* fetching, so its XML is never requested — the
    # collector cutoff and the signal floor share the one run clock.
    fetched: list[str] = []

    def fetcher(url: str) -> str:
        fetched.append(url)
        if "/submissions/" in url:
            return (FIXTURES / "submissions_form_d.json").read_text()
        return (FIXTURES / "form_d_sample.xml").read_text()

    # ~10 months after the latest filing: the Apr-2026 pair is still inside the
    # 365-day horizon, the Sep-2025 filing is well past it.
    now = datetime(2027, 2, 1, tzinfo=timezone.utc)
    companies = _live_companies(EdgarClient(fetcher), ["1950000"], now=now)

    # The stale filing's XML was never fetched (only the index + the two fresh
    # Form D documents).
    doc_fetches = [u for u in fetched if "/submissions/" not in u]
    assert len(doc_fetches) == 2
    # And only the two fresh Form Ds produced funding signals.
    assert len(companies) == 1
    form_d_signals = [
        s
        for s in run_pipeline_detailed(
            companies, observed_at=now, now=now, extractor=RegexExtractor()
        ).signals
        if s.signal_type in ("form_d_funding", "form_d_amendment")
    ]
    assert len(form_d_signals) == 2


def test_live_company_fit_uses_derived_sector():
    # End to end: a profile whose target sector matches the derived SIC sector
    # lifts company_fit above the neutral 0.5; a non-matching sector pulls it
    # below — proving live runs no longer get the flat placeholder.
    companies = _live_companies(EdgarClient(_edgar_fetcher), ["320193"], now=NOW)

    match = run_pipeline(
        companies,
        candidate_profile=CandidateProfile(target_sectors=("electronic",)),
        observed_at=NOW,
        now=NOW,
        extractor=RegexExtractor(),
    )
    miss = run_pipeline(
        companies,
        candidate_profile=CandidateProfile(target_sectors=("biotechnology",)),
        observed_at=NOW,
        now=NOW,
        extractor=RegexExtractor(),
    )
    assert match[0].fit_score > 0.5 > miss[0].fit_score
    assert "Electronic Computers matches target sector" in match[0].why_now


def test_profile_from_args_none_when_no_flags():
    args = build_parser().parse_args(
        ["live", "--cik", "320193", "--user-agent", "jf you@example.com"]
    )
    # No --target-* flags -> no profile -> live fit keeps the neutral literal.
    assert _profile_from_args(args) is None


def test_profile_from_args_builds_from_flags():
    args = build_parser().parse_args(
        [
            "live",
            "--cik",
            "320193",
            "--user-agent",
            "jf you@example.com",
            "--target-sector",
            "robotics",
            "--target-sector",
            "logistics",
            "--target-stage",
            "series_b",
            "--min-employees",
            "50",
            "--max-employees",
            "300",
        ]
    )
    profile = _profile_from_args(args)
    assert profile is not None
    assert profile.target_sectors == ("robotics", "logistics")
    assert profile.target_stages == ("series_b",)
    assert profile.min_employees == 50
    assert profile.max_employees == 300


def test_profile_from_args_size_only():
    # A single size bound is enough to request a profile (its presence, not just
    # the sector flags, must trigger derivation).
    args = build_parser().parse_args(
        [
            "live",
            "--cik",
            "320193",
            "--user-agent",
            "jf you@example.com",
            "--min-employees",
            "10",
        ]
    )
    profile = _profile_from_args(args)
    assert profile is not None
    assert profile.min_employees == 10
