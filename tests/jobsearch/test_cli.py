"""CLI smoke tests for the jobsearch tool — offline against the .eml fixtures."""

import re
from pathlib import Path

import pytest

from jobfinder.jobsearch.cli import _parse_ats_spec, main
from jobfinder.jobsearch.match import rank_jobs
from jobfinder.jobsearch.models import Tier
from jobfinder.jobsearch.normalize import canonicalize
from jobfinder.jobsearch.profile import VP_AI_PROFILE
from jobfinder.jobsearch.sources.eml_dir import read_eml_dir

FIXTURES = Path(__file__).parent / "fixtures"


def test_rank_offline_against_alert_fixtures(capsys):
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C"])
    assert rc == 0
    out = capsys.readouterr().out
    # The VP-of-AI roles surface; the IC engineer is not shown as a match.
    assert "VP, AI & Data" in out
    assert "VP of AI & Analytics" in out
    assert "Senior Machine Learning Engineer" not in out


def test_rank_ic_engineer_is_rejected_not_merely_low():
    # Stronger than "absent from output": assert the IC engineer is hard-rejected
    # by the negative filter, not just filtered out for scoring below min-tier.
    jobs = canonicalize(read_eml_dir(str(FIXTURES)))
    matches = rank_jobs(jobs, VP_AI_PROFILE)
    eng = next(m for m in matches if m.job.title == "Senior Machine Learning Engineer")
    assert eng.rejected
    assert eng.tier is Tier.C


def test_rank_requires_a_source(capsys):
    rc = main(["rank"])
    assert rc == 2
    assert "at least one source" in capsys.readouterr().err


def test_rank_min_tier_a_filters_lower(capsys):
    # A tier-A title (VP, AI & Data, remote) is shown; a tier-B one ("Head of AI"
    # in NY, no remote) is filtered out when --min-tier A. (The header always
    # contains "tier A+", so we assert on the actual rows, not the header.)
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "A"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VP, AI & Data" in out  # tier A, present
    assert "Head of AI" not in out  # tier B, filtered out by --min-tier A
    # And every numbered result row is explicitly an [A] (no lower tier leaks in).
    rows = re.findall(r"^\d+\. \[(\w)\]", out, re.MULTILINE)
    assert rows and all(t == "A" for t in rows)


def test_ats_without_user_agent_errors(capsys):
    rc = main(["rank", "--ats", "greenhouse:acme"])
    assert rc == 2
    assert "user-agent" in capsys.readouterr().err.lower()


def test_bad_alerts_dir_errors(capsys, tmp_path):
    missing = tmp_path / "nope"
    rc = main(["rank", "--alerts-dir", str(missing)])
    assert rc == 2
    assert "rank:" in capsys.readouterr().err


def test_parse_ats_spec_rejects_unknown_provider():
    with pytest.raises(ValueError):
        _parse_ats_spec("monster:acme")
    assert _parse_ats_spec("greenhouse:acme") == ("greenhouse", "acme")


def test_gmail_request_without_credentials_errors(capsys, monkeypatch):
    # Asking for Gmail ingestion with no credentials on disk fails with a clear
    # setup message (rc 2), rather than silently ranking nothing.
    import jobfinder.jobsearch.cli as cli

    monkeypatch.setattr(cli.GmailSource, "from_env", classmethod(lambda cls: None))
    rc = main(["rank", "--gmail-label", "job-alerts"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "gmail" in err and "credentials" in err


def test_gmail_source_built_lazily_and_merged(capsys, monkeypatch):
    # When --gmail-query is given, the source is built and its postings join the
    # ranking; when it is absent, from_env is never called (no OAuth on offline
    # runs).
    import jobfinder.jobsearch.cli as cli
    from jobfinder.jobsearch.models import RawPosting, Source

    built = {"count": 0}

    class _FakeSource:
        def fetch_postings(self, *, label=None, query=None):
            return [
                RawPosting(
                    title="VP of AI",
                    company="Acme",
                    source=Source.LINKEDIN_ALERT,
                    location="Remote",
                    url="https://www.linkedin.com/comm/jobs/view/1/",
                )
            ]

    def _fake_from_env(cls):
        built["count"] += 1
        return _FakeSource()

    monkeypatch.setattr(cli.GmailSource, "from_env", classmethod(_fake_from_env))

    # Offline run: Gmail source must NOT be built.
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C"])
    assert rc == 0
    assert built["count"] == 0

    # Gmail run: source built once, its posting reaches the ranking.
    rc = main(["rank", "--gmail-query", "from:jobalerts-noreply@linkedin.com"])
    assert rc == 0
    assert built["count"] == 1
    assert "VP of AI" in capsys.readouterr().out


def test_rerank_without_key_falls_back_to_layer1(capsys, monkeypatch):
    # --rerank with no GEMINI_API_KEY warns and ranks with pure Layer-1 (rc 0).
    import jobfinder.jobsearch.cli as cli

    monkeypatch.setattr(cli.GeminiReranker, "from_env", classmethod(lambda cls: None))
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--rerank"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "VP, AI & Data" in captured.out
    assert "GEMINI_API_KEY" in captured.err  # warned about the fallback
    assert "LLM re-rank:" not in captured.out  # no annotation without a re-ranker


def test_rerank_not_built_without_flag(capsys, monkeypatch):
    # A run WITHOUT --rerank must never build the re-ranker (pure Layer-1).
    import jobfinder.jobsearch.cli as cli

    def _boom(cls):
        raise AssertionError("from_env must not be called without --rerank")

    monkeypatch.setattr(cli.GeminiReranker, "from_env", classmethod(_boom))
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C"])
    assert rc == 0
    assert "LLM re-rank:" not in capsys.readouterr().out


def test_rerank_annotation_surfaces_in_output(capsys, monkeypatch):
    # With a (fake) re-ranker, the LLM's contribution is rendered — never silently
    # folded into the score.
    import jobfinder.jobsearch.cli as cli
    from jobfinder.jobsearch.rerank import RerankedItem, RerankResponse

    class _FakeReranker:
        def rerank(self, candidates, profile):
            return RerankResponse(
                ranking=[
                    RerankedItem(
                        candidate_id=i,
                        relevance="strong",
                        rationale="great AI leadership scope",
                    )
                    for i in range(len(candidates))
                ]
            )

    monkeypatch.setattr(
        cli.GeminiReranker, "from_env", classmethod(lambda cls: _FakeReranker())
    )
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--rerank"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LLM re-rank:" in out
    assert "great AI leadership scope" in out


def test_empty_gmail_flag_does_not_trigger_oauth(capsys, monkeypatch):
    # An empty-string --gmail-label must count as "not requested" (matching
    # _run_rank's `not args.gmail_*` check), so a run with a real --alerts-dir
    # never touches OAuth and succeeds offline.
    import jobfinder.jobsearch.cli as cli

    def _boom(cls):  # from_env must not be reached for an empty flag
        raise AssertionError("from_env should not be called for an empty flag")

    monkeypatch.setattr(cli.GmailSource, "from_env", classmethod(_boom))
    rc = main(
        ["rank", "--alerts-dir", str(FIXTURES), "--gmail-label", "", "--min-tier", "C"]
    )
    assert rc == 0
    assert "VP, AI & Data" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Slice D — persistence (--db) + list / status subcommands.
# --------------------------------------------------------------------------- #
def test_rank_without_db_writes_nothing(tmp_path, capsys, monkeypatch):
    # A run with no --db is the offline print-and-forget path: no DB file appears
    # in the working directory.
    monkeypatch.chdir(tmp_path)
    rc = main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C"])
    assert rc == 0
    assert list(tmp_path.iterdir()) == []  # nothing persisted


def test_rank_does_not_persist_rejected_matches(tmp_path, capsys):
    # Hard-rejected roles (the IC engineer fixture) are hidden by the rank display;
    # they must NOT land in the CRM either, or `list` would resurface them.
    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    rc = main(["list", "--db", str(db), "--all"])  # --all = even archived shown
    assert rc == 0
    out = capsys.readouterr().out
    assert "Senior Machine Learning Engineer" not in out


def test_rank_db_open_failure_is_clean_error(tmp_path, capsys):
    # A --db path that can't be opened (a directory that doesn't exist) yields a
    # clean rc-2 message, not an uncaught traceback that loses the ranked output.
    bad = tmp_path / "nope" / "crm.db"  # parent dir missing
    rc = main(
        ["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(bad)]
    )
    assert rc == 2
    assert "could not save" in capsys.readouterr().err


def test_status_on_vanished_row_reports_not_false_success(
    tmp_path, capsys, monkeypatch
):
    # If the row is gone between id-resolution and the write, status reports the
    # failure (rc 2) rather than printing a false success.
    import jobfinder.jobsearch.cli as cli

    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    from jobfinder.jobsearch.normalize import job_key
    from jobfinder.jobsearch.store import JobStore

    key = job_key(JobStore(f"sqlite+pysqlite:///{db}").list_jobs()[0].match.job)
    # Force set_status to behave as if the row vanished.
    monkeypatch.setattr(cli.JobStore, "set_status", lambda self, jid, st, **kw: False)
    rc = main(["status", "--db", str(db), key, "applied"])
    assert rc == 2
    assert "no longer exists" in capsys.readouterr().err


def test_rank_with_db_persists_and_list_reads_back(tmp_path, capsys):
    db = tmp_path / "crm.db"
    rc = main(
        ["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "Saved to" in err and "new" in err
    assert db.exists()

    # `list` reads the persisted jobs back, status-annotated, highest score first.
    rc = main(["list", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VP, AI & Data" in out
    assert "[new]" in out  # freshly ingested jobs default to NEW
    assert "id:" in out  # the stable key is shown so `status` can target it


def test_status_subcommand_advances_and_persists(tmp_path, capsys):
    from jobfinder.jobsearch.models import ApplicationStatus
    from jobfinder.jobsearch.normalize import job_key
    from jobfinder.jobsearch.store import JobStore

    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()

    # Resolve a real id from the store, then advance it via the CLI.
    store = JobStore(f"sqlite+pysqlite:///{db}")
    key = job_key(store.list_jobs()[0].match.job)

    rc = main(["status", "--db", str(db), key, "applied"])
    assert rc == 0
    assert "applied" in capsys.readouterr().out

    # The change persisted.
    reread = JobStore(f"sqlite+pysqlite:///{db}")
    assert reread.get(key).status is ApplicationStatus.APPLIED


def test_status_unknown_id_errors(tmp_path, capsys):
    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    rc = main(["status", "--db", str(db), "no-such-id", "applied"])
    assert rc == 2
    assert "no job id" in capsys.readouterr().err


def test_status_empty_id_is_rejected_not_silent_mutation(tmp_path, capsys):
    # An empty fragment must NOT slip past the ambiguity guard and silently mutate
    # the sole stored job — it is its own explicit error (regression for the
    # empty-prefix footgun).
    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    rc = main(["status", "--db", str(db), "", "applied"])
    assert rc == 2
    assert "provide a job id" in capsys.readouterr().err


def test_status_ambiguous_prefix_errors(tmp_path, capsys):
    # A non-empty prefix matching more than one job is rejected with the candidate
    # ids listed, so the user disambiguates rather than mutating the wrong job.
    from datetime import datetime, timezone

    from jobfinder.jobsearch.match import score_job
    from jobfinder.jobsearch.models import CanonicalJob
    from jobfinder.jobsearch.profile import VP_AI_PROFILE
    from jobfinder.jobsearch.store import JobStore

    db = tmp_path / "crm.db"
    store = JobStore(f"sqlite+pysqlite:///{db}")
    now = datetime(2026, 6, 25, tzinfo=timezone.utc)
    # Two jobs at the same company → keys share the company prefix "acme|".
    for title in ("VP of AI", "Head of Data Science"):
        job = CanonicalJob(
            company="Acme",
            title=title,
            normalized_title=title.lower(),
            location="Remote",
        )
        store.save_match(score_job(job, VP_AI_PROFILE, now=now), now=now)

    rc = main(["status", "--db", str(db), "acme|", "applied"])
    assert rc == 2
    assert "ambiguous" in capsys.readouterr().err


def test_status_survives_a_later_rank_run(tmp_path, capsys):
    # End-to-end of the Slice-D crux through the CLI: mark APPLIED, re-rank the
    # same mailbox with --db, and the status is NOT reset.
    from jobfinder.jobsearch.models import ApplicationStatus
    from jobfinder.jobsearch.normalize import job_key
    from jobfinder.jobsearch.store import JobStore

    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    store = JobStore(f"sqlite+pysqlite:///{db}")
    key = job_key(store.list_jobs()[0].match.job)
    main(["status", "--db", str(db), key, "applied"])
    capsys.readouterr()

    # Re-rank the same fixtures into the same DB.
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    assert JobStore(f"sqlite+pysqlite:///{db}").get(key).status is (
        ApplicationStatus.APPLIED
    )


def test_list_empty_db_message(tmp_path, capsys):
    db = tmp_path / "empty.db"
    rc = main(["list", "--db", str(db)])
    assert rc == 0
    assert "No jobs match" in capsys.readouterr().out
