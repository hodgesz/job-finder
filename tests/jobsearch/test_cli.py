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
