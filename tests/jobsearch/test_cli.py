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
