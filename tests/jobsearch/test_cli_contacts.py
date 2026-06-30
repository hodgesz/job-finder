"""CLI smoke tests for the Slice-E subcommands: contacts / add-contact / email / dnc.

Hermetic and offline: a CRM db is seeded by ranking the .eml fixtures, then the
contact/email/do-not-contact subcommands are exercised against it. No network, no
LinkedIn, no LLM.
"""

from pathlib import Path

import pytest

from jobfinder.jobsearch.cli import main
from jobfinder.jobsearch.normalize import job_key
from jobfinder.jobsearch.store import JobStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def seeded_db(tmp_path, capsys):
    """A CRM db seeded with the fixture jobs; yields (db_path, a job_key)."""
    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    store = JobStore(f"sqlite+pysqlite:///{db}")
    key = job_key(store.list_jobs()[0].match.job)
    return db, key


# --------------------------------------------------------------------------- #
# contacts checklist.
# --------------------------------------------------------------------------- #
def test_contacts_checklist(seeded_db, capsys):
    db, key = seeded_db
    rc = main(["contacts", "--db", str(db), key])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Manual contact checklist" in out
    assert "BY HAND" in out
    assert "hiring_manager" in out


def test_contacts_unknown_job(seeded_db, capsys):
    db, _ = seeded_db
    rc = main(["contacts", "--db", str(db), "nonexistent-fragment-zzz"])
    assert rc == 2
    assert "no job id" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# add-contact paste-back round-trip.
# --------------------------------------------------------------------------- #
def test_add_contact_round_trip(seeded_db, capsys):
    db, key = seeded_db
    rc = main(
        [
            "add-contact",
            "--db",
            str(db),
            key,
            "Jane Smith",
            "--role",
            "hiring_manager",
            "--title",
            "CTO",
            "--domain",
            "acme.com",
        ]
    )
    assert rc == 0
    assert "Added contact 'Jane Smith'" in capsys.readouterr().out

    # It shows up under contacts --show with an email guess.
    rc = main(["contacts", "--db", str(db), key, "--show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Jane Smith" in out
    assert "jane.smith@acme.com" in out  # top business-email guess


def test_add_contact_rejects_personal_domain(seeded_db, capsys):
    db, key = seeded_db
    rc = main(
        [
            "add-contact",
            "--db",
            str(db),
            key,
            "Jane Smith",
            "--domain",
            "gmail.com",
        ]
    )
    assert rc == 2
    assert "personal email domain" in capsys.readouterr().err


def test_add_contact_rejects_malformed_domain(seeded_db, capsys):
    # A non-personal but malformed --domain must be rejected, not stored as junk.
    db, key = seeded_db
    rc = main(
        [
            "add-contact",
            "--db",
            str(db),
            key,
            "Jane Smith",
            "--domain",
            "acme dot com",
        ]
    )
    assert rc == 2
    assert "not a valid business domain" in capsys.readouterr().err


def test_add_contact_unknown_job(seeded_db, capsys):
    db, _ = seeded_db
    rc = main(["add-contact", "--db", str(db), "zzz-nope", "Jane Smith"])
    assert rc == 2
    assert "no job id" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# email subcommand.
# --------------------------------------------------------------------------- #
def test_email_basic(tmp_path, capsys):
    db = tmp_path / "crm.db"
    rc = main(["email", "Jane Smith", "acme.com", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "jane.smith@acme.com" in out


def test_email_rejects_personal_domain(tmp_path, capsys):
    db = tmp_path / "crm.db"
    rc = main(["email", "Jane Smith", "gmail.com", "--db", str(db)])
    assert rc == 2
    assert "personal email domain" in capsys.readouterr().err


def test_email_honors_dnc_with_db(tmp_path, capsys):
    db = tmp_path / "crm.db"
    # Suppress the whole domain, then ask for a guess at it.
    main(["dnc", "--db", str(db), "acme.com"])
    capsys.readouterr()
    rc = main(["email", "Jane Smith", "acme.com", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "do-not-contact" in out
    assert "jane.smith@acme.com" not in out


# --------------------------------------------------------------------------- #
# dnc subcommand + the always-honored guarantee end-to-end.
# --------------------------------------------------------------------------- #
def test_dnc_add_and_list(tmp_path, capsys):
    db = tmp_path / "crm.db"
    main(["dnc", "--db", str(db), "jane@acme.com", "--reason", "asked to stop"])
    capsys.readouterr()
    main(["dnc", "--db", str(db), "beta.com"])
    capsys.readouterr()
    rc = main(["dnc", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "jane@acme.com" in out
    assert "beta.com" in out
    assert "asked to stop" in out


def test_dnc_rejects_garbage(tmp_path, capsys):
    db = tmp_path / "crm.db"
    rc = main(["dnc", "--db", str(db), "not an email or domain"])
    assert rc == 2
    assert "not a valid email or domain" in capsys.readouterr().err


def test_dnc_suppresses_contact_email_guess(seeded_db, capsys):
    """End-to-end: a suppressed address never appears under contacts --show."""
    db, key = seeded_db
    main(
        [
            "add-contact",
            "--db",
            str(db),
            key,
            "Jane Smith",
            "--domain",
            "acme.com",
        ]
    )
    capsys.readouterr()
    # Suppress exactly the top guess.
    main(["dnc", "--db", str(db), "jane.smith@acme.com"])
    capsys.readouterr()
    rc = main(["contacts", "--db", str(db), key, "--show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "jane.smith@acme.com" not in out  # suppressed everywhere
    # A non-suppressed lower-confidence guess is still offered.
    assert "jsmith@acme.com" in out


def test_dnc_domain_suppresses_all_guesses_for_contact(seeded_db, capsys):
    db, key = seeded_db
    main(
        [
            "add-contact",
            "--db",
            str(db),
            key,
            "Jane Smith",
            "--domain",
            "acme.com",
        ]
    )
    capsys.readouterr()
    main(["dnc", "--db", str(db), "acme.com"])  # whole domain
    capsys.readouterr()
    rc = main(["contacts", "--db", str(db), key, "--show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "@acme.com" not in out
    assert "do-not-contact" in out  # marked suppressed
