"""CLI tests for the Slice-F outreach subcommands: draft / list-drafts / send.

The headline guarantee is the draft-and-approve gate: assembling a draft and a
dry-run send put NOTHING on the wire, and only an explicit ``send --confirm``
reaches the Gmail send seam — which is a FAKE here (no live network/secrets). The
do-not-contact list is honored at draft AND re-checked at send (defence in depth),
and business-emails-only is enforced. Hermetic and offline.
"""

from pathlib import Path

import pytest

import jobfinder.jobsearch.cli as cli
from jobfinder.jobsearch.cli import main
from jobfinder.jobsearch.normalize import job_key
from jobfinder.jobsearch.store import JobStore

FIXTURES = Path(__file__).parent / "fixtures"

FROM = ["--from-name", "Jon Hodges", "--from-email", "jon@myco.com"]


@pytest.fixture
def seeded_db(tmp_path, capsys):
    """A CRM db seeded with the fixture jobs; yields (db_path, a job_key)."""
    db = tmp_path / "crm.db"
    main(["rank", "--alerts-dir", str(FIXTURES), "--min-tier", "C", "--db", str(db)])
    capsys.readouterr()
    store = JobStore(f"sqlite+pysqlite:///{db}")
    key = job_key(store.list_jobs()[0].match.job)
    return db, key


class _FakeSender:
    """A fake gmail.send seam that records sends instead of hitting the network."""

    instances = []

    def __init__(self):
        self.sent = []
        _FakeSender.instances.append(self)

    def send_email(self, **kwargs):
        self.sent.append(kwargs)
        return "fake-msg-1"


@pytest.fixture
def fake_sender(monkeypatch):
    """Patch GmailSender.from_env to return a recording fake (never real OAuth)."""
    _FakeSender.instances.clear()
    sender = _FakeSender()
    monkeypatch.setattr(cli.GmailSender, "from_env", classmethod(lambda cls: sender))
    return sender


def _draft(db, key, capsys, *, name="Jane Smith", extra=None):
    argv = ["outreach", "draft", "--db", str(db), key, name, "--domain", "acme.com"]
    argv += FROM + (extra or [])
    rc = main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# --------------------------------------------------------------------------- #
# draft: assembles + stores, sends nothing.
# --------------------------------------------------------------------------- #
def test_draft_assembles_and_stores_but_sends_nothing(seeded_db, capsys, fake_sender):
    db, key = seeded_db
    rc, out, _ = _draft(db, key, capsys)
    assert rc == 0
    assert "nothing sent" in out.lower()
    assert "jane.smith@acme.com" in out  # top business guess used
    # Nothing was sent by merely drafting.
    assert fake_sender.sent == []
    # The draft was persisted.
    drafts = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()
    assert len(drafts) == 1
    assert drafts[0].status.value == "drafted"


def test_draft_requires_sender_identity(seeded_db, capsys):
    db, key = seeded_db
    # Missing --from-name/--from-email → argparse rejects (required args).
    with pytest.raises(SystemExit):
        main(
            ["outreach", "draft", "--db", str(db), key, "Jane", "--domain", "acme.com"]
        )


def test_draft_refuses_malformed_from_email(seeded_db, capsys):
    # The sender's own From must pass the SAME business-email gate as recipients —
    # a malformed From ("@acme.com") must not become the stored CAN-SPAM identity.
    db, key = seeded_db
    rc = main(
        [
            "outreach",
            "draft",
            "--db",
            str(db),
            key,
            "Jane",
            "--to-email",
            "jane.smith@acme.com",
            "--from-name",
            "Jon",
            "--from-email",
            "@acme.com",
        ]
    )
    assert rc == 2
    assert "from-email must be a valid business address" in capsys.readouterr().err
    # Nothing was drafted.
    assert JobStore(f"sqlite+pysqlite:///{db}").list_drafts() == []


def test_draft_refuses_personal_recipient_domain(seeded_db, capsys):
    db, key = seeded_db
    rc = main(
        [
            "outreach",
            "draft",
            "--db",
            str(db),
            key,
            "Jane",
            "--to-email",
            "jane@gmail.com",
        ]
        + FROM
    )
    assert rc == 2
    assert "not a valid business email" in capsys.readouterr().err


def test_draft_honors_do_not_contact_at_assembly(seeded_db, capsys):
    db, key = seeded_db
    main(["dnc", "--db", str(db), "acme.com"])
    capsys.readouterr()
    # Every guess at the suppressed domain is filtered → no recipient to draft.
    rc, _, err = _draft(db, key, capsys)
    assert rc == 2
    assert "do-not-contact" in err.lower()


# --------------------------------------------------------------------------- #
# send: the draft-and-approve gate.
# --------------------------------------------------------------------------- #
def test_send_without_confirm_is_dry_run_and_sends_nothing(
    seeded_db, capsys, fake_sender
):
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    rc = main(["outreach", "send", "--db", str(db), did])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in out
    # THE headline assertion: no send happened without --confirm.
    assert fake_sender.sent == []
    # Draft stays DRAFTED (not flipped to sent).
    assert JobStore(f"sqlite+pysqlite:///{db}").get_draft(did).status.value == "drafted"


def test_send_with_confirm_calls_the_seam_and_marks_sent(
    seeded_db, capsys, fake_sender
):
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    rc = main(["outreach", "send", "--db", str(db), did, "--confirm"])
    assert rc == 0
    # The explicit confirm path is the ONLY one that reaches the seam.
    assert len(fake_sender.sent) == 1
    assert fake_sender.sent[0]["to_email"] == "jane.smith@acme.com"
    assert "jon@myco.com" in fake_sender.sent[0]["body"]  # opt-out footer present
    # The store records it as sent.
    assert JobStore(f"sqlite+pysqlite:///{db}").get_draft(did).status.value == "sent"


def test_send_refuses_when_recipient_added_to_dnc_after_drafting(
    seeded_db, capsys, fake_sender
):
    # Defence in depth: a stale draft whose recipient is later suppressed must NOT
    # send even with --confirm.
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    main(["dnc", "--db", str(db), "acme.com"])  # suppress AFTER the draft exists
    capsys.readouterr()
    rc = main(["outreach", "send", "--db", str(db), did, "--confirm"])
    assert rc == 2
    assert "do-not-contact" in capsys.readouterr().err.lower()
    # The send seam was never called.
    assert fake_sender.sent == []
    # And the draft was NOT marked sent.
    assert JobStore(f"sqlite+pysqlite:///{db}").get_draft(did).status.value == "drafted"


def test_send_does_not_resend_an_already_sent_draft(seeded_db, capsys, fake_sender):
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    main(["outreach", "send", "--db", str(db), did, "--confirm"])
    capsys.readouterr()
    # A second confirm must refuse (already sent) and not re-send.
    rc = main(["outreach", "send", "--db", str(db), did, "--confirm"])
    assert rc == 2
    assert "already sent" in capsys.readouterr().err.lower()
    assert len(fake_sender.sent) == 1  # still just the one send


def test_send_without_credentials_reports_setup_and_sends_nothing(
    seeded_db, capsys, monkeypatch
):
    # --confirm but no send credentials → clean setup message, nothing sent.
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    monkeypatch.setattr(cli.GmailSender, "from_env", classmethod(lambda cls: None))
    rc = main(["outreach", "send", "--db", str(db), did, "--confirm"])
    assert rc == 2
    assert "credentials" in capsys.readouterr().err.lower()
    assert JobStore(f"sqlite+pysqlite:///{db}").get_draft(did).status.value == "drafted"


def test_redrafting_a_sent_recipient_warns_and_preserves_sent(
    seeded_db, capsys, fake_sender
):
    # Re-drafting a recipient you already emailed must NOT silently reset the SENT
    # record (that would re-arm the send gate for a double-send). The CLI warns and
    # keeps the sent record.
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    main(["outreach", "send", "--db", str(db), did, "--confirm"])
    capsys.readouterr()
    rc, _, err = _draft(db, key, capsys)  # re-draft the SAME job + recipient
    assert rc == 2
    assert "already emailed" in err.lower()
    # Still exactly one draft, still SENT — nothing reset.
    d = JobStore(f"sqlite+pysqlite:///{db}").get_draft(did)
    assert d.status.value == "sent"
    # And a subsequent send still refuses (guard intact), no second send.
    rc2 = main(["outreach", "send", "--db", str(db), did, "--confirm"])
    assert rc2 == 2
    assert "already sent" in capsys.readouterr().err.lower()
    assert len(fake_sender.sent) == 1


def test_dry_run_still_shows_a_suppressed_draft_with_a_note(
    seeded_db, capsys, fake_sender
):
    # A draft whose recipient was later suppressed must still be REVIEWABLE via a
    # dry run (it just can't be sent) — the DNC re-check blocks the send, not the
    # review.
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id
    main(["dnc", "--db", str(db), "acme.com"])
    capsys.readouterr()
    rc = main(["outreach", "send", "--db", str(db), did])  # no --confirm
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in out
    assert "cannot be sent" in out
    assert "jane.smith@acme.com" in out  # the draft is still displayed
    assert fake_sender.sent == []


def test_send_claims_draft_before_sending_so_a_failed_send_can_retry(
    seeded_db, capsys, monkeypatch
):
    # A genuinely failed send releases the claim (draft back to DRAFTED) so the
    # user can retry — the row is not stuck SENT after a non-delivery.
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id

    class _FailingSender:
        def send_email(self, **kwargs):
            raise cli.GmailSendError("api down")

    monkeypatch.setattr(
        cli.GmailSender, "from_env", classmethod(lambda cls: _FailingSender())
    )
    rc = main(["outreach", "send", "--db", str(db), did, "--confirm"])
    assert rc == 2
    # Released back to DRAFTED for a retry (not stuck SENT after a non-delivery).
    assert JobStore(f"sqlite+pysqlite:///{db}").get_draft(did).status.value == "drafted"


def test_unexpected_send_error_releases_claim_not_stuck_sent(
    seeded_db, capsys, monkeypatch
):
    # A NON-GmailSendError failure during the send (e.g. an unexpected bug) must
    # still release the claim so the draft isn't stranded SENT-but-undelivered —
    # otherwise later sends would be refused as "already sent" for an email that
    # never went out. The error is re-raised (not swallowed).
    db, key = seeded_db
    _draft(db, key, capsys)
    did = JobStore(f"sqlite+pysqlite:///{db}").list_drafts()[0].id

    class _BuggySender:
        def send_email(self, **kwargs):
            raise ValueError("unexpected boom")

    monkeypatch.setattr(
        cli.GmailSender, "from_env", classmethod(lambda cls: _BuggySender())
    )
    with pytest.raises(ValueError):
        main(["outreach", "send", "--db", str(db), did, "--confirm"])
    # The claim was released despite the unexpected error type.
    assert JobStore(f"sqlite+pysqlite:///{db}").get_draft(did).status.value == "drafted"


# --------------------------------------------------------------------------- #
# list-drafts.
# --------------------------------------------------------------------------- #
def test_list_drafts_shows_assembled_drafts(seeded_db, capsys, fake_sender):
    db, key = seeded_db
    _draft(db, key, capsys)
    rc = main(["outreach", "list-drafts", "--db", str(db)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 outreach draft" in out
    assert "jane.smith@acme.com" in out


def test_list_drafts_empty(seeded_db, capsys):
    db, key = seeded_db
    rc = main(["outreach", "list-drafts", "--db", str(db)])
    assert rc == 0
    assert "No drafts" in capsys.readouterr().out


def test_send_ambiguous_fragment_refuses(seeded_db, capsys, fake_sender):
    db, key = seeded_db
    _draft(db, key, capsys, name="Jane Smith")
    _draft(db, key, capsys, name="John Doe", extra=["--to-email", "john.doe@other.com"])
    # "d_" prefixes every draft id → ambiguous.
    rc = main(["outreach", "send", "--db", str(db), "d_"])
    assert rc == 2
    assert "ambiguous" in capsys.readouterr().err.lower()
    assert fake_sender.sent == []
