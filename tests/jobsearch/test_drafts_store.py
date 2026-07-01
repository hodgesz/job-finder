"""Tests for the outreach-drafts persistence layer (Slice F).

Hermetic: an in-memory SQLite store. Covers OutreachEmail round-trip fidelity,
the stable (job, recipient) draft key (re-assembly updates in place), the
DRAFTED→SENT transition via mark_sent, and that re-assembling resets a stale SENT
draft back to DRAFTED (an edited draft is not yet sent).
"""

import pytest

from jobfinder.jobsearch.models import DraftStatus, OutreachEmail
from jobfinder.jobsearch.store import JobStore, draft_key


def _email(to_email="jane.smith@acme.com", subject="Interest in your VP of AI role"):
    return OutreachEmail(
        to_email=to_email,
        to_name="Jane Smith",
        subject=subject,
        body="Hi Jane,\n\nHello.",
        from_name="Jon Hodges",
        from_email="jon@myco.com",
        company="Acme",
        job_title="VP of AI",
        opt_out="Reply 'no thanks' to opt out.",
        tailoring="template",
    )


@pytest.fixture
def store() -> JobStore:
    return JobStore.in_memory()


def test_draft_round_trips_with_full_fidelity(store):
    email = _email()
    sd = store.save_draft("job-1", email)
    got = store.get_draft(sd.id)
    assert got is not None
    assert got.email == email
    assert got.job_id == "job-1"
    assert got.status is DraftStatus.DRAFTED
    assert got.sent_at is None


def test_llm_tailoring_provenance_round_trips(store):
    email = _email()
    email = OutreachEmail(**{**email.__dict__, "tailoring": "llm+template"})
    sd = store.save_draft("job-1", email)
    assert store.get_draft(sd.id).email.tailoring == "llm+template"


def test_draft_key_is_stable_per_job_and_recipient(store):
    # Same job + recipient (case/format-insensitive) → same key → update in place.
    assert draft_key("job-1", "Jane@Acme.com") == draft_key("job-1", "jane@acme.com")
    # Different job or recipient → different key.
    assert draft_key("job-1", "jane@acme.com") != draft_key("job-2", "jane@acme.com")
    assert draft_key("job-1", "jane@acme.com") != draft_key("job-1", "bob@acme.com")


def test_reassembling_same_recipient_updates_in_place(store):
    store.save_draft("job-1", _email(subject="v1"))
    store.save_draft("job-1", _email(subject="v2"))
    drafts = store.list_drafts()
    assert len(drafts) == 1  # one row, not two
    assert drafts[0].email.subject == "v2"


def test_mark_sent_flips_status_and_stamps(store):
    sd = store.save_draft("job-1", _email())
    assert store.mark_sent(sd.id) is True
    got = store.get_draft(sd.id)
    assert got.status is DraftStatus.SENT
    assert got.sent_at is not None


def test_mark_sent_on_missing_draft_returns_false(store):
    assert store.mark_sent("d_nope") is False


def test_reassembling_a_sent_draft_preserves_the_sent_record(store):
    # CRUCIAL: re-drafting a recipient you ALREADY emailed must NOT overwrite the
    # SENT record. Doing so would both erase the "already contacted" fact and
    # silently re-arm the send gate's already-sent guard, enabling a double-send.
    # save_draft returns the existing SENT record unchanged.
    sd = store.save_draft("job-1", _email(subject="v1"))
    store.mark_sent(sd.id)
    returned = store.save_draft("job-1", _email(subject="v2"))
    assert returned.status is DraftStatus.SENT  # signals "already sent" to the CLI
    got = store.get_draft(sd.id)
    assert got.status is DraftStatus.SENT
    assert got.sent_at is not None
    assert got.email.subject == "v1"  # the sent body is preserved, not clobbered


def test_list_drafts_filters_by_status(store):
    sd1 = store.save_draft("job-1", _email(to_email="a@acme.com"))
    store.save_draft("job-2", _email(to_email="b@acme.com"))
    store.mark_sent(sd1.id)
    assert len(store.list_drafts(status=DraftStatus.SENT)) == 1
    assert len(store.list_drafts(status=DraftStatus.DRAFTED)) == 1
    assert len(store.list_drafts()) == 2


def test_find_draft_ids_escapes_like_wildcards(store):
    store.save_draft("job-1", _email())
    # A bare wildcard must NOT match the stored draft (literal prefix only).
    assert store.find_draft_ids("%") == []
    assert store.find_draft_ids("_") == []
