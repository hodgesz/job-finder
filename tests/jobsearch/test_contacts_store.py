"""Tests for contact + do-not-contact persistence in JobStore (Slice E).

Hermetic: in-memory SQLite. Covers contact round-trip + idempotent upsert, the
always-honored do-not-contact list (email and domain entries), and that the new
tables are decoupled from the jobs table (a contact can exist without its job
ever being saved).
"""

from datetime import datetime, timezone

import pytest

from jobfinder.jobsearch.models import Contact, ContactRole, ContactSource
from jobfinder.jobsearch.store import JobStore, contact_key

NOW = datetime(2026, 6, 25, tzinfo=timezone.utc)
LATER = datetime(2026, 6, 27, tzinfo=timezone.utc)


@pytest.fixture
def store() -> JobStore:
    return JobStore.in_memory()


def _contact(
    *,
    name: str = "Jane Smith",
    company: str = "Acme, Inc.",
    role: ContactRole = ContactRole.HIRING_MANAGER,
    domain: str | None = "acme.com",
    title: str | None = "CTO",
) -> Contact:
    return Contact(
        name=name,
        company=company,
        role=role,
        title=title,
        linkedin_url="https://www.linkedin.com/in/janesmith",
        email_domain=domain,
        source=ContactSource.MANUAL,
        notes="met at a conference",
    )


# --------------------------------------------------------------------------- #
# Contacts.
# --------------------------------------------------------------------------- #
def test_contact_round_trips(store: JobStore):
    assert store.save_contact("job-1", _contact(), now=NOW) is True
    got = store.list_contacts("job-1")
    assert len(got) == 1
    c = got[0].contact
    assert c.name == "Jane Smith"
    assert c.role is ContactRole.HIRING_MANAGER
    assert c.title == "CTO"
    assert c.email_domain == "acme.com"
    assert c.source is ContactSource.MANUAL
    assert c.notes == "met at a conference"
    assert got[0].first_seen_at == NOW


def test_recording_same_person_updates_in_place(store: JobStore):
    assert store.save_contact("job-1", _contact(title="CTO"), now=NOW) is True
    # Same name + domain → same contact_key → update, not duplicate.
    assert (
        store.save_contact("job-1", _contact(title="Chief Tech Officer"), now=LATER)
        is False
    )
    got = store.list_contacts("job-1")
    assert len(got) == 1
    assert got[0].contact.title == "Chief Tech Officer"
    assert got[0].first_seen_at == NOW  # preserved
    assert got[0].last_seen_at == LATER  # advanced


def test_same_name_different_domain_are_distinct(store: JobStore):
    store.save_contact("job-1", _contact(domain="acme.com"))
    store.save_contact("job-1", _contact(domain="other.com"))
    assert len(store.list_contacts("job-1")) == 2


def test_contacts_are_scoped_per_job(store: JobStore):
    store.save_contact("job-1", _contact())
    store.save_contact("job-2", _contact(name="Bob Jones"))
    assert len(store.list_contacts("job-1")) == 1
    assert len(store.list_contacts("job-2")) == 1


def test_contact_key_is_normalized():
    a = _contact(name="  Jane   SMITH ", domain="ACME.com")
    b = _contact(name="jane smith", domain="acme.com")
    assert contact_key(a) == contact_key(b)


def test_contact_key_normalizes_domain_forms():
    # Regression: the same person recorded with a URL/@-form domain vs a bare
    # domain must key identically (else duplicate rows).
    bare = _contact(domain="acme.com")
    url = _contact(domain="https://acme.com/careers")
    at = _contact(domain="@acme.com")
    assert contact_key(bare) == contact_key(url) == contact_key(at)


def test_recording_same_person_varying_domain_form_updates_in_place(store: JobStore):
    assert store.save_contact("job-1", _contact(domain="acme.com")) is True
    assert (
        store.save_contact("job-1", _contact(domain="https://acme.com/jobs")) is False
    )
    assert len(store.list_contacts("job-1")) == 1


# --------------------------------------------------------------------------- #
# Do-not-contact list — always honored.
# --------------------------------------------------------------------------- #
def test_dnc_email_suppresses_exact_address(store: JobStore):
    entry = store.add_do_not_contact("Jane.Smith@Acme.com")
    assert entry is not None
    assert entry.kind == "email"
    assert entry.value == "jane.smith@acme.com"  # normalized
    assert store.is_suppressed("jane.smith@acme.com")
    assert store.is_suppressed("JANE.SMITH@ACME.COM")  # case-insensitive
    assert not store.is_suppressed("other@acme.com")


def test_dnc_domain_suppresses_every_address_at_domain(store: JobStore):
    entry = store.add_do_not_contact("acme.com")
    assert entry.kind == "domain"
    assert store.is_suppressed("jane.smith@acme.com")
    assert store.is_suppressed("anyone@acme.com")
    assert not store.is_suppressed("jane@other.com")


def test_dnc_domain_suppresses_subdomains(store: JobStore):
    # Regression: blocking the company domain must cover its sub-domains.
    store.add_do_not_contact("acme.com")
    assert store.is_suppressed("jane@careers.acme.com")
    assert store.is_suppressed("jane@mail.eu.acme.com")
    assert not store.is_suppressed("jane@notacme.com")


def test_dnc_reason_is_consistent_on_reAdd(store: JobStore):
    # Regression: a no-reason re-add must keep AND report the stored reason
    # (return value must not disagree with list_do_not_contact).
    store.add_do_not_contact("beta.com", reason="spammer")
    again = store.add_do_not_contact("beta.com")  # no reason
    assert again is not None
    assert again.reason == "spammer"  # reported, not None
    assert store.list_do_not_contact()[0].reason == "spammer"  # and unchanged


def test_dnc_add_is_idempotent_and_lists_sorted(store: JobStore):
    store.add_do_not_contact("acme.com")
    store.add_do_not_contact("acme.com", reason="competitor")  # update reason
    store.add_do_not_contact("jane@beta.com")
    entries = store.list_do_not_contact()
    assert [e.value for e in entries] == ["acme.com", "jane@beta.com"]
    assert entries[0].reason == "competitor"


def test_dnc_rejects_garbage(store: JobStore):
    assert store.add_do_not_contact("") is None
    assert store.add_do_not_contact("not a domain or email") is None
    assert store.list_do_not_contact() == []


def test_dnc_url_form_normalizes_to_domain(store: JobStore):
    entry = store.add_do_not_contact("https://acme.com/careers")
    assert entry.kind == "domain"
    assert entry.value == "acme.com"
    assert store.is_suppressed("anyone@acme.com")


def test_is_suppressed_on_empty_list_is_false(store: JobStore):
    assert not store.is_suppressed("jane@acme.com")
