"""Hermetic tests for outreach draft assembly + the optional LLM tailoring seam.

Assembly is deterministic and explainable: the subject/body are built from the
job/contact/persona, CAN-SPAM hygiene (truthful subject, real sender identity,
opt-out) is always present, and only business recipient addresses are accepted.
The LLM tailorer is INJECTED — a fake drives it in CI, no live Gemini call — and
degrades to the deterministic template on any failure.
"""

from jobfinder.jobsearch.models import (
    CanonicalJob,
    Contact,
    ContactRole,
    OutreachEmail,
    RawPosting,
    Source,
)
from jobfinder.jobsearch.outreach import (
    DEFAULT_OPT_OUT,
    SenderIdentity,
    TailoredBody,
    assemble_email,
    render_email_text,
)

SENDER = SenderIdentity(name="Jon Hodges", email="jon@myco.com")


def _job(company="Acme", title="VP of AI", location="Remote"):
    return CanonicalJob(
        company=company,
        title=title,
        normalized_title=title.lower(),
        location=location,
        sources=[
            RawPosting(title=title, company=company, source=Source.LINKEDIN_ALERT)
        ],
    )


def _contact(name="Jane Smith", role=ContactRole.HIRING_MANAGER):
    return Contact(name=name, company="Acme", role=role)


# --------------------------------------------------------------------------- #
# Deterministic assembly.
# --------------------------------------------------------------------------- #
def test_assemble_is_deterministic_and_business_addressed():
    email = assemble_email(
        _job(),
        _contact(),
        "jane.smith@acme.com",
        sender=SENDER,
        match_reason="strong VP-of-AI leadership fit",
    )
    assert isinstance(email, OutreachEmail)
    assert email.to_email == "jane.smith@acme.com"
    assert email.tailoring == "template"
    # Same inputs → identical output (no randomness / no hidden state).
    again = assemble_email(
        _job(),
        _contact(),
        "jane.smith@acme.com",
        sender=SENDER,
        match_reason="strong VP-of-AI leadership fit",
    )
    assert again == email


def test_subject_is_truthful_and_names_the_role():
    email = assemble_email(_job(), _contact(), "jane.smith@acme.com", sender=SENDER)
    # CAN-SPAM: the subject is non-deceptive and says exactly what this is.
    assert "VP of AI" in email.subject
    assert "Acme" in email.subject


def test_body_addresses_recipient_by_first_name_and_cites_match_reason():
    email = assemble_email(
        _job(),
        _contact(name="Jane Smith"),
        "jane.smith@acme.com",
        sender=SENDER,
        match_reason="owns AI org and budget",
    )
    assert email.body.startswith("Hi Jane,")
    assert "owns AI org and budget" in email.body


def test_role_specific_line_varies_by_contact_role():
    hm = assemble_email(
        _job(), _contact(role=ContactRole.HIRING_MANAGER), "a@acme.com", sender=SENDER
    )
    rec = assemble_email(
        _job(), _contact(role=ContactRole.RECRUITER), "a@acme.com", sender=SENDER
    )
    assert hm.body != rec.body  # the persona line differs by role


def test_rendered_text_contains_all_can_spam_elements():
    email = assemble_email(_job(), _contact(), "jane.smith@acme.com", sender=SENDER)
    text = render_email_text(email)
    # Truthful identity (real name + reply-to) and a working opt-out are present
    # in what actually gets sent — appended by us, never by the LLM.
    assert "Jon Hodges" in text
    assert "jon@myco.com" in text
    assert DEFAULT_OPT_OUT in text


# --------------------------------------------------------------------------- #
# Business-only + safety refusals (defence in depth at the assembly layer).
# --------------------------------------------------------------------------- #
def test_personal_recipient_domain_is_refused():
    assert (
        assemble_email(_job(), _contact(), "jane.smith@gmail.com", sender=SENDER)
        is None
    )


def test_personal_subdomain_recipient_is_refused():
    # Sub-domains of a personal provider count as personal too.
    assert (
        assemble_email(_job(), _contact(), "jane@mail.gmail.com", sender=SENDER) is None
    )


def test_malformed_recipient_is_refused():
    assert assemble_email(_job(), _contact(), "not-an-email", sender=SENDER) is None
    assert assemble_email(_job(), _contact(), "", sender=SENDER) is None


def test_empty_local_part_recipient_is_refused():
    # "@acme.com" has a valid domain but no mailbox — a domain-only check would
    # pass it (normalize_domain resolves "acme.com"); it must still be refused.
    assert assemble_email(_job(), _contact(), "@acme.com", sender=SENDER) is None


def test_multiple_at_recipient_is_refused():
    # "a@@acme.com" / "a@b@acme.com" resolve a domain via rsplit but are malformed.
    assert assemble_email(_job(), _contact(), "a@@acme.com", sender=SENDER) is None
    assert assemble_email(_job(), _contact(), "a@b@acme.com", sender=SENDER) is None


# --------------------------------------------------------------------------- #
# The optional injected LLM tailoring seam.
# --------------------------------------------------------------------------- #
class _FakeTailorer:
    """A fake body tailorer: records the call and returns a canned body."""

    def __init__(self, body="Tailored body from the LLM.", raises=False):
        self._body = body
        self._raises = raises
        self.calls = []

    def tailor(self, draft, job, contact):
        self.calls.append((draft, job, contact))
        if self._raises:
            raise RuntimeError("LLM boom")
        return TailoredBody(body=self._body)


def test_tailorer_replaces_body_only_and_marks_provenance():
    tailorer = _FakeTailorer(body="A sharper, human-sounding intro.")
    email = assemble_email(
        _job(), _contact(), "jane.smith@acme.com", sender=SENDER, tailorer=tailorer
    )
    assert tailorer.calls  # the seam was actually consulted
    assert email.body == "A sharper, human-sounding intro."
    assert email.tailoring == "llm+template"
    # Subject, identity, and opt-out stay the deterministic ones (LLM body only).
    assert "VP of AI" in email.subject
    text = render_email_text(email)
    assert DEFAULT_OPT_OUT in text
    assert "jon@myco.com" in text


def test_tailorer_failure_degrades_to_template():
    tailorer = _FakeTailorer(raises=True)
    email = assemble_email(
        _job(), _contact(), "jane.smith@acme.com", sender=SENDER, tailorer=tailorer
    )
    baseline = assemble_email(_job(), _contact(), "jane.smith@acme.com", sender=SENDER)
    assert email.body == baseline.body
    assert email.tailoring == "template"


def test_empty_llm_body_degrades_to_template():
    tailorer = _FakeTailorer(body="   ")
    email = assemble_email(
        _job(), _contact(), "jane.smith@acme.com", sender=SENDER, tailorer=tailorer
    )
    assert email.tailoring == "template"
    assert email.body.startswith("Hi Jane,")
