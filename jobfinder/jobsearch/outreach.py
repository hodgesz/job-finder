"""Draft-and-approve outreach assembly (Slice F).

Turns data the tool already has — a ``CanonicalJob``, a recorded ``Contact`` and
their role, the persona/match reason — into a **tailored outreach email** (subject
+ body) for a chosen *business* address. Assembling a draft sends nothing: the
two-step ``outreach send <id> --confirm`` gate (see ``cli``) is the only path that
puts an email on the wire, and the do-not-contact list is re-checked there
(defence in depth). This module only *builds* the draft.

Two properties are structural, not left to a caller's discipline:

1. **CAN-SPAM-style hygiene is baked into the rendered email.** The footer — the
   sender's real identity (a truthful "From") and a plain opt-out line — is
   appended *by us* in :func:`render_email_text`, never by the optional LLM, so a
   tailored body can't drop it. The subject is the deterministic, truthful one;
   the LLM tailors only the *body*. The recipient is always a business address
   (personal domains are refused, here and at the producer ``guess_emails``).
2. **The LLM is an optional enhancement that degrades to the template.** Exactly
   the ``rerank.py`` contract: the ``Tailorer`` is injected (CI drives a fake, no
   live call), ``GeminiTailorer.from_env()`` returns a worker only when
   ``GEMINI_API_KEY`` is set (else ``None`` → pure template), and *any* failure in
   the LLM path falls back to the deterministic template rather than raising. A
   run without a key produces exactly the deterministic draft.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from jobfinder.jobsearch.email_format import is_valid_business_email
from jobfinder.jobsearch.models import (
    CanonicalJob,
    Contact,
    ContactRole,
    OutreachEmail,
)

# Mirrors the re-ranker so a model bump is one place across the detour.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# A plain, truthful opt-out (CAN-SPAM): a clear way for the recipient to ask not
# to be contacted again. The user records that request back via the ``dnc``
# subcommand, which suppresses the address everywhere thereafter.
DEFAULT_OPT_OUT = (
    'If you\'d prefer not to hear from me, just reply with "no thanks" and I '
    "won't follow up."
)

# Role-specific one-liners keyed by how the contact relates to the role. Each is
# truthful and explains *why this person specifically* — so the draft reads as
# deliberate outreach, not a blast. OTHER/unknown falls back to the generic line.
_ROLE_LINES: dict[ContactRole, str] = {
    ContactRole.HIRING_MANAGER: (
        "As the person most likely to own this search, I'd value even a brief "
        "sense of what success looks like in the role."
    ),
    ContactRole.FUNCTION_LEADER: (
        "As a leader in the AI/data organisation, your read on where the team is "
        "headed would mean a lot as I consider the fit."
    ),
    ContactRole.EXECUTIVE: (
        "I know your time is short, so I'll be brief — I wanted to introduce "
        "myself directly given how central this hire is."
    ),
    ContactRole.RECRUITER: (
        "I wanted to make sure my background reaches you directly so it's easy to "
        "route if it's a fit."
    ),
}
_GENERIC_ROLE_LINE = (
    "I wanted to introduce myself directly rather than let my application sit in "
    "a queue."
)


@dataclass(frozen=True)
class SenderIdentity:
    """The candidate's real identity for a truthful CAN-SPAM "From".

    Supplied explicitly by the user (CLI ``--from-name``/``--from-email``); never
    auto-discovered. ``email`` is the real reply-to address the recipient can use
    to opt out.
    """

    name: str
    email: str


class TailoredBody(BaseModel):
    """The LLM's rewritten *body* only — never the subject, identity, or opt-out.

    Keeping the LLM to the body means the truthful subject and the structural
    CAN-SPAM footer are always the deterministic ones, regardless of what the
    model returns. This is the single contract the fake tailorer mimics in tests.
    """

    body: str = Field(
        description="A concise, professional outreach body (no subject, no "
        "signature, no opt-out line — those are added deterministically)."
    )


@runtime_checkable
class Tailorer(Protocol):
    """Anything that can sharpen a draft's body for a specific job + contact.

    Returns a :class:`TailoredBody`. Implementations must raise on failure (the
    caller catches and degrades to the template) rather than returning junk.
    """

    def tailor(
        self,
        draft: OutreachEmail,
        job: CanonicalJob,
        contact: Contact,
    ) -> TailoredBody: ...


_SYSTEM_INSTRUCTION = (
    "You help a candidate for senior AI/data leadership roles (VP of AI, VP of AI "
    "& Data, Head of AI, and close equivalents) write a short, warm, professional "
    "cold-outreach email body to a named contact about a specific role. Keep it "
    "honest and specific to the role and the person — no hype, no fabricated "
    "claims, no invented mutual connections. Return ONLY the body text: no subject "
    "line, no greeting beyond the first name, no signature, and no opt-out line "
    "(those are added separately). Three short paragraphs at most."
)


def _first_name(name: str) -> str:
    """The recipient's first name for a greeting, original casing, or 'there'."""
    token = name.strip().split()
    return token[0] if token else "there"


def _subject(job_title: str, company: str) -> str:
    """A truthful, non-deceptive subject (CAN-SPAM): says exactly what this is."""
    return f"Interest in your {job_title} role at {company}"


def _template_body(
    contact: Contact,
    job: CanonicalJob,
    *,
    match_reason: str | None,
) -> str:
    """Build the deterministic mail-merge body. Pure, explainable, no network."""
    first = _first_name(contact.name)
    title = job.title or "open"
    company = (job.company or "your company").strip() or "your company"
    role_line = _ROLE_LINES.get(contact.role, _GENERIC_ROLE_LINE)
    paragraphs = [
        f"Hi {first},",
        f"I came across the {title} opening at {company} and wanted to introduce "
        "myself directly. I lead AI/data/analytics organisations and am exploring "
        "where I can have the most impact next.",
        role_line,
    ]
    if match_reason:
        # The deterministic Layer-1 reason already explains the fit in the user's
        # own scoring terms — reuse it verbatim so the body never overstates.
        paragraphs.append(f"Why this role caught my eye: {match_reason}")
    paragraphs.append(
        "Would you be open to a short conversation? I'm happy to work around your "
        "schedule."
    )
    return "\n\n".join(paragraphs)


def render_email_text(email: OutreachEmail) -> str:
    """Compose the full email text: body + the structural CAN-SPAM footer.

    The footer (a divider, the sender's real name + reply-to email, and the
    opt-out line) is appended HERE, deterministically — never by the LLM — so the
    sender's truthful identity and a working opt-out are always present in what
    actually gets sent, no matter how the body was produced.
    """
    return (
        f"{email.body}\n\n--\n{email.from_name}\n{email.from_email}\n\n{email.opt_out}"
    )


def assemble_email(
    job: CanonicalJob,
    contact: Contact,
    to_email: str,
    *,
    sender: SenderIdentity,
    match_reason: str | None = None,
    opt_out: str = DEFAULT_OPT_OUT,
    tailorer: Tailorer | None = None,
) -> OutreachEmail | None:
    """Assemble a tailored outreach draft, or ``None`` if it can't be built safely.

    Returns ``None`` (rather than raising) for the cases where honest, compliant
    outreach is impossible: a missing/malformed recipient address, or a *personal*
    recipient domain (business emails only — re-checked here even though
    ``guess_emails`` already refuses personal domains, so this layer is safe on its
    own). ``to_email`` is expected to come pre-vetted from ``guess_emails`` (already
    filtered against the do-not-contact list); the send gate re-checks suppression.

    The body is the deterministic template by default. When a ``tailorer`` is
    given and succeeds, it replaces the *body only* and ``tailoring`` becomes
    ``"llm+template"``; any failure keeps the template body (``tailoring`` stays
    ``"template"``). The subject, the sender identity, and the opt-out are always
    the deterministic, truthful ones.
    """
    # One honest-recipient gate: well-formed address + non-personal business
    # domain. Covers the malformed shapes a domain-only check would let through
    # ("@acme.com", "a@@acme.com"), and the personal-domain refusal (defence in
    # depth — guess_emails already refuses personal domains at the producer).
    if not is_valid_business_email(to_email):
        return None

    company = (job.company or contact.company or "").strip()
    job_title = job.title or "the open role"
    draft = OutreachEmail(
        to_email=to_email.strip(),
        to_name=contact.name.strip(),
        subject=_subject(job_title, company or "your company"),
        body=_template_body(contact, job, match_reason=match_reason),
        from_name=sender.name.strip(),
        from_email=sender.email.strip(),
        company=company,
        job_title=job_title,
        opt_out=opt_out,
        tailoring="template",
    )

    if tailorer is None:
        return draft
    try:
        tailored = tailorer.tailor(draft, job, contact)
        body = tailored.body.strip()
    except Exception:
        # Enhancement, not a hard dependency: keep the deterministic template.
        return draft
    if not body:
        return draft  # empty LLM body is no better than no LLM
    return replace(draft, body=body, tailoring="llm+template")


def _tailor_prompt(
    draft: OutreachEmail, job: CanonicalJob, contact: Contact, match_reason: str | None
) -> str:
    """Render the id-keyed context the LLM tailors against (no PII beyond the name)."""
    loc = job.location or "location not stated"
    role = contact.role.value
    lines = [
        f"Contact: {contact.name} (role relative to the job: {role})",
        f"Company: {draft.company or 'unknown'}",
        f"Role: {draft.job_title} [{loc}]",
    ]
    if match_reason:
        lines.append(f"Why it fits (the candidate's own scoring): {match_reason}")
    lines.append(
        "\nWrite the outreach body now (body text only, three short paragraphs at "
        "most)."
    )
    return "\n".join(lines)


class GeminiTailorer:
    """LLM body-tailorer backed by Gemini structured output.

    The genai client is injected so tests supply a fake; use ``from_env()`` to
    build a real client from ``GEMINI_API_KEY``. Mirrors ``GeminiReranker``.
    """

    def __init__(
        self,
        client,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        match_reason: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        # The match reason is run-specific context, threaded in at construction so
        # the ``Tailorer`` protocol's ``tailor`` signature stays job+contact only.
        self._match_reason = match_reason

    @classmethod
    def from_env(
        cls, *, model: str = DEFAULT_GEMINI_MODEL, match_reason: str | None = None
    ) -> GeminiTailorer | None:
        """Build from ``GEMINI_API_KEY``, or return ``None`` if unset/unavailable.

        Mirrors ``GeminiReranker.from_env`` — a missing key or an uninstalled
        ``google-genai`` both yield ``None``, so the caller degrades to the
        deterministic template.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            from google import genai
        except ImportError:
            return None
        return cls(
            genai.Client(api_key=api_key), model=model, match_reason=match_reason
        )

    def tailor(
        self, draft: OutreachEmail, job: CanonicalJob, contact: Contact
    ) -> TailoredBody:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=_tailor_prompt(draft, job, contact, self._match_reason),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TailoredBody,
                temperature=0.3,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, TailoredBody):
            return parsed
        return TailoredBody.model_validate_json(response.text)
