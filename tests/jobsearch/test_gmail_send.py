"""Hermetic tests for the Gmail send seam (Slice F).

Drives ``GmailSender`` against a *fake* Gmail service mirroring the real
``googleapiclient`` chained-builder shape
(``service.users().messages().send(...).execute()``), so there is no live
network, no OAuth, and no secrets — exactly like the read-only ``GmailSource``
tests. We assert the send builds a well-formed RFC-822 message, requests the NEW
gmail.send scope (never readonly), and that ``from_env`` degrades to ``None``
without credentials.
"""

import base64

import pytest

from jobfinder.jobsearch.sources.gmail_send import (
    GMAIL_SEND_SCOPE,
    SEND_TOKEN_FILENAME,
    GmailSender,
    GmailSendError,
    build_raw_message,
)


class _FakeExec:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class _FakeMessages:
    def __init__(self, service):
        self._service = service

    def send(self, *, userId, body):
        self._service.sent.append({"userId": userId, "body": body})
        if self._service.raise_on_send:
            return _FakeExec(lambda: (_ for _ in ()).throw(RuntimeError("api down")))
        return _FakeExec(lambda: {"id": "sent-123"})


class _FakeUsers:
    def __init__(self, service):
        self._service = service

    def messages(self):
        return _FakeMessages(self._service)


class FakeSendService:
    """A minimal Gmail send API double. Records every send() call in ``sent``."""

    def __init__(self, *, raise_on_send=False):
        self.sent = []
        self.raise_on_send = raise_on_send

    def users(self):
        return _FakeUsers(self)


def _decode_sent(service):
    """Decode the RFC-822 bytes of the single recorded send."""
    raw = service.sent[0]["body"]["raw"]
    return base64.urlsafe_b64decode(raw).decode("utf-8")


# --------------------------------------------------------------------------- #
# Message construction.
# --------------------------------------------------------------------------- #
def test_build_raw_message_encodes_headers_and_body():
    raw = build_raw_message(
        to_email="jane@acme.com",
        to_name="Jane Smith",
        from_email="jon@myco.com",
        from_name="Jon Hodges",
        subject="Interest in your VP of AI role",
        body="Hi Jane,\n\nHello.",
    )
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    assert "To: Jane Smith <jane@acme.com>" in decoded
    assert "From: Jon Hodges <jon@myco.com>" in decoded
    assert "Subject: Interest in your VP of AI role" in decoded
    assert "Hi Jane," in decoded


def test_display_name_with_angle_brackets_cannot_inject_a_recipient():
    # A hostile/odd display name must not smuggle a second address into the To
    # header. email.headerregistry.Address quotes the display name, so the only
    # real address remains the intended one.
    raw = build_raw_message(
        to_email="jane@acme.com",
        to_name="Jane <ceo@evil.com>",
        from_email="jon@myco.com",
        from_name="Jon Hodges",
        subject="Hi",
        body="x",
    )
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    to_line = next(ln for ln in decoded.splitlines() if ln.startswith("To:"))
    # The hostile name is fully quoted as a display name, so the ONLY routable
    # address is the intended one: the To line ends with "<jane@acme.com>" and the
    # injected address survives only inside the quoted display-name string.
    assert to_line.strip().endswith("<jane@acme.com>")
    assert '"Jane <ceo@evil.com>"' in to_line  # neutralised: quoted, not routable


def test_display_name_with_comma_is_quoted():
    raw = build_raw_message(
        to_email="jane@acme.com",
        to_name="Smith, Jane",
        from_email="jon@myco.com",
        from_name="Jon",
        subject="Hi",
        body="x",
    )
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    # A comma in a display name must be quoted so it isn't read as an address
    # separator; the real recipient address is still present.
    assert "jane@acme.com" in decoded
    assert '"Smith, Jane"' in decoded


# --------------------------------------------------------------------------- #
# Sending via the injected service.
# --------------------------------------------------------------------------- #
def test_send_email_calls_service_and_returns_message_id():
    service = FakeSendService()
    mid = GmailSender(service).send_email(
        to_email="jane@acme.com",
        to_name="Jane Smith",
        from_email="jon@myco.com",
        from_name="Jon Hodges",
        subject="Hello",
        body="Body text.",
    )
    assert mid == "sent-123"
    assert len(service.sent) == 1
    decoded = _decode_sent(service)
    assert "jane@acme.com" in decoded
    assert "Body text." in decoded


def test_send_email_wraps_api_failure_in_clean_error():
    service = FakeSendService(raise_on_send=True)
    with pytest.raises(GmailSendError):
        GmailSender(service).send_email(
            to_email="jane@acme.com",
            to_name="Jane",
            from_email="jon@myco.com",
            from_name="Jon",
            subject="Hi",
            body="x",
        )


# --------------------------------------------------------------------------- #
# Scope + credential degrade.
# --------------------------------------------------------------------------- #
def test_send_scope_is_gmail_send_not_readonly():
    # Guard: the send seam uses the NEW gmail.send scope, never readonly.
    assert GMAIL_SEND_SCOPE.endswith("gmail.send")
    assert "readonly" not in GMAIL_SEND_SCOPE


def test_from_env_returns_none_without_credentials(tmp_path):
    # No send token and no client secret → cannot send (degrade to None).
    assert GmailSender.from_env(cred_dir=tmp_path) is None


def test_from_env_partial_credentials_raise_clean_error(tmp_path):
    # A stale send token whose client secret was removed must surface as a clean
    # GmailSendError, NOT an opaque OAuth traceback.
    (tmp_path / SEND_TOKEN_FILENAME).write_text("{not valid creds}")
    with pytest.raises(GmailSendError):
        GmailSender.from_env(cred_dir=tmp_path)
