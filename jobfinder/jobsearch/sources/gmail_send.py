"""Send an approved outreach email via Gmail (the draft-and-approve send seam).

This is the *outbound* counterpart to the read-only ``gmail.GmailSource``. It is
the first and only place in the whole tool that can put an email on the wire, so
two things are deliberate and load-bearing:

1. **A NEW, narrower-than-it-sounds scope.** Slice B obtained only
   ``gmail.readonly``. Sending needs ``gmail.send`` — a *separate* OAuth scope and
   a *separate* cached token (``gmail_send_token.json``), so granting send never
   silently rides on the read grant and vice-versa. ``gmail.send`` can only create
   and send messages; it cannot read the mailbox.
2. **The service is injected, exactly like ``GmailSource``.** Every method is
   unit-testable offline against a fake service with the same call shape
   (``service.users().messages().send(...).execute()``); ``from_env()`` returns a
   sender only when send credentials exist on disk (else ``None``), and the live
   OAuth/``build()`` path (``_build_service``) is ``# pragma: no cover`` — never
   exercised in CI.

This module does NOT decide *whether* to send: the CLI's explicit
``outreach send <id> --confirm`` gate and the do-not-contact re-check live there.
This only performs the send once the human has approved it. Single-user, own
account.
"""

from __future__ import annotations

import base64
from email.headerregistry import Address
from email.message import EmailMessage
from pathlib import Path

# A NEW scope, distinct from gmail.readonly (Slice B). gmail.send permits
# creating and sending messages only — it grants no read access to the mailbox.
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"

# Credentials live alongside the read-only ones but the send token is a SEPARATE
# file: the send grant must be obtained deliberately and never share a cache with
# the read-only token (so neither scope silently widens the other).
DEFAULT_CRED_DIR = Path.home() / ".config" / "job-finder"
CLIENT_SECRET_FILENAME = "gmail_client_secret.json"
SEND_TOKEN_FILENAME = "gmail_send_token.json"


class GmailSendError(RuntimeError):
    """Raised when send credentials exist but can't be authorized, or a send fails.

    Subclasses ``RuntimeError`` so the CLI's existing ``except RuntimeError`` catch
    reports it as a clean message instead of a raw traceback.
    """


def build_raw_message(
    *,
    to_email: str,
    to_name: str,
    from_email: str,
    from_name: str,
    subject: str,
    body: str,
) -> str:
    """Build a base64url-encoded RFC-822 message for Gmail ``messages.send``.

    Uses stdlib ``email`` so headers are correctly encoded (UTF-8 bodies, display
    names) without a new dependency. Address headers are built with
    ``email.headerregistry.Address`` — NOT a raw ``f"{name} <{addr}>"`` string — so
    a display name containing ``<``, ``>``, or a comma (e.g. a contact named
    "Smith, Jane <x@y>") is properly quoted/escaped and cannot inject a second
    address or corrupt the recipient header. Returns the web-safe base64 string
    Gmail's ``raw`` field expects. Pure and side-effect-free — the network call is
    :meth:`GmailSender.send_email`.
    """
    message = EmailMessage()
    message["To"] = _address(to_name, to_email)
    message["From"] = _address(from_name, from_email)
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def _address(display_name: str, email_address: str) -> Address:
    """A safely-quoted address header from a display name + address.

    ``Address`` splits the address into local part + domain and escapes the display
    name, so header-injection via the name (or an odd address) is not possible. A
    malformed address with the wrong number of ``@`` is rejected up front by the
    outreach layer (``is_valid_business_email``); here we split defensively and let
    ``Address`` handle quoting."""
    local, _, domain = email_address.strip().partition("@")
    return Address(
        display_name=(display_name or "").strip(), username=local, domain=domain
    )


class GmailSender:
    """Sends an approved email via an injected Gmail API service.

    ``service`` is a Gmail API resource (``googleapiclient.discovery.build`` for
    real use, or a fake with the same call shape in tests). Only the
    ``users().messages().send()`` call is used.
    """

    def __init__(self, service, *, user_id: str = "me") -> None:
        self._service = service
        self._user_id = user_id

    @classmethod
    def from_env(cls, *, cred_dir: str | Path | None = None) -> GmailSender | None:
        """Build a live sender from on-disk send credentials, or return None.

        Credentials live under ``cred_dir`` (default ``~/.config/job-finder/``): the
        installed-app client secret (``gmail_client_secret.json``, shared with the
        read path) and a SEPARATE cached send token (``gmail_send_token.json``).
        Returns ``None`` when no client secret and no send token exist, so a tool
        without send set up simply cannot send (the CLI reports a clean setup
        message). A partial/unusable credential state surfaces as a clean
        :class:`GmailSendError`, not an opaque OAuth traceback.
        """
        cred_dir = Path(cred_dir) if cred_dir else DEFAULT_CRED_DIR
        secret_file = cred_dir / CLIENT_SECRET_FILENAME
        token_file = cred_dir / SEND_TOKEN_FILENAME

        # No credentials of any kind → cannot send (the CLI prints a setup hint).
        if not token_file.exists() and not secret_file.exists():
            return None

        try:
            service = _build_service(secret_file, token_file)
        except Exception as exc:  # live OAuth/network — normalise to a clean error
            raise GmailSendError(
                f"could not authorize Gmail send with credentials under {cred_dir}: "
                f"{exc}. Place {CLIENT_SECRET_FILENAME} there and authorize the "
                "send scope once."
            ) from exc
        return cls(service)

    def send_email(
        self,
        *,
        to_email: str,
        to_name: str,
        from_email: str,
        from_name: str,
        subject: str,
        body: str,
    ) -> str:
        """Send one email; return the provider message id. Raises on failure.

        The caller (the CLI send gate) has already obtained explicit ``--confirm``
        approval and re-checked the do-not-contact list before reaching here — this
        method just performs the approved send.
        """
        raw = build_raw_message(
            to_email=to_email,
            to_name=to_name,
            from_email=from_email,
            from_name=from_name,
            subject=subject,
            body=body,
        )
        try:
            result = (
                self._service.users()
                .messages()
                .send(userId=self._user_id, body={"raw": raw})
                .execute()
            )
        except Exception as exc:  # transient API failure — clean error to the CLI
            raise GmailSendError(f"Gmail send failed: {exc}") from exc
        return (result or {}).get("id", "")


def _build_service(
    secret_file: Path, token_file: Path
):  # pragma: no cover - live OAuth/network
    """Build an authenticated Gmail send service from on-disk credentials.

    Loads the cached SEND token, refreshing when expired; falls back to the
    installed-app consent flow for the ``gmail.send`` scope using the client
    secret, caching the resulting send token separately from the read-only one.
    Touches the network (and a browser on first consent), so it is exercised only
    by the user-driven live send, never in CI — the testable surface is
    ``GmailSender`` with an injected service.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    scopes = [GMAIL_SEND_SCOPE]
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if creds and creds.valid:
        pass
    elif creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(secret_file), scopes)
        creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)
