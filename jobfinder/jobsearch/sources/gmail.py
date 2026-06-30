"""Read LinkedIn job-alert emails straight from Gmail (read-only).

This is the live counterpart to ``eml_dir.read_eml_dir``: instead of requiring
the user to export ``.eml`` files to a folder, ``GmailSource`` lists the
job-alert messages in their mailbox, fetches each one's raw RFC-822 bytes, and
runs them through the *same* ``parse_alert_email`` parser. The output is
``list[RawPosting]`` — identical to the offline reader — so nothing downstream
(normalize/dedupe/rank) changes when this source is used.

Two design facts mirror the rest of the codebase:

1. **The Gmail service is injected** (like ``AtsClient``'s ``Fetcher`` and
   ``EdgarClient``), so every method is unit-testable fully offline against a
   fake service returning fixture payloads — no live network, no secrets in CI.
2. **Env-keyed, degrades to None** (like ``GeminiExtractor.from_env()`` and
   ``NullEnrichmentClient``): ``from_env()`` returns a working source only when
   OAuth credentials exist on disk, else ``None`` — so a run without Gmail
   credentials behaves exactly like the offline ``.eml`` path.

Compliance: the OAuth scope is **read-only** (``gmail.readonly``) — this source
only reads the user's own mailbox. It never sends, modifies, or deletes mail;
outbound draft-and-approve is a much later slice. Single-user, own account.
"""

from __future__ import annotations

import base64
from pathlib import Path

from jobfinder.jobsearch.models import RawPosting
from jobfinder.jobsearch.sources.linkedin_email import parse_alert_email

# Read-only scope ONLY. Widening this (e.g. to send/modify) is out of scope for
# the job-search tool's safe ingestion path and must be a deliberate decision.
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# Default on-disk credential location, outside the repo working tree. The
# installed-app OAuth client secret and the cached user token live here; both
# are secrets and must never be committed.
DEFAULT_CRED_DIR = Path.home() / ".config" / "job-finder"
CLIENT_SECRET_FILENAME = "gmail_client_secret.json"
TOKEN_FILENAME = "gmail_token.json"

# Gmail's messages.list caps pageSize at 500; we page until exhausted but cap the
# total scanned so a misconfigured query can't walk an entire mailbox.
_PAGE_SIZE = 100
_MAX_MESSAGES = 1000


class GmailAuthError(RuntimeError):
    """Raised when on-disk Gmail credentials exist but can't be authorized.

    Subclasses ``RuntimeError`` so the CLI's existing ``except RuntimeError``
    catch reports it as a clean setup message instead of a raw traceback.
    """


def _decode_raw(raw_b64url: str) -> bytes:
    """Decode a Gmail ``format=raw`` payload (base64url) to RFC-822 bytes.

    Gmail returns the full message base64url-encoded; pad defensively since the
    web-safe alphabet may arrive without ``=`` padding.
    """
    padded = raw_b64url + "=" * (-len(raw_b64url) % 4)
    return base64.urlsafe_b64decode(padded)


class GmailSource:
    """Reads job-alert emails from a Gmail mailbox via an injected service.

    ``service`` is a Gmail API resource (``googleapiclient.discovery.build`` for
    real use, or a fake with the same call shape in tests). Only the read-only
    ``users().messages()`` / ``users().labels()`` calls are used.
    """

    def __init__(self, service, *, user_id: str = "me") -> None:
        self._service = service
        self._user_id = user_id

    @classmethod
    def from_env(cls, *, cred_dir: str | Path | None = None) -> GmailSource | None:
        """Build a live source from on-disk OAuth credentials, or return None.

        Credentials live under ``cred_dir`` (default ``~/.config/job-finder/``):
        an installed-app client secret (``gmail_client_secret.json``) and the
        cached user token (``gmail_token.json``). Returns ``None`` when *no*
        credentials exist on disk, so a run without Gmail set up degrades to the
        offline ``.eml``/ATS paths exactly as before. When a client secret is
        present but no valid token is cached, the installed-app flow runs once
        (opens a browser for the single user to consent) and the resulting token
        is cached for subsequent runs.

        A partial or unusable credential state (e.g. a stale token whose secret
        was removed, or a revoked token) surfaces as a ``GmailAuthError`` rather
        than an opaque ``FileNotFoundError``/``RefreshError`` traceback, so the
        CLI can report a clean "authorize Gmail" message.
        """
        cred_dir = Path(cred_dir) if cred_dir else DEFAULT_CRED_DIR
        secret_file = cred_dir / CLIENT_SECRET_FILENAME
        token_file = cred_dir / TOKEN_FILENAME

        # No credentials of any kind → behave like today (offline only).
        if not token_file.exists() and not secret_file.exists():
            return None

        try:
            service = _build_service(secret_file, token_file)
        except Exception as exc:  # live OAuth/network — normalise to a clean error
            raise GmailAuthError(
                f"could not authorize Gmail with credentials under {cred_dir}: "
                f"{exc}. Place {CLIENT_SECRET_FILENAME} there and re-authorize "
                "(read-only)."
            ) from exc
        return cls(service)

    def fetch_postings(
        self, *, label: str | None = None, query: str | None = None
    ) -> list[RawPosting]:
        """Fetch matching alert emails and parse them into RawPostings.

        ``label`` is a human-facing Gmail label name (e.g. ``"job-alerts"``),
        resolved to its id; ``query`` is a Gmail search string (e.g.
        ``"from:jobalerts-noreply@linkedin.com"``). Either, both, or neither may
        be given; with neither, every message is scanned (capped). Non-alert
        messages contribute nothing — ``parse_alert_email`` returns ``[]`` for
        anything it can't read — so a broad query is tolerated.
        """
        postings: list[RawPosting] = []
        for message_id in self._list_message_ids(label=label, query=query):
            raw = self._fetch_raw(message_id)
            if raw:
                postings.extend(parse_alert_email(raw))
        return postings

    def _resolve_label_id(self, label: str) -> str | None:
        """Map a human label name to its Gmail label id (case-insensitive)."""
        response = self._service.users().labels().list(userId=self._user_id).execute()
        wanted = label.strip().lower()
        for entry in response.get("labels", []):
            if (entry.get("name") or "").strip().lower() == wanted:
                return entry.get("id")
        return None

    def _list_message_ids(self, *, label: str | None, query: str | None) -> list[str]:
        """List message ids matching the label/query, paging until exhausted.

        A label name that doesn't exist in the mailbox yields no messages (rather
        than silently scanning everything), so a typo fails closed.
        """
        label_ids: list[str] | None = None
        if label:
            label_id = self._resolve_label_id(label)
            if label_id is None:
                return []
            label_ids = [label_id]

        messages_api = self._service.users().messages()
        ids: list[str] = []
        page_token: str | None = None
        while True:
            request = messages_api.list(
                userId=self._user_id,
                labelIds=label_ids,
                q=query,
                pageSize=_PAGE_SIZE,
                pageToken=page_token,
            )
            response = request.execute()
            for message in response.get("messages", []):
                message_id = message.get("id")
                if message_id:
                    ids.append(message_id)
            if len(ids) >= _MAX_MESSAGES:
                return ids[:_MAX_MESSAGES]
            page_token = response.get("nextPageToken")
            if not page_token:
                return ids

    def _fetch_raw(self, message_id: str) -> bytes | None:
        """Fetch one message's full RFC-822 bytes (Gmail ``format=raw``)."""
        message = (
            self._service.users()
            .messages()
            .get(userId=self._user_id, id=message_id, format="raw")
            .execute()
        )
        raw = message.get("raw")
        return _decode_raw(raw) if raw else None


def _build_service(
    secret_file: Path, token_file: Path
):  # pragma: no cover - live OAuth/network
    """Build an authenticated Gmail API service from on-disk credentials.

    Loads a cached token, refreshing it when expired; falls back to the
    installed-app consent flow using the client secret. This touches the network
    and (on first run) a browser, so it is exercised only by the user-driven live
    smoke, never in CI — the testable surface is ``GmailSource`` with an injected
    service.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    scopes = [GMAIL_READONLY_SCOPE]
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
