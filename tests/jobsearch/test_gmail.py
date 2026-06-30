"""Hermetic tests for the live Gmail alert-email source.

These drive ``GmailSource`` against a *fake* Gmail service that mirrors the real
``googleapiclient`` chained-builder shape
(``service.users().messages().list(...).execute()`` etc.), so there is no live
network, no OAuth, and no secrets — exactly like the ATS/EDGAR fetcher tests. We
assert that the source produces the same ``RawPosting`` output as the offline
``.eml`` reader, that label/query scoping is passed through correctly, and that
``from_env`` degrades to ``None`` when no credentials are on disk.
"""

import base64
from pathlib import Path

import pytest

from jobfinder.jobsearch.models import Source
from jobfinder.jobsearch.sources.gmail import (
    GMAIL_READONLY_SCOPE,
    TOKEN_FILENAME,
    GmailAuthError,
    GmailSource,
    _decode_raw,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _raw_b64url(eml_bytes: bytes) -> str:
    """Encode RFC-822 bytes the way Gmail's ``format=raw`` returns them."""
    return base64.urlsafe_b64encode(eml_bytes).decode("ascii")


class _FakeMessages:
    def __init__(self, store, calls):
        self._store = store
        self._calls = calls

    def list(self, *, userId, labelIds=None, q=None, pageSize=None, pageToken=None):
        self._calls.append(
            {"op": "list", "labelIds": labelIds, "q": q, "pageToken": pageToken}
        )
        return _FakeExec(
            lambda: self._store.list_page(
                label_ids=labelIds, q=q, page_token=pageToken, page_size=pageSize
            )
        )

    def get(self, *, userId, id, format):
        self._calls.append({"op": "get", "id": id, "format": format})
        return _FakeExec(lambda: self._store.get_message(id, fmt=format))


class _FakeLabels:
    def __init__(self, store, calls):
        self._store = store
        self._calls = calls

    def list(self, *, userId):
        self._calls.append({"op": "labels.list"})
        return _FakeExec(lambda: {"labels": self._store.labels})


class _FakeUsers:
    def __init__(self, store, calls):
        self._store = store
        self._calls = calls

    def messages(self):
        return _FakeMessages(self._store, self._calls)

    def labels(self):
        return _FakeLabels(self._store, self._calls)


class _FakeExec:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class FakeGmailService:
    """A minimal in-memory Gmail API double.

    ``messages`` maps id -> {"raw": b64url, "labelIds": [...], "text": "..."}.
    ``labels`` is the labels.list payload. ``q`` matching is a simple substring
    over an optional per-message ``text`` (enough to prove the query is honored).
    ``page_size`` drives synthetic paging so multi-page listing is exercised.
    """

    def __init__(self, messages, labels=(), page_size=None):
        self._messages = messages
        self.labels = list(labels)
        self._page_size = page_size
        self.calls = []

    def users(self):
        return _FakeUsers(self, self.calls)

    def _matching_ids(self, label_ids, q):
        ids = []
        for mid, msg in self._messages.items():
            if label_ids and not (set(label_ids) & set(msg.get("labelIds", []))):
                continue
            if q and q.lower() not in (msg.get("text", "") or "").lower():
                continue
            ids.append(mid)
        return ids

    def list_page(self, *, label_ids, q, page_token, page_size):
        ids = self._matching_ids(label_ids, q)
        size = page_size if self._page_size is None else self._page_size
        start = int(page_token) if page_token else 0
        page = ids[start : start + size]
        result = {"messages": [{"id": mid} for mid in page]}
        nxt = start + size
        if nxt < len(ids):
            result["nextPageToken"] = str(nxt)
        return result

    def get_message(self, message_id, *, fmt):
        assert fmt == "raw"  # the source must request the full RFC-822 body
        return {"raw": self._messages[message_id]["raw"]}


def _service_from_eml(names, *, labels=(), page_size=None, label_map=None):
    """Build a FakeGmailService from named .eml fixtures.

    ``label_map`` assigns labelIds per message id; ``labels`` is the labels.list
    payload (name<->id). Each message's searchable ``text`` is its raw content,
    so a ``q`` substring over sender/subject is honored by the fake.
    """
    messages = {}
    for i, name in enumerate(names):
        eml_bytes = (FIXTURES / name).read_bytes()
        mid = f"m{i}"
        messages[mid] = {
            "raw": _raw_b64url(eml_bytes),
            "labelIds": (label_map or {}).get(mid, []),
            "text": eml_bytes.decode("utf-8", "replace"),
        }
    return FakeGmailService(messages, labels=labels, page_size=page_size)


def test_single_message_parses_like_eml_reader():
    service = _service_from_eml(["li_alert_single.eml"])
    postings = GmailSource(service).fetch_postings()
    assert len(postings) == 1
    assert postings[0].title == "VP of AI & Analytics"
    assert postings[0].company == "Umbrella Inc"
    assert postings[0].source is Source.LINKEDIN_ALERT
    # The LinkedIn URL is captured for manual review (never auto-fetched).
    assert "/jobs/view/3899999999" in (postings[0].url or "")


def test_multi_message_inbox_aggregates_all_postings():
    service = _service_from_eml(["li_alert_single.eml", "li_alert_multi.eml"])
    postings = GmailSource(service).fetch_postings()
    # The single alert (1 job) + the multi alert (>1 job) all come through.
    titles = [p.title for p in postings]
    assert "VP of AI & Analytics" in titles
    assert len(postings) > 1


def test_non_alert_message_contributes_nothing():
    # A mixed inbox: a real alert plus a non-job email. Only the alert yields
    # postings; the non-job message parses to [] rather than erroring.
    service = _service_from_eml(["li_alert_single.eml", "not_a_job.eml"])
    postings = GmailSource(service).fetch_postings()
    assert [p.title for p in postings] == ["VP of AI & Analytics"]


def test_empty_inbox_yields_no_postings():
    service = FakeGmailService(messages={})
    assert GmailSource(service).fetch_postings() == []


def test_query_filters_messages_and_is_passed_through():
    service = _service_from_eml(["li_alert_single.eml", "li_alert_multi.eml"])
    # The fake filters on a substring of each message; the real Gmail API parses
    # operators server-side — what matters here is that the query string reaches
    # ``messages.list`` verbatim and that matching messages come back.
    query = "jobalerts-noreply@linkedin.com"
    postings = GmailSource(service).fetch_postings(query=query)
    assert any(c["op"] == "list" and c["q"] == query for c in service.calls)
    assert postings  # both fixtures share that sender


def test_query_with_no_matches_returns_empty():
    service = _service_from_eml(["li_alert_single.eml"])
    postings = GmailSource(service).fetch_postings(query="from:nobody@example.com")
    assert postings == []


def test_label_resolves_name_to_id_and_filters():
    service = _service_from_eml(
        ["li_alert_single.eml", "li_alert_multi.eml"],
        labels=[{"id": "Label_7", "name": "job-alerts"}],
        label_map={"m0": ["Label_7"]},  # only the single alert carries the label
    )
    postings = GmailSource(service).fetch_postings(label="job-alerts")
    # Only the labelled message is read.
    assert [p.title for p in postings] == ["VP of AI & Analytics"]
    # The human label name was resolved to its id and passed as labelIds.
    assert any(c.get("labelIds") == ["Label_7"] for c in service.calls)


def test_label_matching_is_case_insensitive():
    service = _service_from_eml(
        ["li_alert_single.eml"],
        labels=[{"id": "Label_7", "name": "Job-Alerts"}],
        label_map={"m0": ["Label_7"]},
    )
    postings = GmailSource(service).fetch_postings(label="job-alerts")
    assert len(postings) == 1


def test_unknown_label_fails_closed():
    # A typo'd label must yield zero messages, not silently scan the whole inbox.
    service = _service_from_eml(
        ["li_alert_single.eml"],
        labels=[{"id": "Label_7", "name": "job-alerts"}],
        label_map={"m0": ["Label_7"]},
    )
    postings = GmailSource(service).fetch_postings(label="does-not-exist")
    assert postings == []
    # The source short-circuited before listing any messages.
    assert not any(c["op"] == "get" for c in service.calls)


def test_paging_walks_all_pages():
    # Force one message per page so the nextPageToken loop is exercised.
    service = _service_from_eml(
        ["li_alert_single.eml", "li_alert_multi.eml"], page_size=1
    )
    postings = GmailSource(service).fetch_postings()
    list_calls = [c for c in service.calls if c["op"] == "list"]
    assert len(list_calls) >= 2  # more than one page was fetched
    assert postings


def test_decode_raw_handles_unpadded_base64url():
    # Gmail's web-safe base64 may arrive without '=' padding; decode must cope
    # and round-trip bytes that use the URL-safe '-'/'_' alphabet.
    payload = b"\xfb\xff\xfe job alert body"
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    assert _decode_raw(encoded) == payload


def test_from_env_returns_none_without_credentials(tmp_path):
    # No token and no client secret on disk → degrade to None (offline behaviour).
    assert GmailSource.from_env(cred_dir=tmp_path) is None


def test_from_env_partial_credentials_raise_clean_auth_error(tmp_path):
    # A stale token whose client secret was removed must surface as a clean
    # GmailAuthError (a RuntimeError the CLI catches), NOT an opaque
    # FileNotFoundError/RefreshError traceback from the live OAuth build.
    (tmp_path / TOKEN_FILENAME).write_text("{not valid creds}")
    with pytest.raises(GmailAuthError):
        GmailSource.from_env(cred_dir=tmp_path)


def test_readonly_scope_is_read_only():
    # Guard against an accidental scope widening to send/modify.
    assert GMAIL_READONLY_SCOPE.endswith("gmail.readonly")
