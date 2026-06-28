"""Public ATS job-board readers (Pillar I).

Many companies publish their open roles through a hosted applicant-tracking
system (ATS) that exposes a *public, structured* JSON job board — no login, no
scraping of rendered HTML, no terms-of-service grey area. We read three of the
most common ones:

    Greenhouse  https://boards-api.greenhouse.io/v1/boards/{token}/jobs
    Lever       https://api.lever.co/v0/postings/{token}?mode=json
    Ashby       https://api.ashbyhq.com/posting-api/job-board/{token}

The hiring-pattern *interpretation* lives in ``jobfinder.signals.ats_hiring``;
this module's only job is to fetch and normalize the three providers' differing
shapes into one ``JobPosting`` record so the signal logic is provider-agnostic.

Two design facts, mirroring the EDGAR client:

1. **Legal/ethical use.** These are the providers' own public job-board APIs,
   intended for programmatic access. We send a descriptive ``User-Agent`` and
   leave LinkedIn / TOS-violating scraping out entirely (see README "Legal &
   ethical use").
2. **The network call is injected** (``fetch_url``) so every parser and the
   signal logic are unit-testable fully offline against fixtures — no live HTTP
   in CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

# A Fetcher takes a URL and returns the response body as text (same contract as
# the EDGAR client, so a single HTTP layer could serve both later).
Fetcher = Callable[[str], str]

Provider = str  # one of PROVIDERS

# Canonical human-facing board URLs (used as evidence; the parsers read these to
# stamp each JobBoard.url, so this is the single source for board URLs).
_BOARD_URLS = {
    "greenhouse": "https://boards.greenhouse.io/{token}",
    "lever": "https://jobs.lever.co/{token}",
    "ashby": "https://jobs.ashbyhq.com/{token}",
}


@dataclass(frozen=True)
class JobPosting:
    """One open role, normalized across providers.

    ``updated_at`` is the most recent timestamp the provider exposes for the
    posting (Greenhouse ``updated_at``, Lever ``createdAt``, Ashby
    ``publishedAt``), normalized to a tz-aware UTC datetime. It is ``None`` when
    the provider omits it; recency-based logic simply skips such postings rather
    than guessing.
    """

    id: str
    title: str
    department: str | None = None
    team: str | None = None
    location: str | None = None
    commitment: str | None = None  # e.g. "Full-time", "FullTime"
    updated_at: datetime | None = None
    url: str | None = None


@dataclass(frozen=True)
class JobBoard:
    """A snapshot of one company's public job board."""

    provider: Provider
    token: str  # board slug, e.g. "stripe"
    url: str  # canonical human-facing board URL
    postings: list[JobPosting] = field(default_factory=list)


def default_fetcher(user_agent: str) -> Fetcher:
    """Build a urllib-based fetcher that sends a descriptive ``User-Agent``.

    Unlike SEC, these APIs do not mandate a contact email, but sending a
    descriptive UA is good-citizen behaviour (and some providers throttle empty
    UAs). We require a non-empty UA rather than silently defaulting.
    """
    if not user_agent or not user_agent.strip():
        raise ValueError(
            "Provide a descriptive User-Agent identifying the caller, "
            "e.g. 'job-finder research jane@example.com'."
        )

    import urllib.request

    def fetch(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (https only, fixed hosts)
            return resp.read().decode("utf-8")

    return fetch


def _parse_dt(value: object) -> datetime | None:
    """Normalize a provider timestamp to a tz-aware UTC datetime, or None.

    Handles the three shapes we see across providers:
      - ISO 8601 strings, with or without a trailing 'Z' (Greenhouse, Ashby).
      - Epoch milliseconds as an int (Lever ``createdAt``).
      - Epoch seconds as an int (defensive; some feeds use seconds).
    A naive datetime string is assumed to already be UTC.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Lever uses milliseconds; anything past ~year 33658 in seconds is
        # really milliseconds. 1e11 seconds ≈ year 5138, far beyond any real
        # posting, so values above it are milliseconds.
        seconds = value / 1000.0 if value >= 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # datetime.fromisoformat handles offsets and (3.11+) a trailing 'Z'.
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _clean(value: object) -> str | None:
    """Trim a provider string field; map empties / placeholders to None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in ("no department", "no team", "n/a"):
        return None
    return text


def parse_greenhouse(payload: str | dict, *, token: str) -> JobBoard:
    """Parse a Greenhouse ``/jobs`` payload into a JobBoard.

    Greenhouse nests location under ``location.name`` and lists one or more
    ``departments`` (each ``{id, name, ...}``); we take the first named
    department. ``updated_at`` is an ISO 8601 string with an offset.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    postings: list[JobPosting] = []
    for job in data.get("jobs", []):
        departments = [_clean(d.get("name")) for d in job.get("departments", []) or []]
        department = next((d for d in departments if d), None)
        location = None
        loc = job.get("location")
        if isinstance(loc, dict):
            location = _clean(loc.get("name"))
        postings.append(
            JobPosting(
                id=str(job.get("id", "")),
                title=_clean(job.get("title")) or "",
                department=department,
                location=location,
                updated_at=_parse_dt(job.get("updated_at")),
                url=job.get("absolute_url"),
            )
        )
    return JobBoard(
        provider="greenhouse",
        token=token,
        url=_BOARD_URLS["greenhouse"].format(token=token),
        postings=postings,
    )


def parse_lever(payload: str | list, *, token: str) -> JobBoard:
    """Parse a Lever ``/postings`` payload (a JSON list) into a JobBoard.

    Lever's title is ``text``; department/team/location/commitment live under
    ``categories``; ``createdAt`` is epoch milliseconds.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    postings: list[JobPosting] = []
    for post in data or []:
        categories = post.get("categories") or {}
        postings.append(
            JobPosting(
                id=str(post.get("id", "")),
                title=_clean(post.get("text")) or "",
                department=_clean(categories.get("department")),
                team=_clean(categories.get("team")),
                location=_clean(categories.get("location")),
                commitment=_clean(categories.get("commitment")),
                updated_at=_parse_dt(post.get("createdAt")),
                url=post.get("hostedUrl"),
            )
        )
    return JobBoard(
        provider="lever",
        token=token,
        url=_BOARD_URLS["lever"].format(token=token),
        postings=postings,
    )


def parse_ashby(payload: str | dict, *, token: str) -> JobBoard:
    """Parse an Ashby posting-API payload into a JobBoard.

    Ashby exposes ``department`` and ``team`` as flat strings, ``location`` as a
    string, ``employmentType`` (e.g. "FullTime"), and ``publishedAt`` as an ISO
    timestamp (falling back to ``updatedAt``).
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    postings: list[JobPosting] = []
    for job in data.get("jobs", []):
        postings.append(
            JobPosting(
                id=str(job.get("id", "")),
                title=_clean(job.get("title")) or "",
                department=_clean(job.get("department")),
                team=_clean(job.get("team")),
                location=_clean(job.get("location")),
                commitment=_clean(job.get("employmentType")),
                updated_at=_parse_dt(job.get("publishedAt") or job.get("updatedAt")),
                url=job.get("jobUrl") or job.get("applyUrl"),
            )
        )
    return JobBoard(
        provider="ashby",
        token=token,
        url=_BOARD_URLS["ashby"].format(token=token),
        postings=postings,
    )


# One record per provider: its machine-readable API endpoint and its parser.
# Keeping these together (rather than in parallel dicts) means adding a provider
# is a single entry and the collections can never drift out of sync. PROVIDERS
# is derived from it, so it is impossible for a provider to be advertised
# (and accepted by the CLI) without a parser behind it.
_REGISTRY: dict[str, tuple[str, Callable[..., JobBoard]]] = {
    "greenhouse": (
        "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true",
        parse_greenhouse,
    ),
    "lever": (
        "https://api.lever.co/v0/postings/{token}?mode=json",
        parse_lever,
    ),
    "ashby": (
        "https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=false",
        parse_ashby,
    ),
}

PROVIDERS = tuple(_REGISTRY)


class AtsClient:
    """Thin public-job-board reader. Network access is injected for testability."""

    def __init__(self, fetch_url: Fetcher):
        self._fetch = fetch_url

    @classmethod
    def with_user_agent(cls, user_agent: str) -> AtsClient:
        return cls(default_fetcher(user_agent))

    def fetch_board(self, provider: Provider, token: str) -> JobBoard:
        """Fetch and normalize one company's public board for `provider`."""
        provider = provider.lower()
        try:
            api_url_template, parser = _REGISTRY[provider]
        except KeyError:
            raise ValueError(
                f"Unknown ATS provider {provider!r}; expected one of {PROVIDERS}."
            ) from None
        payload = self._fetch(api_url_template.format(token=token))
        return parser(payload, token=token)
