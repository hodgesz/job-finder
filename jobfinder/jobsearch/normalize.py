"""Normalize and de-duplicate postings across sources into canonical jobs.

The same role often appears in a LinkedIn alert *and* on the company's ATS board
(and possibly twice within one alert). This module folds those into one
``CanonicalJob`` while preserving every source's provenance, so the ranked list
shows one row per real job — with a real apply URL when an ATS board provided one.

Dedupe is layered and conservative (Slice A uses no embeddings):

1. **Hard keys** — exact apply URL, or ``source:source_job_id`` (LinkedIn job id /
   ATS requisition id). A hard-key collision is the same job, full stop.
2. **Soft key** — ``normalized_company`` + a token-set signature of the
   normalized title + a *coarse* location bucket. This catches the LinkedIn-vs-ATS
   pairing where the ids differ but it's plainly the same posting (and where the
   two sources phrase the location differently — "Remote (United States)" vs
   "Remote"), without risking false merges across different roles at the same
   company (different titles → different signatures).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from jobfinder.jobsearch.models import CanonicalJob, RawPosting, Source
from jobfinder.sources.ats import JobBoard


def _as_utc(dt: datetime) -> datetime:
    """Stamp a naive datetime as UTC so naive + aware values stay comparable."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


_WORKPLACE_RE = re.compile(r"\b(remote|hybrid|on-?site)\b", re.IGNORECASE)


def _workplace_from_location(location: str | None) -> str | None:
    """Derive remote/hybrid/on-site from a location string, when stated."""
    if not location:
        return None
    m = _WORKPLACE_RE.search(location)
    return m.group(1).lower().replace("onsite", "on-site") if m else None


# Seniority/role tokens that are noise for *matching the same posting* — dropped
# from the soft-key signature so "VP, AI & Data" and "VP - AI and Data" align.
# (Scoring reads the original title; this only affects dedupe grouping.)
_TITLE_NOISE = {"the", "of", "and", "a", "an", "for", "to"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# Common company-suffix noise so "ExampleCo, Inc." == "ExampleCo".
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:inc|inc\.|llc|l\.l\.c\.|ltd|ltd\.|corp|corp\.|corporation|co|co\.|"
    r"company|gmbh|plc|sa|ag)\b",
    re.IGNORECASE,
)

_ATS_PROVIDER_SOURCE = {
    "greenhouse": Source.GREENHOUSE,
    "lever": Source.LEVER,
    "ashby": Source.ASHBY,
}


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/extra whitespace; expand a couple of aliases.

    Keeps the role words (VP, AI, data) — only flattens formatting — so the
    normalized form is still human-readable and stable across "VP of AI" /
    "VP, AI" / "Vp  AI"."""
    text = title.lower().replace("&", " and ").replace("/", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_company(company: str) -> str:
    text = company.lower().replace("&", " and ")
    text = _COMPANY_SUFFIX_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _title_signature(normalized_title: str) -> frozenset[str]:
    """Order-independent token signature of a normalized title for soft matching."""
    return frozenset(t for t in normalized_title.split() if t and t not in _TITLE_NOISE)


def _normalize_location(location: str | None) -> str:
    if not location:
        return ""
    return _WS_RE.sub(" ", location.lower()).strip()


def _location_bucket(location: str | None) -> str:
    """Coarse location key for soft dedupe.

    A LinkedIn alert says "Remote (United States)" where an ATS board says just
    "Remote" for the *same* role; keying the soft match on the full normalized
    string would wrongly split them. Any run mentioning "remote" collapses to the
    single bucket ``"remote"`` so the two sources merge; non-remote locations keep
    their normalized form (a conservative tertiary disambiguator that still lets
    distinct on-site roles at one company stay separate)."""
    norm = _normalize_location(location)
    if re.search(r"\bremote\b", norm):
        return "remote"
    return norm


def board_to_raw(board: JobBoard) -> list[RawPosting]:
    """Adapt a core ``JobBoard`` (ATS) into this tool's RawPostings.

    Reuses the existing ``AtsClient``/``JobPosting`` rather than re-fetching; the
    ATS requisition id becomes ``source_job_id`` so an ATS posting can dedupe
    against the same role seen in a LinkedIn alert via the soft key.
    """
    source = _ATS_PROVIDER_SOURCE.get(board.provider, Source.MANUAL)
    raws: list[RawPosting] = []
    for post in board.postings:
        raws.append(
            RawPosting(
                title=post.title,
                company=board.token,  # the board slug names the company
                source=source,
                url=post.url,
                source_job_id=post.id or None,
                location=post.location,
                department=post.department,
                posted_at=post.updated_at,
            )
        )
    return raws


def _hard_keys(raw: RawPosting) -> list[str]:
    keys: list[str] = []
    if raw.url:
        keys.append(f"url:{raw.url.strip().rstrip('/').lower()}")
    if raw.source_job_id:
        keys.append(f"{raw.source.value}:{raw.source_job_id}")
    return keys


def _soft_key(raw: RawPosting) -> tuple[str, frozenset[str], str]:
    return (
        normalize_company(raw.company),
        _title_signature(normalize_title(raw.title)),
        _location_bucket(raw.location),
    )


def _merge(group: list[RawPosting]) -> CanonicalJob:
    """Build one CanonicalJob from a group of duplicate RawPostings.

    Prefers an ATS apply URL over a LinkedIn job URL; takes the first non-empty
    value for each display field; keeps the most recent ``posted_at``.
    """
    # Stable, source-priority order: ATS first (real apply form), LinkedIn last.
    ats_first = sorted(group, key=lambda r: r.source == Source.LINKEDIN_ALERT)
    primary = ats_first[0]

    def first(attr: str) -> str | None:
        for raw in ats_first:
            value = getattr(raw, attr)
            if value:
                return value
        return None

    # Company *display* name prefers a LinkedIn alert's real name ("Stripe, Inc.")
    # over an ATS board slug ("stripe"), which `board_to_raw` puts in `company`.
    # (The slug still drives soft-key dedupe; this only affects what's shown.)
    company = next(
        (r.company for r in group if r.source == Source.LINKEDIN_ALERT and r.company),
        None,
    ) or (first("company") or primary.company)

    # Best apply URL: an ATS url beats a LinkedIn job-view url.
    apply_url = next(
        (r.url for r in ats_first if r.url and r.source != Source.LINKEDIN_ALERT),
        None,
    ) or first("url")

    # Location: prefer whichever source says "remote" over a bare city — an ATS
    # board often lists a HQ city ("Austin, TX") for a role a LinkedIn alert
    # correctly flagged "Remote", and a plain ATS-first pick would mis-score the
    # merged role as on-site. Fall back to the first non-empty location otherwise.
    location = next(
        (
            r.location
            for r in ats_first
            if r.location and _location_bucket(r.location) == "remote"
        ),
        None,
    ) or first("location")
    workplace_type = next(
        (r.workplace_type for r in ats_first if r.workplace_type), None
    ) or _workplace_from_location(location)

    # Pick the most recent post date. Coerce to tz-aware UTC first: a LinkedIn
    # Date header with the RFC 2822 "-0000" no-zone marker parses naive, while
    # ATS timestamps are aware — comparing the two raises TypeError, and a merged
    # group routinely mixes both (the LI↔ATS pairing this module exists for).
    posted_dates = [_as_utc(r.posted_at) for r in group if r.posted_at]
    posted_at = max(posted_dates) if posted_dates else None

    return CanonicalJob(
        company=company,
        title=primary.title,
        normalized_title=normalize_title(primary.title),
        location=location,
        workplace_type=workplace_type,
        department=first("department"),
        best_apply_url=apply_url,
        posted_at=posted_at,
        sources=list(group),
    )


def canonicalize(
    raw_postings: list[RawPosting], ats_boards: list[JobBoard] | None = None
) -> list[CanonicalJob]:
    """Merge LinkedIn-alert and ATS postings into de-duplicated canonical jobs.

    Returns one ``CanonicalJob`` per distinct real role, each carrying all the
    ``RawPosting`` sources that mapped onto it. Input order is preserved by
    first appearance.
    """
    all_raws = list(raw_postings)
    for board in ats_boards or []:
        all_raws.extend(board_to_raw(board))

    # Union-find over postings: any two postings that share a hard key (exact URL
    # or source:source_job_id) or a soft key (company + title-signature + coarse
    # location bucket) belong to the same role. A full union-find — not a one-pass
    # group-id lookup — so a third posting that shares a hard key with one group
    # and a soft key with another transitively merges both into one role.
    parent = list(range(len(all_raws)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the earlier root so groups preserve first-seen order.
            parent[max(ra, rb)] = min(ra, rb)

    key_owner: dict = {}
    for i, raw in enumerate(all_raws):
        keys: list = list(_hard_keys(raw))
        soft = _soft_key(raw)
        # Only use the soft key when it carries BOTH a real company and a real
        # title signature. A blank company (LinkedIn alert whose metadata didn't
        # parse) or blank title must not soft-collapse with another blank one —
        # that would false-merge two unrelated roles. (Such postings can still
        # merge via a hard key: exact URL or source:source_job_id.)
        if soft[0] and soft[1]:
            keys.append(soft)
        for key in keys:
            if key in key_owner:
                union(i, key_owner[key])
            else:
                key_owner[key] = i

    # Bucket postings by root, preserving the order in which each root first appears.
    groups: dict[int, list[RawPosting]] = {}
    order: list[int] = []
    for i, raw in enumerate(all_raws):
        root = find(i)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(raw)

    return [_merge(groups[root]) for root in order]
