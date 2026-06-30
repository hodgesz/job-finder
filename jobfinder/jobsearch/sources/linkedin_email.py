"""Parse a LinkedIn job-alert email into ``RawPosting`` records.

This is the compliance-safe LinkedIn ingestion path: LinkedIn's own job alerts
email us the new matching postings, and we parse *the email*. We never fetch or
scrape LinkedIn itself — the job URL is captured for the human to open manually.

The parser is pure and offline (stdlib ``email`` + ``html.parser`` only, no new
dependency), so it is fully unit-testable against saved ``.eml`` fixtures.

LinkedIn alert emails are HTML with one anchor per job whose ``href`` points at
``…/jobs/view/<id>/…``; the anchor text is the job title, and the company and
location follow as nearby text (often as ``Company · Location``). We key on the
``/jobs/view/`` anchor, take the company/location from the text that follows it
up to the next job anchor, and tolerate the heavy table/markup LinkedIn wraps it
in. A plain-text fallback covers the rare text-only alert.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email import message_from_bytes, message_from_string
from email.message import EmailMessage
from email.policy import default as default_policy
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser

from jobfinder.jobsearch.models import RawPosting, Source

# A LinkedIn job-view URL carries the numeric job id we use as the dedupe key.
_JOB_VIEW_RE = re.compile(r"/jobs/view/(\d+)", re.IGNORECASE)

# A "Company · Location" or "Company — Location" pairing inside one text run.
# LinkedIn uses a middot (·) most often; we also accept an en/em dash.
_COMPANY_LOCATION_SEP = re.compile(r"\s*[·•|–—]\s*")

# Text runs that are LinkedIn chrome, not job data — dropped while gathering the
# company/location text that follows a job title.
_BOILERPLATE_RE = re.compile(
    r"^(?:view job|see all|apply|easy apply|actively recruiting|"
    r"be an early applicant|view all jobs|unsubscribe|"
    r"\d+\+? (?:applicants?|connections?)|promoted|new)\.?$",
    re.IGNORECASE,
)

# A location-looking run: an explicit remote/hybrid marker, or "City, ST"/"City,
# Country" with a comma. Used to tell company from location when they arrive as
# separate runs rather than a single "Company · Location" pairing.
_LOCATION_HINT_RE = re.compile(
    r"\bremote\b|\bhybrid\b|\bon-?site\b|,\s*[A-Za-z]", re.IGNORECASE
)
_WORKPLACE_RE = re.compile(r"\b(remote|hybrid|on-?site)\b", re.IGNORECASE)


def _decode(raw: str | bytes) -> EmailMessage:
    """Parse raw RFC-822 bytes/str into an EmailMessage (modern policy)."""
    if isinstance(raw, bytes):
        msg = message_from_bytes(raw, policy=default_policy)
    else:
        msg = message_from_string(raw, policy=default_policy)
    return msg  # type: ignore[return-value]


def _best_body(msg: EmailMessage) -> tuple[str, bool]:
    """Return the richest body to parse and whether it is HTML.

    Prefers ``text/html`` (LinkedIn's real alert format); falls back to
    ``text/plain``. ``get_body`` walks multipart/alternative for us.
    """
    html_part = msg.get_body(preferencelist=("html",))
    if html_part is not None:
        return html_part.get_content(), True
    text_part = msg.get_body(preferencelist=("plain",))
    if text_part is not None:
        return text_part.get_content(), False
    # Single-part message: use whatever it is, guessing by content type.
    content = msg.get_content() if not msg.is_multipart() else ""
    return content, msg.get_content_type() == "text/html"


def _alert_keyword(subject: str | None) -> str | None:
    """Pull the saved-search term out of an alert subject, if present.

    LinkedIn subjects look like ``"VP of AI": ExampleCo is hiring`` or
    ``Your job alert for vp of ai``; we take a quoted phrase first, else the text
    after "job alert for".
    """
    if not subject:
        return None
    quoted = re.search(r"[\"“]([^\"”]{2,60})[\"”]", subject)
    if quoted:
        return quoted.group(1).strip()
    after = re.search(r"job alert for\s+(.{2,60})", subject, re.IGNORECASE)
    if after:
        return after.group(1).strip(" .:-")
    return None


class _LinkedInHtmlParser(HTMLParser):
    """Walks alert HTML into an ordered token stream.

    Emits a ``("job", id, url, title)`` token when an anchor pointing at
    ``/jobs/view/<id>`` closes, and ``("text", run)`` tokens for other visible
    text. ``normalize`` is done by the assembler; this only flattens the markup.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[tuple] = []
        self._a_href: str | None = None
        self._a_text: list[str] = []
        self._skip_depth = 0  # inside <style>/<script>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script"):
            self._skip_depth += 1
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            self._a_href = href
            self._a_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script") and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "a":
            href = self._a_href or ""
            text = _clean_run("".join(self._a_text))
            match = _JOB_VIEW_RE.search(href)
            if match and text:
                self.tokens.append(("job", match.group(1), href, text))
            elif text:
                self.tokens.append(("text", text))
            self._a_href = None
            self._a_text = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._a_href is not None:
            self._a_text.append(data)
            return
        run = _clean_run(data)
        if run:
            self.tokens.append(("text", run))


def _clean_run(text: str) -> str:
    """Collapse whitespace and unescape entities in one text run."""
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _split_company_location(
    runs: list[str],
) -> tuple[str | None, str | None, str | None]:
    """From the text runs after a job title, derive (company, location, snippet).

    Handles the common "Company · Location" single run, and the case where they
    arrive as separate runs (first non-boilerplate run = company; a later
    location-looking run = location). Anything left over is joined as a snippet.
    """
    meaningful = [r for r in runs if r and not _BOILERPLATE_RE.match(r)]
    company: str | None = None
    location: str | None = None
    leftovers: list[str] = []

    for run in meaningful:
        parts = [p for p in _COMPANY_LOCATION_SEP.split(run) if p]
        if company is None and len(parts) >= 2:
            # "Company · Location" in one run.
            company, location = parts[0], parts[1]
            leftovers.extend(parts[2:])
            continue
        if company is None:
            company = run
            continue
        if location is None and _LOCATION_HINT_RE.search(run):
            location = run
            continue
        leftovers.append(run)

    snippet = " ".join(leftovers) or None
    return company, location, snippet


def _workplace_type(location: str | None) -> str | None:
    if not location:
        return None
    m = _WORKPLACE_RE.search(location)
    return m.group(1).lower().replace("onsite", "on-site") if m else None


def _assemble(
    tokens: list[tuple], *, keyword: str | None, posted_at: datetime | None
) -> list[RawPosting]:
    """Turn the flat token stream into one RawPosting per job-view anchor.

    Each job token starts a posting; the text tokens up to the next job token are
    its company/location/snippet. LinkedIn often links the same job from BOTH a
    descriptive title anchor and a generic "View job" button (same id); these are
    collapsed to one posting that keeps the *descriptive* title (not "View job",
    whichever anchor comes first) and the company/location runs from whichever
    anchor carried them.
    """
    # job_id -> accumulator, in first-seen order.
    acc: dict[str, dict] = {}
    order: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i][0] != "job":
            i += 1
            continue
        _, job_id, url, title = tokens[i]
        # Gather following text runs until the next job anchor.
        runs: list[str] = []
        j = i + 1
        while j < n and tokens[j][0] != "job":
            runs.append(tokens[j][1])
            j += 1

        title_is_chrome = bool(_BOILERPLATE_RE.match(title))
        if job_id not in acc:
            order.append(job_id)
            acc[job_id] = {"title": title, "url": url, "runs": runs}
        else:
            entry = acc[job_id]
            # Upgrade a chrome title ("View job") to a descriptive one.
            if _BOILERPLATE_RE.match(entry["title"]) and not title_is_chrome:
                entry["title"] = title
                entry["url"] = url
            # Fill company/location runs from whichever anchor carried them.
            if not entry["runs"] and runs:
                entry["runs"] = runs
        i = j

    postings: list[RawPosting] = []
    for job_id in order:
        entry = acc[job_id]
        company, location, snippet = _split_company_location(entry["runs"])
        postings.append(
            RawPosting(
                title=entry["title"],
                company=company or "",
                source=Source.LINKEDIN_ALERT,
                url=entry["url"],
                source_job_id=job_id,
                location=location,
                workplace_type=_workplace_type(location),
                posted_at=posted_at,
                snippet=snippet,
                alert_keyword=keyword,
            )
        )
    return postings


# Plain-text fallback: lines like "Title - Company - Location" or "Title at
# Company (Location)". Best-effort only; real LinkedIn alerts are HTML.
_PLAINTEXT_JOB_RE = re.compile(
    r"^(?P<title>.+?)\s+(?:-|–|—|at)\s+(?P<company>.+?)"
    r"(?:\s*[-–—(]\s*(?P<location>[^)]+?)\)?)?$"
)


def _assemble_plaintext(
    body: str, *, keyword: str | None, posted_at: datetime | None
) -> list[RawPosting]:
    postings: list[RawPosting] = []
    for line in body.splitlines():
        run = _clean_run(line)
        if not run or _BOILERPLATE_RE.match(run):
            continue
        m = _PLAINTEXT_JOB_RE.match(run)
        if not m or len(run) > 160:
            continue
        location = (m.group("location") or "").strip() or None
        postings.append(
            RawPosting(
                title=m.group("title").strip(),
                company=(m.group("company") or "").strip(),
                source=Source.LINKEDIN_ALERT,
                location=location,
                workplace_type=_workplace_type(location),
                posted_at=posted_at,
                alert_keyword=keyword,
            )
        )
    return postings


def parse_alert_email(raw_eml: str | bytes) -> list[RawPosting]:
    """Parse one LinkedIn job-alert email into its postings.

    Returns an empty list for a non-job email or one we can't read, rather than
    raising — a folder of mixed mail should skip cleanly. The LinkedIn job URL is
    stored on each posting for *manual* review; it is never fetched here.
    """
    msg = _decode(raw_eml)
    keyword = _alert_keyword(msg.get("subject"))
    posted_at = _parse_date_header(msg.get("date"))
    body, is_html = _best_body(msg)
    if not body:
        return []
    if is_html:
        parser = _LinkedInHtmlParser()
        parser.feed(body)
        postings = _assemble(parser.tokens, keyword=keyword, posted_at=posted_at)
        if postings:
            return postings
        # HTML carried no job-view anchors: fall back to the text/plain
        # alternative, which multipart alerts often include with the same jobs.
        plain = msg.get_body(preferencelist=("plain",))
        if plain is not None:
            return _assemble_plaintext(
                plain.get_content(), keyword=keyword, posted_at=posted_at
            )
        return []
    return _assemble_plaintext(body, keyword=keyword, posted_at=posted_at)


def _parse_date_header(value: str | None) -> datetime | None:
    """Parse the email Date header into a tz-aware (UTC) datetime, or None.

    A Date header with the RFC 2822 ``-0000`` "no zone" marker (common in bulk
    mail, including LinkedIn's) parses to a *naive* datetime; we stamp it UTC so
    every ``posted_at`` is tz-aware and comparable downstream (``_merge`` picks
    the most recent across sources; mixing naive + aware would raise TypeError).
    """
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except TypeError, ValueError, OverflowError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
