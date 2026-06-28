"""Minimal SEC EDGAR client.

Scope for Slice 1: list a company's recent filings via the submissions API
and fetch the primary document for a filing. Just enough for the 8-K
Item 5.02 signal module to work against real filings.

Two EDGAR facts drive this design:

1. EDGAR enforces a fair-access policy: requests MUST send a descriptive
   ``User-Agent`` (it returns HTTP 403 otherwise). We make the UA explicit
   and required rather than silently defaulting.
2. The submissions JSON (``/submissions/CIK##########.json``) stores filings
   as *parallel arrays* under ``filings.recent`` (accessionNumber, form,
   filingDate, items, primaryDocument, ...). Crucially it includes a per
   filing ``items`` string (e.g. ``"5.02"`` or ``"2.02,9.01"``), so we can
   filter 8-K filings by item *from the index alone*, without downloading
   every document.

The network call is injected (``fetch_url``) so the parser and signal logic
can be tested fully offline against fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
)

# A Fetcher takes a URL and returns the response body as text.
Fetcher = Callable[[str], str]


@dataclass(frozen=True)
class Filing:
    """One filing from the submissions index."""

    cik: str
    accession_number: str  # dashed form, e.g. "0001140361-26-015711"
    form: str  # e.g. "8-K"
    filing_date: date
    report_date: date | None
    items: list[str] = field(default_factory=list)  # e.g. ["5.02", "9.01"]
    primary_document: str = ""
    primary_doc_description: str | None = None

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def primary_document_url(self) -> str:
        return ARCHIVES_DOC_URL.format(
            cik=int(self.cik),
            accession_nodash=self.accession_nodash,
            document=self.primary_document,
        )

    @property
    def index_url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession_nodash}/{self.accession_number}-index.htm"
        )


def default_fetcher(user_agent: str) -> Fetcher:
    """Build a urllib-based fetcher that sends the required SEC User-Agent.

    `user_agent` must identify the caller per SEC policy, e.g.
    ``"job-finder research jane@example.com"``. EDGAR returns 403 without it.
    """
    if not user_agent or "@" not in user_agent:
        raise ValueError(
            "SEC requires a descriptive User-Agent including a contact email, "
            "e.g. 'job-finder research jane@example.com'."
        )

    import urllib.request

    def fetch(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (https only, fixed hosts)
            return resp.read().decode("utf-8")

    return fetch


def _normalize_cik(cik: str | int) -> str:
    """EDGAR submissions URLs use a zero-padded 10-digit CIK."""
    digits = str(cik).lstrip("CIKcik").strip()
    return digits.zfill(10)


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def parse_submissions(payload: str | dict) -> list[Filing]:
    """Parse a submissions API payload into Filing records.

    Accepts either a raw JSON string or an already-parsed dict (handy for
    fixtures). Reads the parallel arrays under ``filings.recent``.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    cik = _normalize_cik(data["cik"])
    recent = data["filings"]["recent"]

    accession = recent["accessionNumber"]
    forms = recent["form"]
    filing_dates = recent["filingDate"]
    report_dates = recent.get("reportDate", [""] * len(accession))
    items = recent.get("items", [""] * len(accession))
    primary_docs = recent.get("primaryDocument", [""] * len(accession))
    primary_descs = recent.get("primaryDocDescription", [""] * len(accession))

    filings: list[Filing] = []
    for i in range(len(accession)):
        raw_items = items[i] or ""
        filings.append(
            Filing(
                cik=cik,
                accession_number=accession[i],
                form=forms[i],
                filing_date=_parse_date(filing_dates[i]),
                report_date=_parse_date(report_dates[i]),
                items=[s.strip() for s in raw_items.split(",") if s.strip()],
                primary_document=primary_docs[i],
                primary_doc_description=primary_descs[i] or None,
            )
        )
    return filings


class EdgarClient:
    """Thin EDGAR reader. Network access is injected for testability."""

    def __init__(self, fetch_url: Fetcher):
        self._fetch = fetch_url

    @classmethod
    def with_user_agent(cls, user_agent: str) -> EdgarClient:
        return cls(default_fetcher(user_agent))

    def recent_filings(self, cik: str | int) -> list[Filing]:
        url = SUBMISSIONS_URL.format(cik10=_normalize_cik(cik))
        return parse_submissions(self._fetch(url))

    def recent_8k(self, cik: str | int, *, item: str | None = None) -> list[Filing]:
        """Recent 8-K filings, optionally restricted to those disclosing `item`.

        Includes amendments (``8-K/A``), which can themselves disclose
        Item 5.02 events. `item` is matched from the index `items` field, so
        this does not download any documents.
        """
        filings = [f for f in self.recent_filings(cik) if f.form in ("8-K", "8-K/A")]
        if item is not None:
            filings = [f for f in filings if item in f.items]
        return filings

    def fetch_document(self, filing: Filing) -> str:
        return self._fetch(filing.primary_document_url)
