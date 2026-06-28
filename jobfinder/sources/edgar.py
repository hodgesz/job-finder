"""Minimal SEC EDGAR client.

Scope so far: list a company's recent filings via the submissions API and
fetch the primary document for a filing. Enough for the 8-K Item 5.02 signal
module (Slice 1) and the Form D funding signal module (Slice 2) to work
against real filings.

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

Form D filings (``form == "D"`` / ``"D/A"``) carry no useful ``items``; their
structured data lives in the filing's ``primary_doc.xml`` (offering amounts,
issuer, related persons). ``parse_form_d`` reads that XML.

The network call is injected (``fetch_url``) so the parser and signal logic
can be tested fully offline against fixtures.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
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


@dataclass(frozen=True)
class RelatedPerson:
    """A related person from a Form D (officer, director, or promoter)."""

    name: str
    relationships: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FormD:
    """Structured data extracted from a Form D primary document.

    Form D reports a private securities offering. The fields we care about
    for funding signals: how much has been sold (``total_amount_sold``) versus
    how much of the offering is still open (``total_remaining``). A large
    amount sold with little remaining is freshly available budget to deploy.

    ``total_remaining`` is ``None`` when the filing reports it as "Indefinite"
    (a legitimate Form D value for open-ended offerings).
    """

    issuer_cik: str
    issuer_name: str
    accession_number: str
    total_offering_amount: float | None = None
    total_amount_sold: float | None = None
    total_remaining: float | None = None
    industry_group: str | None = None
    is_amendment: bool = False
    related_persons: list[RelatedPerson] = field(default_factory=list)


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


def _parse_amount(value: str | None) -> float | None:
    """Parse a Form D money field.

    Form D reports dollar amounts as integer strings, but ``totalRemaining``
    (and occasionally other fields) may legitimately be the literal
    ``"Indefinite"`` for open-ended offerings — that maps to ``None``.
    """
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "indefinite":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _localname(tag: str) -> str:
    """Strip any XML namespace from a tag, e.g. '{ns}cik' -> 'cik'."""
    return tag.rsplit("}", 1)[-1]


def _find(elem: ET.Element | None, name: str) -> ET.Element | None:
    """Namespace-agnostic single-child search by local tag name."""
    if elem is None:
        return None
    for child in elem.iter():
        if _localname(child.tag) == name:
            return child
    return None


def _text(elem: ET.Element | None, name: str) -> str | None:
    found = _find(elem, name)
    return found.text.strip() if found is not None and found.text else None


def parse_form_d(payload: str, *, accession_number: str = "") -> FormD:
    """Parse a Form D ``primary_doc.xml`` into a FormD record.

    Form D XML (``edgarSubmission``) carries the issuer under
    ``primaryIssuer`` and the money under ``offeringData/offeringSalesAmounts``.
    Parsing is namespace-agnostic (EDGAR has shipped the schema both with and
    without a default namespace over the years).
    """
    root = ET.fromstring(payload)

    submission_type = _text(root, "submissionType") or ""
    primary_issuer = _find(root, "primaryIssuer")
    offering_data = _find(root, "offeringData")
    sales = _find(offering_data, "offeringSalesAmounts")
    industry = _find(offering_data, "industryGroup")

    related: list[RelatedPerson] = []
    persons_list = _find(root, "relatedPersonsList")
    if persons_list is not None:
        for info in persons_list:
            if _localname(info.tag) != "relatedPersonInfo":
                continue
            name_elem = _find(info, "relatedPersonName")
            first = _text(name_elem, "firstName") or ""
            last = _text(name_elem, "lastName") or ""
            full = " ".join(p for p in (first, last) if p).strip()
            rels = [
                r.text.strip()
                for r in info.iter()
                if _localname(r.tag) == "relationship" and r.text and r.text.strip()
            ]
            if full:
                related.append(RelatedPerson(name=full, relationships=rels))

    return FormD(
        issuer_cik=_normalize_cik(_text(primary_issuer, "cik") or "0"),
        issuer_name=_text(primary_issuer, "entityName") or "",
        accession_number=accession_number,
        total_offering_amount=_parse_amount(_text(sales, "totalOfferingAmount")),
        total_amount_sold=_parse_amount(_text(sales, "totalAmountSold")),
        total_remaining=_parse_amount(_text(sales, "totalRemaining")),
        industry_group=_text(industry, "industryGroupType"),
        is_amendment=submission_type.upper() == "D/A",
        related_persons=related,
    )


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

    def recent_form_d(self, cik: str | int) -> list[Filing]:
        """Recent Form D filings (initial ``D`` and amendments ``D/A``)."""
        return [f for f in self.recent_filings(cik) if f.form in ("D", "D/A")]

    def fetch_document(self, filing: Filing) -> str:
        return self._fetch(filing.primary_document_url)

    def fetch_form_d(self, filing: Filing) -> FormD:
        """Fetch and parse a Form D filing's primary XML document."""
        payload = self._fetch(filing.primary_document_url)
        return parse_form_d(payload, accession_number=filing.accession_number)
