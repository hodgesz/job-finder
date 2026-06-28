"""Wire contract for the 8-K extraction A2A service.

These models are the *formal contract* the service speaks across the A2A
boundary. They are deliberately built on ``jobfinder.schemas`` (the
framework-free domain models) and stdlib/pydantic only — no ADK, no LangGraph
— so either side of the boundary can depend on them without dragging in a
framework. The response carries ``Signal`` directly: the domain object IS the
contract, which is the whole point of having kept ``schemas.py`` pure.

The request mirrors the fields of ``sources.edgar.Filing`` that the signal
extractor actually reads (items, accession number, primary-document URL,
report date) plus the document text and company id. ``to_filing()``
reconstructs a ``Filing`` so the existing in-process extractor runs unchanged
on the far side.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from jobfinder.schemas import Signal
from jobfinder.sources.edgar import Filing


class FilingRef(BaseModel):
    """The subset of ``Filing`` the 8-K extractor needs, as a wire type.

    A frozen dataclass (``Filing``) does not serialize over JSON on its own;
    this pydantic mirror does, and ``to_filing()`` rebuilds the dataclass so
    the unchanged in-process extractor runs on the service side.
    """

    cik: str
    accession_number: str
    form: str = "8-K"
    filing_date: date | None = None
    report_date: date | None = None
    items: list[str] = Field(default_factory=list)
    primary_document: str = ""
    primary_doc_description: str | None = None

    @classmethod
    def from_filing(cls, filing: Filing) -> FilingRef:
        return cls(
            cik=filing.cik,
            accession_number=filing.accession_number,
            form=filing.form,
            filing_date=filing.filing_date,
            report_date=filing.report_date,
            items=list(filing.items),
            primary_document=filing.primary_document,
            primary_doc_description=filing.primary_doc_description,
        )

    def to_filing(self) -> Filing:
        # filing_date is required on Filing; default to report_date or epoch-min
        # if a caller omitted it (it does not affect 5.02 signal extraction).
        filing_date = self.filing_date or self.report_date or date.min
        return Filing(
            cik=self.cik,
            accession_number=self.accession_number,
            form=self.form,
            filing_date=filing_date,
            report_date=self.report_date,
            items=list(self.items),
            primary_document=self.primary_document,
            primary_doc_description=self.primary_doc_description,
        )


class EightKExtractionRequest(BaseModel):
    """A request to extract 8-K Item 5.02 signals from one filing document."""

    company_id: str = Field(..., description="Internal stable company id.")
    filing: FilingRef
    document: str = Field(..., description="The 8-K primary-document text/HTML.")
    observed_at: datetime | None = Field(
        None,
        description="When the system observed the filing; defaults to now on the service.",
    )


class EightKExtractionResponse(BaseModel):
    """The signals extracted from one filing.

    ``Signal`` is the same evidence-backed domain model the in-process pipeline
    produces and the store persists — the contract is the domain object. Each
    Signal already records how it was classified (``extracted_facts``'s
    ``extraction_method`` plus its confidence: 0.9 for an LLM pass, 0.8 for the
    regex fallback), so the response does not duplicate that at the top level.
    """

    company_id: str
    signals: list[Signal] = Field(default_factory=list)
