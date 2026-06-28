"""8-K Item 5.02 signal extraction.

Item 5.02 of Form 8-K covers "Departure of Directors or Certain Officers;
Election of Directors; Appointment of Certain Officers; Compensatory
Arrangements of Certain Officers." It is the cleanest public source of
executive-transition signals.

The signal we most care about is a *leadership vacuum*: a departure with no
named successor in the same filing. The plan's interpretation table:

    8-K Item 5.02(b) departure with no 5.02(c) appointment
        -> possible open executive search / succession gap

So this module:
  1. Confirms the filing discloses Item 5.02 (from the index `items` field).
  2. Strips the document to text.
  3. Detects departure language and appointment language.
  4. Classifies successor_present vs. successor_missing.
  5. Emits a `Signal` (8k_exec_departure / 8k_exec_appointment) with evidence.

`parse_item_502` is deliberately conservative keyword/heuristic matching: it
is deterministic, testable, and cheap, and serves as both a pre-filter and an
offline fallback. `signals_from_filing` routes classification through
`jobfinder.signals.extraction.extract_events`, which prefers a structured LLM
pass (better at dense filing prose) and degrades to this regex parser when no
LLM is configured or a call fails.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from jobfinder.schemas import Evidence, Signal
from jobfinder.sources.edgar import Filing

ITEM_502 = "5.02"

# Match the actual item label "Item 5.02", not a bare "5.02" substring (which
# would also fire on "15.02", "5.02%", "$15.02", etc.). The index `items`
# field remains the authoritative source; this is only the body fallback.
_ITEM_502_LABEL_RE = re.compile(r"\bitem\s*5\.02\b", re.IGNORECASE)

# Roles we care most about for executive-opportunity detection.
_EXEC_ROLE_PATTERNS = {
    "CEO": r"chief executive officer|\bCEO\b",
    "CFO": r"chief financial officer|\bCFO\b|principal financial officer",
    "COO": r"chief operating officer|\bCOO\b",
    "CRO": r"chief revenue officer|\bCRO\b",
    "President": r"\bpresident\b",
    "Controller": r"\bcontroller\b|principal accounting officer",
    "Director": r"\bdirector\b",
}

# Language that indicates someone is actually leaving.
#
# These must be *event*-shaped, not compensation wording. Item 5.02(e) covers
# compensatory arrangements, so a filing can mention "severance upon
# termination" or "retirement benefits" with nobody departing. We therefore
# match departure verbs (resigned, stepped down, transitioned from a role) and
# gate the comp-prone words: bare "termination"/"retirement" are excluded;
# "termination" must be tied to employment, and "retirement" to a person.
_DEPARTURE_RE = re.compile(
    r"\b(?:"
    r"resign(?:ed|ation|s)?"
    r"|depart(?:ed|ure|s)?"
    r"|step(?:ping|ped)?\s+down"
    r"|will\s+transition\s+from"
    r"|transition(?:ed|ing|s)?\s+from\s+(?:his|her|their)\s+role"
    r"|removed?\s+(?:as|from)\b"
    r"|retir(?:e|ed|es|ing)\b"  # verb forms only, not the noun "retirement"
    r"|(?:his|her|their)\s+retirement"  # "announced her retirement"
    r"|terminat\w*\s+(?:his|her|their|the)\s+employment"
    r"|employment[^.]{0,30}\bterminat\w+"
    r")",
    re.IGNORECASE,
)

# Language that indicates someone is being put into a role (a successor).
_APPOINTMENT_RE = re.compile(
    r"\b(appoint(?:ed|ment|s)?|elect(?:ed|ion|s)?|named?|"
    r"will\s+become|promoted?|assume(?:d|s)?\s+the\s+role|"
    r"hired?|join(?:ed|s)?\s+as)\b",
    re.IGNORECASE,
)

# The standard Item 5.02 caption is itself "Departure of Directors or Certain
# Officers; Election of Directors; Appointment of Certain Officers;
# Compensatory Arrangements of Certain Officers". Left in place, this
# boilerplate matches both the departure and appointment regexes on EVERY
# 5.02 filing, making everything look successor-present. Because the caption
# is fixed text, we strip it (and any "Item 5.02" label) by matching its known
# clauses rather than a fragile span regex.
_ITEM_502_CAPTION_CLAUSES = [
    r"item\s*5\.02",
    r"departure\s+of\s+directors?\s+or\s+certain\s+officers",
    r"election\s+of\s+directors?",
    r"appointment\s+of\s+certain\s+officers",
    r"compensatory\s+arrangements?\s+of\s+certain\s+officers",
]
_ITEM_502_CAPTION_RE = re.compile(
    r"(?:" + r"|".join(_ITEM_502_CAPTION_CLAUSES) + r")[;.\s]*",
    re.IGNORECASE,
)


def _strip_item_caption(text: str) -> str:
    """Remove the boilerplate Item 5.02 heading so event regexes only see the
    actual disclosed narrative, not the caption's own 'Appointment/Election'."""
    return _ITEM_502_CAPTION_RE.sub(" ", text)


@dataclass(frozen=True)
class ExecEvents:
    """Structured result of parsing an Item 5.02 document."""

    has_item_502: bool
    has_departure: bool
    has_appointment: bool
    roles: list[str]
    successor_present: bool

    @property
    def is_leadership_vacuum(self) -> bool:
        """A departure with no accompanying appointment -> succession gap."""
        return self.has_departure and not self.successor_present


def strip_html(document: str) -> str:
    """Collapse an HTML (or plain-text) filing into normalized text."""
    text = re.sub(r"(?is)<script.*?</script>", " ", document)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_item_502(document: str, *, item_known: bool | None = None) -> ExecEvents:
    """Parse a filing document for Item 5.02 executive events.

    `item_known` lets the caller assert (from the index `items` field) that
    the filing discloses Item 5.02 even if the literal string is formatted
    oddly in the body.
    """
    text = strip_html(document)

    has_item = bool(item_known) or bool(_ITEM_502_LABEL_RE.search(text))

    # Detect events against the body with the boilerplate caption removed, so
    # the caption's own "Departure/Election/Appointment" wording is not read
    # as disclosed events.
    body = _strip_item_caption(text)
    has_departure = bool(_DEPARTURE_RE.search(body))
    has_appointment = bool(_APPOINTMENT_RE.search(body))

    roles = [
        role
        for role, pat in _EXEC_ROLE_PATTERNS.items()
        if re.search(pat, body, re.IGNORECASE)
    ]

    return ExecEvents(
        has_item_502=has_item,
        has_departure=has_departure,
        has_appointment=has_appointment,
        roles=roles,
        # Treat an appointment in the same filing as a present successor.
        successor_present=has_appointment,
    )


def discloses_item_502(items: list[str], document: str) -> bool:
    """Cheap pre-filter: does this filing disclose Item 5.02 at all?

    The index ``items`` field is authoritative; fall back to scanning the body
    for the literal item label. Shared by ``signals_from_filing`` and the A2A
    graph's pre-filter node so both gate the (possibly LLM-backed) extractor
    the same way.
    """
    if ITEM_502 in items:
        return True
    return bool(_ITEM_502_LABEL_RE.search(strip_html(document)))


def _utcnow() -> datetime:
    # Wrapped so callers/tests can monkeypatch if they need determinism.
    return datetime.now(timezone.utc)


def signals_from_filing(
    filing: Filing,
    document: str,
    *,
    company_id: str,
    observed_at: datetime | None = None,
    extractor=None,
) -> list[Signal]:
    """Produce Signal(s) from one 8-K filing + its document text.

    Event classification goes through `extract_events`, which prefers an LLM
    extractor (better at filing prose) and falls back to the deterministic
    regex parser when no LLM is available. Pass `extractor` to inject one;
    otherwise it is resolved from the environment (GEMINI_API_KEY) or the
    regex fallback.

    Returns an empty list if the filing does not disclose Item 5.02 or has no
    detectable executive event.
    """
    # Cheap pre-filter: only filings whose index discloses 5.02 reach the
    # (possibly LLM-backed) extractor. The `items` field is authoritative.
    item_known = ITEM_502 in filing.items
    if not discloses_item_502(filing.items, document):
        return []

    # Lazy import avoids a circular dependency (extraction imports this module).
    from jobfinder.signals.extraction import extract_events

    events = extract_events(document, extractor=extractor, item_known=item_known)
    if not (events.has_departure or events.has_appointment):
        return []

    observed = observed_at or _utcnow()
    effective = (
        datetime.combine(filing.report_date, datetime.min.time(), tzinfo=timezone.utc)
        if filing.report_date
        else None
    )
    evidence = [
        Evidence(
            source="sec_edgar",
            url=filing.primary_document_url,
            locator=filing.accession_number,
            excerpt=strip_html(document)[:300],
            retrieved_at=observed,
        )
    ]
    # LLM extraction is higher-confidence than blunt regex matching.
    confidence = 0.9 if events.extraction_method == "llm" else 0.8

    def _facts(event_type: str) -> dict:
        matching = [e for e in events.events if e.event_type == event_type]
        return {
            "officers": [
                {
                    "name": e.officer_name,
                    "role": e.role,
                    "effective_date": e.effective_date,
                }
                for e in matching
            ],
            "roles": [e.role for e in matching if e.role],
            "extraction_method": events.extraction_method,
            "items": filing.items,
        }

    def _roles_str(event_type: str) -> str:
        roles = [e.role for e in events.events if e.event_type == event_type and e.role]
        return (
            ", ".join(dict.fromkeys(roles)) if roles else "unspecified officer/director"
        )

    signals: list[Signal] = []
    if events.has_departure:
        vacuum = events.is_leadership_vacuum
        facts = _facts("departure")
        facts.update(
            {"successor_named": events.successor_named, "leadership_vacuum": vacuum}
        )
        signals.append(
            Signal(
                id=f"{filing.accession_number}:departure",
                company_id=company_id,
                signal_type="8k_exec_departure",
                source="sec_edgar",
                observed_at=observed,
                effective_at=effective,
                title=f"8-K Item 5.02 executive departure ({_roles_str('departure')})",
                summary=(
                    "Item 5.02 discloses an executive departure "
                    + (
                        "with no named successor in the same filing (possible "
                        "open executive search / succession gap)."
                        if vacuum
                        else "alongside a named successor."
                    )
                ),
                extracted_facts=facts,
                evidence=evidence,
                confidence=confidence,
                # A vacuum is the higher-value signal; weight strength accordingly.
                strength=0.75 if vacuum else 0.4,
            )
        )
    if events.has_appointment:
        signals.append(
            Signal(
                id=f"{filing.accession_number}:appointment",
                company_id=company_id,
                signal_type="8k_exec_appointment",
                source="sec_edgar",
                observed_at=observed,
                effective_at=effective,
                title=f"8-K Item 5.02 executive appointment ({_roles_str('appointment')})",
                summary="Item 5.02 discloses an executive appointment/election.",
                extracted_facts=_facts("appointment"),
                evidence=evidence,
                confidence=confidence,
                strength=0.5,
            )
        )
    return signals
