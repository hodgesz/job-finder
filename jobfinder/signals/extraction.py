"""LLM-backed Item 5.02 event extraction, with deterministic fallback.

The deterministic regex parser in `sec_8k.py` is cheap and good enough as a
*pre-filter*, but SEC filing prose is dense and adversarial to keyword rules
(boilerplate captions, compensatory wording, interim/acting appointments).
This module adds a structured LLM extraction step that does the actual
classify-the-event work, and falls back to the regex parser when no API key
is configured or the call fails — so CI and offline runs stay functional and
the LLM is an enhancement, not a hard dependency.

Design:
- `ExtractedEvents` is the typed contract the LLM must return (and what the
  regex fallback is mapped into), so downstream `Signal` construction does not
  care which path produced it.
- The LLM client is injected (`Extractor` protocol) so tests run offline
  against canned responses — no live API calls in CI.
- Provider: Gemini via `google-genai`, matching the ADK+Gemini orchestrator.
"""

from __future__ import annotations

import os
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from jobfinder.signals.sec_8k import parse_item_502

# Default model; overridable per call. Kept in one place for easy bump.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

EventType = Literal["departure", "appointment", "compensatory_only", "other"]


class ExecEvent(BaseModel):
    """One executive event disclosed in an Item 5.02 filing."""

    event_type: EventType
    officer_name: str | None = Field(
        None, description="Name of the officer/director, if stated."
    )
    role: str | None = Field(None, description="Role, e.g. 'Chief Financial Officer'.")
    effective_date: str | None = Field(
        None, description="Effective date as written, if stated."
    )


class ExtractedEvents(BaseModel):
    """Structured result of extracting events from an Item 5.02 document.

    This is the single contract both the LLM and the regex fallback produce.
    """

    events: list[ExecEvent] = Field(default_factory=list)
    successor_named: bool = Field(
        False,
        description="True if an appointment/election names a successor for a departed role.",
    )
    extraction_method: Literal["llm", "regex_fallback"] = "regex_fallback"

    @property
    def has_departure(self) -> bool:
        return any(e.event_type == "departure" for e in self.events)

    @property
    def has_appointment(self) -> bool:
        return any(e.event_type == "appointment" for e in self.events)

    @property
    def is_leadership_vacuum(self) -> bool:
        """A departure with no named successor -> possible succession gap."""
        return self.has_departure and not self.successor_named


@runtime_checkable
class Extractor(Protocol):
    """Anything that can turn an Item 5.02 document into ExtractedEvents."""

    def extract(self, document: str) -> ExtractedEvents: ...


_SYSTEM_INSTRUCTION = (
    "You extract executive-transition events from SEC Form 8-K Item 5.02 "
    "filings. Classify each disclosed event as 'departure' (an officer or "
    "director actually leaving/resigning/retiring/being removed), "
    "'appointment' (someone elected or appointed to a role), "
    "'compensatory_only' (only compensation/severance/retirement-benefit "
    "terms, with nobody actually leaving or joining), or 'other'. Treat the "
    "standard Item 5.02 caption itself as boilerplate, not an event. Set "
    "successor_named true only when a named person fills a role that was "
    "vacated. Return only events actually disclosed in the body."
)


def regex_fallback(document: str, *, item_known: bool | None = None) -> ExtractedEvents:
    """Map the deterministic parser's output into ExtractedEvents.

    Used when the LLM is unavailable. Loses per-officer detail (names/dates)
    but preserves the departure/appointment/vacuum classification.
    """
    parsed = parse_item_502(document, item_known=item_known)
    role = ", ".join(parsed.roles) if parsed.roles else None
    events: list[ExecEvent] = []
    if parsed.has_departure:
        events.append(ExecEvent(event_type="departure", role=role))
    if parsed.has_appointment:
        events.append(ExecEvent(event_type="appointment", role=role))
    return ExtractedEvents(
        events=events,
        successor_named=parsed.successor_present,
        extraction_method="regex_fallback",
    )


class GeminiExtractor:
    """Item 5.02 extractor backed by Gemini structured output.

    The genai client is injected so tests can supply a fake. Use
    `from_env()` to build a real client from GEMINI_API_KEY.
    """

    def __init__(self, client, *, model: str = DEFAULT_GEMINI_MODEL):
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls, *, model: str = DEFAULT_GEMINI_MODEL) -> GeminiExtractor | None:
        """Build from GEMINI_API_KEY, or return None if unset/unavailable."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            from google import genai
        except ImportError:
            return None
        return cls(genai.Client(api_key=api_key), model=model)

    def extract(self, document: str) -> ExtractedEvents:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=document,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ExtractedEvents,
                temperature=0.0,
            ),
        )
        # Prefer the SDK's parsed object; fall back to validating raw text.
        parsed = getattr(response, "parsed", None)
        result = (
            parsed
            if isinstance(parsed, ExtractedEvents)
            else ExtractedEvents.model_validate_json(response.text)
        )
        result.extraction_method = "llm"
        return result


def extract_events(
    document: str,
    *,
    extractor: Extractor | None = None,
    item_known: bool | None = None,
) -> ExtractedEvents:
    """Extract Item 5.02 events, preferring the LLM and falling back to regex.

    Resolution order:
      1. An explicitly provided `extractor` (used by tests and callers).
      2. A Gemini extractor built from GEMINI_API_KEY, if available.
      3. The deterministic regex parser.

    Any failure in the LLM path degrades to the regex fallback rather than
    raising, so signal extraction never hard-depends on the model.
    """
    chosen = extractor or GeminiExtractor.from_env()
    if chosen is not None:
        try:
            return chosen.extract(document)
        except Exception:
            # Enhancement, not a hard dependency: degrade to deterministic parse.
            pass
    return regex_fallback(document, item_known=item_known)
