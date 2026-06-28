"""Tests for the Slice 4 A2A extraction of the 8-K signal specialist.

Everything here is hermetic: no live Gemini, no network. The LangGraph graph
runs the deterministic regex fallback (or an injected fake extractor), the ADK
agent runs over an in-memory Runner, and the to_a2a() Starlette app is driven
with Starlette's in-process TestClient. The orchestrator and RemoteA2aAgent are
only *constructed* (which is offline); actually running them would reach Gemini
and the remote service, so we don't.

These prove the A2A contract end-to-end without migrating any other module and
without breaking the in-process path (covered by test_sec_8k_signal etc.).
"""

import json
from datetime import date, datetime, timezone

import pytest
from starlette.testclient import TestClient

from jobfinder.a2a.contract import (
    EightKExtractionRequest,
    EightKExtractionResponse,
    FilingRef,
)
from jobfinder.a2a.graph import extract_signals
from jobfinder.a2a.orchestrator import (
    agent_card_url,
    build_orchestrator,
    remote_8k_agent,
)
from jobfinder.a2a.service import (
    AGENT_NAME,
    build_a2a_app,
    build_agent,
    build_agent_card,
)
from jobfinder.schemas import Signal
from jobfinder.signals.extraction import ExecEvent, ExtractedEvents
from jobfinder.sources.edgar import Filing

# ADK's A2A support emits an [EXPERIMENTAL] UserWarning on construction; it is
# expected and not a defect, so silence it across this module.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

OBSERVED = datetime(2026, 5, 1, tzinfo=timezone.utc)

CAPTION = (
    "Item 5.02 Departure of Directors or Certain Officers; Election of Directors; "
    "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers. "
)
# Departure with no successor named -> leadership vacuum.
VACUUM_DOC = (
    CAPTION + "On May 1, 2026, Jane Doe, the Company's Chief Financial Officer, "
    "notified the Board of her resignation, effective immediately."
)
# A filing that does not disclose Item 5.02 at all.
NON_502_DOC = "Item 2.02 Results of Operations and Financial Condition. Revenue rose."


class FakeExtractor:
    """Returns a canned ExtractedEvents — stands in for a live Gemini call."""

    def __init__(self, result: ExtractedEvents):
        self._result = result
        self.calls = 0

    def extract(self, document: str) -> ExtractedEvents:
        self.calls += 1
        return self._result


def _filing(items: list[str]) -> Filing:
    return Filing(
        cik="320193",
        accession_number="0001140361-26-015711",
        form="8-K",
        filing_date=date(2026, 4, 20),
        report_date=date(2026, 5, 1),
        items=items,
        primary_document="ef20071035_8k.htm",
    )


def _request(items: list[str], document: str) -> EightKExtractionRequest:
    return EightKExtractionRequest(
        company_id="apple",
        filing=FilingRef.from_filing(_filing(items)),
        document=document,
        observed_at=OBSERVED,
    )


# --- FilingRef wire contract ------------------------------------------------


def test_filing_ref_roundtrips_through_json():
    original = _filing(["5.02", "9.01"])
    ref = FilingRef.from_filing(original)
    rebuilt = FilingRef.model_validate_json(ref.model_dump_json()).to_filing()
    assert rebuilt.cik == original.cik
    assert rebuilt.accession_number == original.accession_number
    assert rebuilt.items == original.items
    assert rebuilt.report_date == original.report_date
    # primary_document_url is derived; it should survive the roundtrip.
    assert rebuilt.primary_document_url == original.primary_document_url


def test_filing_ref_defaults_filing_date_when_missing():
    # filing_date is required on the Filing dataclass; the contract allows it to
    # be omitted and falls back to report_date.
    ref = FilingRef(cik="1", accession_number="a", report_date=date(2026, 1, 2))
    assert ref.to_filing().filing_date == date(2026, 1, 2)


# --- LangGraph graph --------------------------------------------------------


def test_graph_extracts_leadership_vacuum_signal():
    resp = extract_signals(_request(["5.02"], VACUUM_DOC))
    assert isinstance(resp, EightKExtractionResponse)
    types_ = {s.signal_type for s in resp.signals}
    assert "8k_exec_departure" in types_
    departure = next(s for s in resp.signals if s.signal_type == "8k_exec_departure")
    assert departure.extracted_facts["leadership_vacuum"] is True
    # Each Signal records how it was classified — the single source of truth.
    assert departure.extracted_facts["extraction_method"] == "regex_fallback"
    # The wire contract carries real, evidence-backed domain Signals.
    assert all(isinstance(s, Signal) for s in resp.signals)
    assert departure.evidence


def test_graph_short_circuits_non_502_filing():
    resp = extract_signals(_request(["2.02"], NON_502_DOC))
    assert resp.signals == []


def test_graph_is_hermetic_by_default_even_with_api_key(monkeypatch):
    # The standing service must NOT make a live Gemini call per request just
    # because GEMINI_API_KEY happens to be in its environment. With no injected
    # extractor, extract_signals pins the deterministic RegexExtractor.
    monkeypatch.setenv("GEMINI_API_KEY", "sk-should-never-be-used")

    # Trip-wire: if the env-resolved Gemini path is ever reached, fail loudly
    # instead of making a network call.
    def _boom(*args, **kwargs):
        raise AssertionError("GeminiExtractor.from_env() must not run in the service")

    monkeypatch.setattr("jobfinder.signals.extraction.GeminiExtractor.from_env", _boom)

    resp = extract_signals(_request(["5.02"], VACUUM_DOC))
    departure = next(s for s in resp.signals if s.signal_type == "8k_exec_departure")
    assert departure.extracted_facts["extraction_method"] == "regex_fallback"


def test_graph_uses_injected_extractor_and_reports_llm_method():
    canned = ExtractedEvents(
        events=[ExecEvent(event_type="departure", officer_name="Jane Doe", role="CFO")],
        successor_named=False,
        extraction_method="llm",
    )
    fake = FakeExtractor(canned)
    resp = extract_signals(_request(["5.02"], VACUUM_DOC), extractor=fake)
    assert fake.calls == 1
    departure = next(s for s in resp.signals if s.signal_type == "8k_exec_departure")
    # The Signal carries the LLM classification method + its higher confidence.
    assert departure.extracted_facts["extraction_method"] == "llm"
    assert departure.confidence == 0.9
    # LLM path carries per-officer detail the regex path cannot.
    assert departure.extracted_facts["officers"][0]["name"] == "Jane Doe"


# --- ADK service agent (sync surface) ---------------------------------------


def test_service_agent_run_extraction_returns_response():
    agent = build_agent()
    resp = agent.run_extraction(_request(["5.02"], VACUUM_DOC).model_dump_json())
    assert isinstance(resp, EightKExtractionResponse)
    assert resp.company_id == "apple"
    assert any(s.signal_type == "8k_exec_departure" for s in resp.signals)


# --- ADK service agent over an in-memory Runner (the real agent loop) -------


@pytest.mark.asyncio
async def test_service_agent_over_runner_yields_signal_json():
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = build_agent()
    runner = InMemoryRunner(agent=agent, app_name="test_a2a")
    session = await runner.session_service.create_session(
        app_name="test_a2a", user_id="u"
    )
    request = _request(["5.02"], VACUUM_DOC)
    message = types.Content(
        role="user", parts=[types.Part(text=request.model_dump_json())]
    )

    chunks: list[str] = []
    async for event in runner.run_async(
        user_id="u", session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            chunks.append("".join(p.text or "" for p in event.content.parts))

    payload = json.loads(chunks[-1])
    assert payload["company_id"] == "apple"
    assert any(s["signal_type"] == "8k_exec_departure" for s in payload["signals"])


@pytest.mark.asyncio
async def test_service_agent_reports_error_on_malformed_request():
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = build_agent()
    runner = InMemoryRunner(agent=agent, app_name="test_a2a_err")
    session = await runner.session_service.create_session(
        app_name="test_a2a_err", user_id="u"
    )
    message = types.Content(role="user", parts=[types.Part(text="{not json}")])

    chunks: list[str] = []
    async for event in runner.run_async(
        user_id="u", session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            chunks.append("".join(p.text or "" for p in event.content.parts))

    payload = json.loads(chunks[-1])
    assert payload["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_service_agent_reports_error_on_downstream_failure():
    # A request that parses cleanly can still blow up downstream: a non-numeric
    # cik survives schema validation but breaks Filing.primary_document_url
    # (which calls int(cik)) while building Evidence. The service boundary must
    # answer with structured JSON, not crash the A2A stream.
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = build_agent()
    bad = EightKExtractionRequest(
        company_id="apple",
        filing=FilingRef(
            cik="not-a-number",
            accession_number="0001140361-26-015711",
            items=["5.02"],
            primary_document="x.htm",
        ),
        document=VACUUM_DOC,
    )
    # The unguarded sync surface raises...
    with pytest.raises(ValueError):
        agent.run_extraction(bad.model_dump_json())

    # ...but the agent loop wraps it into a structured error event.
    runner = InMemoryRunner(agent=agent, app_name="test_a2a_down")
    session = await runner.session_service.create_session(
        app_name="test_a2a_down", user_id="u"
    )
    message = types.Content(role="user", parts=[types.Part(text=bad.model_dump_json())])
    chunks: list[str] = []
    async for event in runner.run_async(
        user_id="u", session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            chunks.append("".join(p.text or "" for p in event.content.parts))

    payload = json.loads(chunks[-1])
    assert payload["error"] == "extraction_failed"


# --- Agent card + to_a2a() Starlette app ------------------------------------


@pytest.mark.asyncio
async def test_agent_card_advertises_the_specialist():
    card = await build_agent_card().build()
    assert card.name == AGENT_NAME
    assert card.description
    assert card.url.rstrip("/") == "http://localhost:8001"


def test_to_a2a_app_serves_well_known_card():
    app = build_a2a_app(port=8001)
    with TestClient(app) as client:
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        card = response.json()
        assert card["name"] == AGENT_NAME
        assert card["url"].rstrip("/") == "http://localhost:8001"


# --- RemoteA2aAgent orchestrator (constructed offline) ----------------------


def test_agent_card_url_appends_well_known_path():
    assert agent_card_url("http://svc:9000") == (
        "http://svc:9000/.well-known/agent-card.json"
    )
    # Trailing slash on the base must not double up.
    assert agent_card_url("http://svc:9000/") == (
        "http://svc:9000/.well-known/agent-card.json"
    )


def test_remote_8k_agent_points_at_service_card():
    remote = remote_8k_agent("http://svc:9000")
    assert remote.name == AGENT_NAME


def test_orchestrator_wires_remote_specialist_as_subagent():
    orchestrator = build_orchestrator()
    assert orchestrator.model == "gemini-flash-latest"
    assert [sub.name for sub in orchestrator.sub_agents] == [AGENT_NAME]
