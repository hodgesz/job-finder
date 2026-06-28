"""The 8-K extraction service: a custom ADK BaseAgent over the LangGraph graph.

``to_a2a()`` accepts an ADK ``BaseAgent``. We deliberately do NOT use
``LlmAgent`` for the service body: the work is deterministic structured
extraction, not a model conversation, and we want the *agent loop* to be
hermetic (no live Gemini just to run the service). So this is a thin custom
BaseAgent whose ``_run_async_impl`` reads the request JSON off the invocation's
``user_content``, runs the LangGraph graph, and yields one Event carrying the
response JSON.

Note this is distinct from the LLM that may run *inside* extraction: the
specialist can use ``GeminiExtractor`` for classification (injected, optional),
but the agent envelope itself needs no model. Tests construct the agent with
the default (regex) extractor so the whole path is offline.

``build_a2a_app()`` wraps the agent in a ``to_a2a()`` Starlette app, which
serves the agent card at ``/.well-known/agent-card.json`` and the A2A RPC
endpoint. ``build_agent_card()`` exposes the card for tests without binding a
port.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import ValidationError

from jobfinder.a2a.contract import EightKExtractionRequest, EightKExtractionResponse
from jobfinder.a2a.graph import extract_signals
from jobfinder.signals.extraction import Extractor

AGENT_NAME = "eightk_signal_specialist"
AGENT_DESCRIPTION = (
    "Extracts executive-transition signals (8-K Item 5.02 departures and "
    "appointments, including leadership-vacuum detection) from a single SEC "
    "Form 8-K filing. Input: a JSON EightKExtractionRequest (company_id, "
    "filing reference, document text). Output: a JSON EightKExtractionResponse "
    "carrying evidence-backed Signal objects."
)


def _extract_request_text(content: types.Content | None) -> str:
    """Concatenate the text parts of an ADK Content into the raw request body."""
    if content is None or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


class EightKSignalAgent(BaseAgent):
    """Custom ADK agent that runs the LangGraph 8-K specialist.

    The extractor is injected and defaults to None; the underlying
    ``extract_signals`` then pins a deterministic ``RegexExtractor``, so the
    service stays hermetic by default (no live Gemini, even with
    ``GEMINI_API_KEY`` set). Inject a ``GeminiExtractor`` to opt into the LLM.
    """

    # BaseAgent is a pydantic model; declare the extra field so assignment
    # in __init__ is allowed.
    extractor: Extractor | None = None

    def __init__(self, *, extractor: Extractor | None = None, **kwargs):
        super().__init__(
            name=kwargs.pop("name", AGENT_NAME),
            description=kwargs.pop("description", AGENT_DESCRIPTION),
            **kwargs,
        )
        self.extractor = extractor

    def run_extraction(self, raw_request: str) -> EightKExtractionResponse:
        """Parse a raw JSON request, run the graph, return the response.

        Kept as a sync method so it is trivially unit-testable without the ADK
        Runner machinery. Raises ``ValidationError`` on a malformed request.
        """
        request = EightKExtractionRequest.model_validate_json(raw_request)
        return extract_signals(request, extractor=self.extractor)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        raw_request = _extract_request_text(ctx.user_content)
        try:
            response = self.run_extraction(raw_request)
            payload = response.model_dump_json()
        except ValidationError as exc:
            payload = json.dumps(
                {"error": "invalid_request", "detail": json.loads(exc.json())}
            )
        except Exception as exc:
            # A request can parse cleanly yet still fail downstream (e.g. a
            # non-numeric cik blows up Filing.primary_document_url). A service
            # boundary must answer with structured JSON, not crash the A2A
            # stream and hand the client an opaque 500.
            payload = json.dumps({"error": "extraction_failed", "detail": str(exc)})
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text=payload)]),
            turn_complete=True,
        )


def build_agent(*, extractor: Extractor | None = None) -> EightKSignalAgent:
    """Construct the service agent (regex-hermetic by default)."""
    return EightKSignalAgent(extractor=extractor)


def build_a2a_app(
    *,
    extractor: Extractor | None = None,
    host: str = "localhost",
    port: int = 8001,
):
    """Wrap the service agent in an A2A Starlette app.

    Serves the agent card at ``/.well-known/agent-card.json`` and the A2A RPC
    endpoint. This is a factory (it returns the app); run it programmatically,
    e.g. ``uvicorn.run(build_a2a_app(), host="localhost", port=8001)``, or bind
    its result to a module-level name and point uvicorn at that name.
    """
    # Imported here so plain ``import jobfinder.a2a.service`` does not require
    # the [a2a] extra unless someone actually stands up the server.
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    return to_a2a(build_agent(extractor=extractor), host=host, port=port)


def build_agent_card(*, host: str = "localhost", port: int = 8001):
    """Build the agent card the service would publish, without binding a port.

    Used by tests to assert the contract (name/description/url) and by tooling
    that wants the card without standing up the server.
    """
    from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder

    builder = AgentCardBuilder(
        agent=build_agent(),
        rpc_url=f"http://{host}:{port}/",
    )
    return builder
