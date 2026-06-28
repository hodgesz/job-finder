"""ADK + Gemini orchestrator that consumes the 8-K specialist over A2A.

This is the *other* side of the seam: an ADK ``LlmAgent`` (Gemini-backed)
whose sub-agent is a ``RemoteA2aAgent`` pointed at the 8-K service's
well-known agent card. The orchestrator can delegate "extract executive
signals from this filing" to the remote specialist via the A2A protocol,
proving the contract end-to-end.

Unlike the service body, the orchestrator genuinely *is* an LLM agent — it
reasons about which specialist to call and synthesizes results — so it carries
a Gemini model. Constructing the agent does NOT call the model or the network
(ADK resolves both lazily at run time), which keeps this module importable and
unit-testable offline; only an actual ``runner.run_async`` would reach Gemini
and the remote service.

``AGENT_CARD_WELL_KNOWN_PATH`` resolves to ``/.well-known/agent-card.json``;
``remote_8k_agent`` appends it to the service base URL so the orchestrator
fetches the card the ``to_a2a()`` app publishes.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    AGENT_CARD_WELL_KNOWN_PATH,
    RemoteA2aAgent,
)

from jobfinder.a2a.service import AGENT_DESCRIPTION, AGENT_NAME

# Matches GeminiExtractor's default model so the whole stack speaks one model
# family; overridable per construction.
DEFAULT_ORCHESTRATOR_MODEL = "gemini-flash-latest"

DEFAULT_SERVICE_BASE_URL = "http://localhost:8001"

_ORCHESTRATOR_INSTRUCTION = (
    "You are an executive-opportunity intelligence orchestrator. When given a "
    "SEC Form 8-K filing to analyze, delegate extraction to the "
    f"'{AGENT_NAME}' specialist, which returns evidence-backed Signal objects. "
    "Do not invent signals: rely on the specialist's structured output and "
    "summarize what it found, including whether a leadership vacuum was "
    "detected."
)


def agent_card_url(base_url: str = DEFAULT_SERVICE_BASE_URL) -> str:
    """The well-known agent-card URL for the 8-K service at ``base_url``."""
    return base_url.rstrip("/") + AGENT_CARD_WELL_KNOWN_PATH


def remote_8k_agent(base_url: str = DEFAULT_SERVICE_BASE_URL) -> RemoteA2aAgent:
    """A RemoteA2aAgent handle to the deployed 8-K specialist.

    Points at the service's published well-known agent card. Construction is
    cheap and offline; the card is fetched lazily on first use.
    """
    return RemoteA2aAgent(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        agent_card=agent_card_url(base_url),
    )


def build_orchestrator(
    *,
    base_url: str = DEFAULT_SERVICE_BASE_URL,
    model: str = DEFAULT_ORCHESTRATOR_MODEL,
) -> LlmAgent:
    """Build the Gemini orchestrator that consumes the 8-K A2A specialist.

    Importable/constructible offline; running it (``runner.run_async``) is what
    reaches Gemini and the remote service.
    """
    return LlmAgent(
        name="opportunity_orchestrator",
        model=model,
        instruction=_ORCHESTRATOR_INSTRUCTION,
        description=(
            "Routes SEC-filing analysis to remote signal specialists and "
            "synthesizes executive-opportunity intelligence."
        ),
        sub_agents=[remote_8k_agent(base_url)],
    )
