"""The 8-K signal specialist as a LangGraph StateGraph.

This is the unit of work the A2A service exposes. It is intentionally a real
LangGraph graph (not just a function call) so the service is genuinely a
"LangGraph specialist behind A2A", matching the target architecture — but the
nodes delegate to the existing, well-tested in-process extractor
(``signals_from_filing`` -> ``extract_events``) rather than re-implementing it.
That keeps a single source of truth for 8-K classification across both the
in-process pipeline and the A2A service.

The graph has two nodes:

    pre_filter  -> only filings whose index discloses Item 5.02 (or whose body
                   mentions it) proceed; everything else short-circuits to an
                   empty result. This mirrors the cheap pre-filter the
                   in-process path applies before the (possibly LLM-backed)
                   extractor.
    extract     -> runs the injected extractor and builds Signal objects.

Hermeticity: the extractor is injected via graph state (``extractor``), so the
graph never reaches for a live Gemini client on its own. In tests and CI no
extractor (or a ``RegexExtractor``) is passed and the deterministic regex path
runs; a real ``GeminiExtractor`` can be injected in production.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from jobfinder.a2a.contract import EightKExtractionRequest, EightKExtractionResponse
from jobfinder.schemas import Signal
from jobfinder.signals.extraction import Extractor
from jobfinder.signals.sec_8k import discloses_item_502, signals_from_filing


class ExtractionState(TypedDict, total=False):
    """State threaded through the 8-K extraction graph."""

    request: EightKExtractionRequest
    extractor: Extractor | None
    discloses_502: bool
    signals: list[Signal]


def _pre_filter(state: ExtractionState) -> dict[str, Any]:
    """Decide whether the filing discloses Item 5.02 at all.

    The index ``items`` field is authoritative; fall back to scanning the body
    for the literal item label. Filings that disclose nothing 5.02-related are
    not worth handing to the (possibly LLM-backed) extractor.
    """
    req = state["request"]
    discloses = discloses_item_502(req.filing.items, req.document)
    return {"discloses_502": discloses}


def _route_after_filter(state: ExtractionState) -> str:
    return "extract" if state.get("discloses_502") else "empty"


def _extract(state: ExtractionState) -> dict[str, Any]:
    """Run the in-process 8-K extractor and capture the resulting signals.

    Each emitted Signal already records how it was classified (its
    ``extracted_facts['extraction_method']`` and its confidence), so the graph
    does not re-derive a run-level method — that was a redundant second source
    of truth that also misreported when an LLM classified a filing as
    compensatory-only (no departure/appointment -> no signals to inspect).
    """
    req = state["request"]
    observed = req.observed_at or datetime.now(timezone.utc)
    signals = signals_from_filing(
        req.filing.to_filing(),
        req.document,
        company_id=req.company_id,
        observed_at=observed,
        extractor=state.get("extractor"),
    )
    return {"signals": signals}


def _empty(state: ExtractionState) -> dict[str, Any]:
    """Short-circuit result for filings that do not disclose Item 5.02."""
    return {"signals": []}


def build_graph():
    """Compile the 8-K extraction StateGraph."""
    builder = StateGraph(ExtractionState)
    builder.add_node("pre_filter", _pre_filter)
    builder.add_node("extract", _extract)
    builder.add_node("empty", _empty)

    builder.add_edge(START, "pre_filter")
    builder.add_conditional_edges(
        "pre_filter",
        _route_after_filter,
        {"extract": "extract", "empty": "empty"},
    )
    builder.add_edge("extract", END)
    builder.add_edge("empty", END)
    return builder.compile()


# Module-level singleton — the compiled graph is stateless and reusable.
GRAPH = build_graph()


def extract_signals(
    request: EightKExtractionRequest,
    *,
    extractor: Extractor | None = None,
) -> EightKExtractionResponse:
    """Run a request through the compiled graph and shape the wire response.

    This is the synchronous entry point the ADK service node calls. The
    extractor is injected (default None -> hermetic regex fallback unless a
    GeminiExtractor is resolved from the environment by the in-process path).
    """
    final = GRAPH.invoke({"request": request, "extractor": extractor})
    return EightKExtractionResponse(
        company_id=request.company_id,
        signals=final.get("signals", []),
    )
