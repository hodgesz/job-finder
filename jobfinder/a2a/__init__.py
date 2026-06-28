"""A2A extraction of the 8-K signal specialist.

Slice 4 crosses the modular-monolith -> A2A seam for the first time. The
in-process 8-K specialist (``jobfinder.signals.sec_8k`` / ``extraction``) is
re-exposed here as a standalone service:

    LangGraph StateGraph   (the specialist's real work; ``graph.py``)
        wrapped in a thin custom ADK BaseAgent   (``service.py``)
        exposed via google-adk's ``to_a2a()`` Starlette app
            consumed by an ADK RemoteA2aAgent orchestrator   (``orchestrator.py``)

The wire contract (``contract.py``) is built on ``jobfinder.schemas`` — the
same framework-free Signal/Evidence models the in-process pipeline uses — so
the A2A boundary speaks the domain language, not an ADK/LangGraph-specific one.

Nothing here replaces the in-process path: ``pipeline.run_pipeline`` and the
CLI keep working unchanged. A2A is an *additional* consumption path.
"""

from __future__ import annotations

from jobfinder.a2a.contract import EightKExtractionRequest, EightKExtractionResponse

__all__ = ["EightKExtractionRequest", "EightKExtractionResponse"]
