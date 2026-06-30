"""Layer-2 LLM relevance re-rank over the deterministic Layer-1 top-N.

Layer 1 (``match.rank_jobs``) is the stable, explainable, offline base: a
human can read exactly why every job scored what it did. Layer 2 is an
*optional enhancement* that asks Gemini to re-order the Layer-1 top-N against
the VP-of-AI profile and attach a short rationale, then merges that judgement
back into the ranking. It is never a hard dependency — exactly the
``extract_events`` → ``GeminiExtractor.from_env()`` contract in
``jobfinder.signals.extraction``:

- The LLM client is **injected** (``Reranker`` protocol), so CI is fully
  hermetic — tests drive a fake re-ranker returning canned orderings; no live
  network, no secrets, no real Gemini call.
- **Env-keyed, degrades to Layer-1:** ``GeminiReranker.from_env()`` returns a
  working re-ranker only when ``GEMINI_API_KEY`` is set, else ``None``. Any
  failure in the LLM path (or a ``None`` re-ranker) leaves the Layer-1 order
  untouched, so a run without a key behaves exactly like Layer 1.

Two invariants protect the result:

1. **No silent override.** The Layer-1 score/tier are authoritative and never
   mutated; the LLM's contribution is surfaced as an additive ``LlmRerank``
   annotation (rank + relevance + rationale) on each re-ranked match, so the
   ranked output shows *why* a job moved.
2. **No hallucinated jobs.** The LLM can only re-order the candidate set it was
   given; any candidate id it invents (or repeats) is ignored, and any
   candidate it omits keeps its Layer-1 relative order, appended after the
   re-ranked ones. The LLM cannot introduce a job that was not in the top-N.

The response model below is a small detour-local pydantic model used *only* for
the LLM I/O boundary (google-genai structured output wants a pydantic schema);
the domain models (``JobMatch`` etc.) stay pure dataclasses, decoupled from the
core ``schemas.py``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from jobfinder.jobsearch.models import JobMatch, LlmRerank
from jobfinder.jobsearch.profile import TargetProfile

# Default model; matches the core GeminiExtractor so a key bump is one place.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# How many Layer-1 candidates go to the LLM by default (cost control). Only the
# non-rejected top of the ranking is ever sent; the rest stay in Layer-1 order.
DEFAULT_RERANK_TOP = 20

Relevance = Literal["strong", "moderate", "weak"]


class RerankedItem(BaseModel):
    """The LLM's verdict for one candidate, keyed by the id we assigned it."""

    candidate_id: int = Field(
        description="The integer id of the candidate being ranked (as provided)."
    )
    relevance: Relevance = Field(
        description="How relevant this role is to the VP-of-AI target profile."
    )
    rationale: str = Field(
        description="One short sentence on why this role ranks where it does."
    )


class RerankResponse(BaseModel):
    """Structured LLM re-rank: the candidate ids in best-first order.

    This is the single contract the LLM returns and the fake re-ranker mimics in
    tests. It is an LLM-I/O model only — never stored on the domain dataclasses.
    """

    ranking: list[RerankedItem] = Field(
        default_factory=list,
        description="Candidates in best-first order; only ids that were provided.",
    )


@runtime_checkable
class Reranker(Protocol):
    """Anything that can re-order a set of candidate matches.

    ``candidates`` is the Layer-1 top-N (best-first). The id of each candidate is
    its index in that list. Implementations return a ``RerankResponse`` ordering
    those ids; unknown/duplicate ids are ignored by the caller.
    """

    def rerank(
        self, candidates: list[JobMatch], profile: TargetProfile
    ) -> RerankResponse: ...


_SYSTEM_INSTRUCTION = (
    "You are screening job postings for a candidate targeting senior executive "
    "AI/data/analytics leadership roles (VP of AI, VP of AI & Data, VP of AI & "
    "Analytics, Head of AI, Chief AI/Data Officer, and close equivalents). You "
    "are given a numbered list of candidate roles that a deterministic pre-filter "
    "already judged plausible. Re-order them best-first for this candidate, and "
    "give each a relevance verdict (strong/moderate/weak) and one-sentence "
    "rationale. Only use the integer ids provided — never invent a role. Prefer "
    "genuine AI/data leadership scope and seniority over keyword coincidence."
)


def _candidate_prompt(candidates: list[JobMatch]) -> str:
    """Render the candidate set as a compact, id-keyed prompt body."""
    lines = ["Candidate roles (id: title — company [location] | Layer-1 score):"]
    for cid, m in enumerate(candidates):
        job = m.job
        loc = job.location or "location n/a"
        lines.append(
            f"{cid}: {job.title} — {job.company} [{loc}] "
            f"| score {m.score:.0f}/100 tier {m.tier.value}"
        )
    lines.append(
        "\nReturn every id exactly once, ordered best-first, with a relevance "
        "verdict and a one-sentence rationale each."
    )
    return "\n".join(lines)


class GeminiReranker:
    """Layer-2 re-ranker backed by Gemini structured output.

    The genai client is injected so tests supply a fake; use ``from_env()`` to
    build a real client from ``GEMINI_API_KEY``.
    """

    def __init__(self, client, *, model: str = DEFAULT_GEMINI_MODEL) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls, *, model: str = DEFAULT_GEMINI_MODEL) -> GeminiReranker | None:
        """Build from ``GEMINI_API_KEY``, or return ``None`` if unset/unavailable.

        Mirrors ``GeminiExtractor.from_env`` — a missing key or an uninstalled
        ``google-genai`` both yield ``None``, so the caller degrades to Layer-1.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            from google import genai
        except ImportError:
            return None
        return cls(genai.Client(api_key=api_key), model=model)

    def rerank(
        self, candidates: list[JobMatch], profile: TargetProfile
    ) -> RerankResponse:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=_candidate_prompt(candidates),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RerankResponse,
                temperature=0.0,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, RerankResponse):
            return parsed
        return RerankResponse.model_validate_json(response.text)


def _merge(candidates: list[JobMatch], response: RerankResponse) -> list[JobMatch]:
    """Apply the LLM ordering to the candidate set, defensively.

    Accepts only ids in ``range(len(candidates))``, each at most once (the first
    mention wins); ignores invented or duplicate ids. Candidates the LLM omits
    keep their Layer-1 relative order and are appended after the re-ranked ones,
    un-annotated — so the LLM can re-order but never drop or invent a job.
    """
    n = len(candidates)
    seen: set[int] = set()
    reranked: list[JobMatch] = []
    for item in response.ranking:
        cid = item.candidate_id
        if cid < 0 or cid >= n or cid in seen:
            continue  # invented or duplicate id — ignore
        seen.add(cid)
        # rank is the 1-based position among the ACCEPTED items, so skipping an
        # invalid/duplicate id never leaves a gap (the first kept job is always
        # #1) — matches what the CLI renders and the LlmRerank docstring promises.
        annotation = LlmRerank(
            rank=len(reranked) + 1,
            relevance=item.relevance,
            rationale=item.rationale.strip(),
        )
        reranked.append(replace(candidates[cid], llm=annotation))
    # Any candidate the LLM didn't rank keeps its Layer-1 order, appended after.
    leftovers = [candidates[cid] for cid in range(n) if cid not in seen]
    return reranked + leftovers


def rerank_matches(
    matches: list[JobMatch],
    profile: TargetProfile,
    *,
    reranker: Reranker | None = None,
    top_n: int = DEFAULT_RERANK_TOP,
) -> list[JobMatch]:
    """Re-rank the Layer-1 top-N with the LLM, preferring it, falling back cleanly.

    ``matches`` is the Layer-1 output (``rank_jobs``: best-first, rejected last).
    The first ``top_n`` *non-rejected* matches are the candidate set sent to the
    LLM; everything else (the non-rejected tail beyond ``top_n`` and all rejected
    matches) keeps its Layer-1 order and is appended unchanged.

    Resolution mirrors ``extract_events``:
      1. An explicitly provided ``reranker`` (tests and callers).
      2. A Gemini re-ranker from ``GEMINI_API_KEY``, if available.
      3. No re-ranker → return ``matches`` unchanged (pure Layer-1).

    Any failure in the LLM path degrades to the original Layer-1 order rather
    than raising, so Layer 2 is an enhancement and never breaks ranking.
    """
    chosen = reranker or GeminiReranker.from_env()
    if chosen is None:
        return matches

    # Partition in a single pass that does NOT assume ``matches`` is pre-sorted
    # non-rejected-first: candidates are the first ``top_n`` non-rejected matches;
    # ``rest`` is everything else (rejected, or non-rejected beyond ``top_n``) in
    # original order. This keeps "no job dropped or duplicated" true for any input
    # ordering, not just ``rank_jobs``'s.
    candidates: list[JobMatch] = []
    rest: list[JobMatch] = []
    for m in matches:
        if not m.rejected and len(candidates) < top_n:
            candidates.append(m)
        else:
            rest.append(m)
    if not candidates:
        return matches

    try:
        response = chosen.rerank(candidates, profile)
    except Exception:
        # Enhancement, not a hard dependency: keep the Layer-1 order.
        return matches

    return _merge(candidates, response) + rest
