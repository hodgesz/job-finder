"""Tests for the Layer-2 Gemini relevance re-rank.

Fully hermetic: a fake re-ranker returns canned orderings — no live network, no
secrets, no real Gemini call. Covers the merge invariants (re-order, no
hallucinated jobs, no dropped jobs), degrade-to-Layer-1, and the annotation that
surfaces the LLM's contribution.
"""

from datetime import datetime, timezone

from jobfinder.jobsearch.match import rank_jobs, score_job
from jobfinder.jobsearch.models import CanonicalJob, JobMatch, LlmRerank
from jobfinder.jobsearch.normalize import normalize_title
from jobfinder.jobsearch.profile import VP_AI_PROFILE
from jobfinder.jobsearch.rerank import (
    GeminiReranker,
    RerankedItem,
    RerankResponse,
    rerank_matches,
)

NOW = datetime(2026, 6, 29, tzinfo=timezone.utc)


def _job(title, *, company="Acme", location="Remote"):
    return CanonicalJob(
        company=company,
        title=title,
        normalized_title=normalize_title(title),
        location=location,
        workplace_type="remote" if location and "remote" in location.lower() else None,
        best_apply_url="https://x/1",
        posted_at=NOW,
    )


def _matches(*titles):
    """Layer-1 matches for a set of titles, ranked best-first."""
    jobs = [_job(t) for t in titles]
    return rank_jobs(jobs, VP_AI_PROFILE, now=NOW)


class _FakeReranker:
    """A re-ranker returning a fixed RerankResponse, recording the candidates it saw."""

    def __init__(self, response: RerankResponse):
        self._response = response
        self.seen: list[JobMatch] | None = None

    def rerank(self, candidates, profile):
        self.seen = candidates
        return self._response


class _BoomReranker:
    def rerank(self, candidates, profile):
        raise RuntimeError("LLM exploded")


class _ReverseReranker:
    """Re-ranks by reversing whatever candidate order it is given (robust to the
    Layer-1 sort order, which `rank_jobs` controls)."""

    def rerank(self, candidates, profile):
        n = len(candidates)
        return RerankResponse(
            ranking=[
                RerankedItem(
                    candidate_id=cid, relevance="strong", rationale=f"pick {cid}"
                )
                for cid in reversed(range(n))
            ]
        )


def test_none_reranker_returns_layer1_unchanged(monkeypatch):
    # No explicit re-ranker AND no key → pure Layer-1, no LLM resolution at all.
    # (delenv guards against a real GEMINI_API_KEY triggering a live call here.)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    matches = _matches("VP of AI", "Head of AI")
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=None)
    assert out == matches
    assert all(m.llm is None for m in out)


def test_reranker_reorders_and_annotates():
    matches = _matches("VP of AI", "VP of Data Science")
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=_ReverseReranker())

    # The LLM reversed the Layer-1 candidate order; titles flip accordingly.
    assert [m.job.title for m in out] == [matches[1].job.title, matches[0].job.title]
    assert out[0].llm == LlmRerank(rank=1, relevance="strong", rationale="pick 1")
    assert out[1].llm == LlmRerank(rank=2, relevance="strong", rationale="pick 0")
    # Layer-1 score/tier are untouched — only the annotation is added.
    assert out[0].score == matches[1].score
    assert out[0].tier == matches[1].tier


def test_llm_cannot_inject_unknown_jobs():
    matches = _matches("VP of AI", "Head of AI")
    response = RerankResponse(
        ranking=[
            RerankedItem(candidate_id=99, relevance="strong", rationale="ghost"),
            RerankedItem(candidate_id=0, relevance="strong", rationale="real"),
        ]
    )
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=_FakeReranker(response))
    titles = [m.job.title for m in out]
    # The invented id is ignored; only the two real candidates remain (no ghost).
    assert len(out) == 2
    assert "ghost" not in " ".join(m.llm.rationale for m in out if m.llm)
    # id 0 was ranked; id 1 (omitted) keeps Layer-1 order, appended un-annotated.
    assert titles[0] == matches[0].job.title
    assert out[-1].llm is None


def test_duplicate_id_first_mention_wins():
    matches = _matches("VP of AI", "Head of AI")
    response = RerankResponse(
        ranking=[
            RerankedItem(candidate_id=0, relevance="strong", rationale="first"),
            RerankedItem(candidate_id=0, relevance="weak", rationale="dup"),
            RerankedItem(candidate_id=1, relevance="moderate", rationale="second"),
        ]
    )
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=_FakeReranker(response))
    assert len(out) == 2
    assert out[0].llm.rationale == "first"  # dup ignored
    assert out[1].llm.rationale == "second"


def test_omitted_candidate_keeps_layer1_order_appended():
    matches = _matches("VP of AI", "VP of Data Science", "Head of AI")
    # LLM ranks only the middle candidate; the other two must survive in order.
    response = RerankResponse(
        ranking=[RerankedItem(candidate_id=1, relevance="strong", rationale="mid")]
    )
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=_FakeReranker(response))
    assert len(out) == 3
    assert out[0].job.title == matches[1].job.title
    assert out[0].llm is not None
    # Leftovers (0 and 2) keep their relative Layer-1 order, un-annotated.
    assert [m.job.title for m in out[1:]] == [
        matches[0].job.title,
        matches[2].job.title,
    ]
    assert all(m.llm is None for m in out[1:])


def test_llm_failure_degrades_to_layer1():
    matches = _matches("VP of AI", "Head of AI")
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=_BoomReranker())
    assert out == matches
    assert all(m.llm is None for m in out)


def test_rejected_matches_are_not_sent_or_reordered():
    jobs = [
        _job("Senior ML Engineer"),  # rejected
        _job("VP of AI"),
        _job("Head of AI"),
    ]
    matches = rank_jobs(jobs, VP_AI_PROFILE, now=NOW)
    fake = _FakeReranker(
        RerankResponse(
            ranking=[
                RerankedItem(candidate_id=0, relevance="strong", rationale="a"),
                RerankedItem(candidate_id=1, relevance="moderate", rationale="b"),
            ]
        )
    )
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=fake)
    # The fake only ever saw the non-rejected candidates.
    assert fake.seen is not None
    assert all(not m.rejected for m in fake.seen)
    # The rejected job stays last and un-annotated.
    assert out[-1].rejected
    assert out[-1].llm is None


def test_top_n_bounds_candidate_set():
    matches = _matches("VP of AI", "VP of Data Science", "Head of AI")
    fake = _FakeReranker(RerankResponse(ranking=[]))
    rerank_matches(matches, VP_AI_PROFILE, reranker=fake, top_n=2)
    assert fake.seen is not None
    assert len(fake.seen) == 2  # only the top 2 went to the LLM
    # The third match (beyond top_n) is preserved as a tail.
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=fake, top_n=2)
    assert len(out) == 3


def test_empty_ranking_preserves_all_candidates_as_layer1():
    matches = _matches("VP of AI", "Head of AI")
    out = rerank_matches(
        matches, VP_AI_PROFILE, reranker=_FakeReranker(RerankResponse(ranking=[]))
    )
    assert [m.job.title for m in out] == [m.job.title for m in matches]
    assert all(m.llm is None for m in out)


def test_no_candidates_returns_matches_unchanged():
    # All matches rejected → nothing to send; return unchanged without calling LLM.
    matches = rank_jobs([_job("Senior ML Engineer")], VP_AI_PROFILE, now=NOW)
    fake = _FakeReranker(RerankResponse(ranking=[]))
    out = rerank_matches(matches, VP_AI_PROFILE, reranker=fake)
    assert out == matches
    assert fake.seen is None  # LLM never invoked


def test_partition_is_order_independent_no_dup_or_drop():
    # rerank_matches must not assume the input is sorted non-rejected-first: even
    # an interleaved list (a rejected match BEFORE a non-rejected one) must yield
    # every job exactly once — no duplicate, no drop.
    rejected = rank_jobs([_job("Senior ML Engineer")], VP_AI_PROFILE, now=NOW)[0]
    good_a = score_job(_job("VP of AI"), VP_AI_PROFILE, now=NOW)
    good_b = score_job(_job("Head of AI"), VP_AI_PROFILE, now=NOW)
    interleaved = [good_a, rejected, good_b]  # rejected sits in the middle

    out = rerank_matches(
        interleaved, VP_AI_PROFILE, reranker=_ReverseReranker(), top_n=20
    )
    # Same multiset of jobs, exactly once each.
    assert sorted(m.job.title for m in out) == sorted(m.job.title for m in interleaved)
    assert len(out) == len(interleaved)
    # The rejected job is never sent to the LLM (no annotation), the two good ones
    # are re-ranked (annotated).
    rejected_out = next(m for m in out if m.rejected)
    assert rejected_out.llm is None
    assert sum(1 for m in out if m.llm is not None) == 2


def test_from_env_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiReranker.from_env() is None


def test_from_env_builds_with_key(monkeypatch):
    # With a key set, from_env builds a reranker (stubbing the genai client so no
    # real network/credentials are touched).
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    import google.genai as genai

    monkeypatch.setattr(genai, "Client", lambda **kw: object())
    reranker = GeminiReranker.from_env()
    assert isinstance(reranker, GeminiReranker)


def test_gemini_reranker_parses_structured_response():
    # Drive GeminiReranker.rerank with a fake genai client whose response carries
    # a parsed RerankResponse — exercises the SDK-parsed happy path, no network.
    matches = _matches("VP of AI", "Head of AI")
    parsed = RerankResponse(
        ranking=[RerankedItem(candidate_id=0, relevance="strong", rationale="x")]
    )

    class _Resp:
        def __init__(self):
            self.parsed = parsed
            self.text = parsed.model_dump_json()

    class _Models:
        def generate_content(self, **kwargs):
            return _Resp()

    class _Client:
        models = _Models()

    reranker = GeminiReranker(_Client())
    out = reranker.rerank(matches, VP_AI_PROFILE)
    assert out == parsed


def test_gemini_reranker_falls_back_to_text_when_parsed_missing():
    matches = _matches("VP of AI")
    payload = RerankResponse(
        ranking=[RerankedItem(candidate_id=0, relevance="moderate", rationale="y")]
    )

    class _Resp:
        parsed = None  # SDK didn't parse; rerank must validate .text
        text = payload.model_dump_json()

    class _Client:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return _Resp()

    reranker = GeminiReranker(_Client())
    out = reranker.rerank(matches, VP_AI_PROFILE)
    assert out.ranking[0].rationale == "y"


def test_score_job_unaffected_smoke():
    # Sanity: a plain Layer-1 match has llm=None (back-compat of the dataclass).
    m = score_job(_job("VP of AI"), VP_AI_PROFILE, now=NOW)
    assert m.llm is None
