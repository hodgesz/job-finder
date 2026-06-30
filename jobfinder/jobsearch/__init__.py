"""Personal job-search automation — an ISOLATED side-tool (a detour).

This subpackage is **not** part of the core "Executive Opportunity Intelligence
System" (the 8-K / Form D / scoring pipeline rooted at ``jobfinder.schemas`` and
``jobfinder.scoring``). It exists only to support a personal hunt for VP-of-AI /
VP-of-AI-&-Data / VP-of-AI-&-Analytics roles, single-user and self-only.

Design boundary (kept deliberately):

- **Decoupled from the core.** It reuses *infrastructure* patterns (the injectable
  ``Fetcher``/``AtsClient`` in ``jobfinder.sources.ats``, the whole-word token
  matching idiom from ``jobfinder.fit``) but defines its OWN domain models
  (``jobfinder.jobsearch.models``) and never imports ``schemas``/``scoring`` —
  job-posting-vs-candidate fit is a different concept from company-opportunity
  scoring.
- **Own entrypoint.** ``python -m jobfinder.jobsearch`` (the core ``jobfinder.cli``
  is untouched).
- **Compliance first.** No LinkedIn scraping/automation/messaging — LinkedIn enters
  only via its own job-alert *emails* and manual review links. Any outbound action
  (applying, emailing) is draft-and-approve, never autonomous.

Slice A (this iteration): parse saved LinkedIn job-alert ``.eml`` files, join them
with the existing public ATS feeds, de-duplicate, and rank against a VP-of-AI
target profile with a deterministic (no-LLM, offline) Layer-1 scorer.
"""
