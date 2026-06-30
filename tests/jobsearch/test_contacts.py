"""Tests for contact target-list ranking + the manual checklist (Slice E).

Hermetic, pure: the target list is deterministic and explainable, and the
checklist never touches the network (it only suggests what to search for by
hand).
"""

from jobfinder.jobsearch.contacts import render_checklist, target_contacts
from jobfinder.jobsearch.models import CanonicalJob, ContactRole


def _job(company: str = "Acme, Inc.", title: str = "VP of AI") -> CanonicalJob:
    return CanonicalJob(
        company=company,
        title=title,
        normalized_title=title.lower(),
        location="Remote",
    )


def test_target_list_is_ranked_and_explainable():
    targets = target_contacts(_job())
    assert targets, "expected target roles"
    # Priorities are 1-based and strictly increasing in list order.
    assert [t.priority for t in targets] == list(range(1, len(targets) + 1))
    # The hiring-manager chain is first; the recruiter is included.
    assert targets[0].role is ContactRole.HIRING_MANAGER
    roles = {t.role for t in targets}
    assert ContactRole.RECRUITER in roles
    assert ContactRole.FUNCTION_LEADER in roles
    assert ContactRole.EXECUTIVE in roles
    # Every target explains itself and scopes its search to the company.
    for t in targets:
        assert t.rationale
        assert "Acme" in t.linkedin_search


def test_target_list_is_deterministic():
    assert target_contacts(_job()) == target_contacts(_job())


def test_no_company_yields_no_targets():
    assert target_contacts(_job(company="")) == []


def test_checklist_renders_manual_searches():
    job = _job()
    out = render_checklist(job, target_contacts(job))
    assert "Manual contact checklist" in out
    assert "VP of AI" in out
    assert "Acme" in out
    # The operating model is spelled out: search by hand, business emails only.
    assert "BY HAND" in out
    assert "do-not-contact" in out
    # No automation language / URLs that imply scraping.
    assert "scrape" not in out.lower()


def test_checklist_handles_no_company():
    job = _job(company="")
    out = render_checklist(job, target_contacts(job))
    assert "nothing to search for" in out.lower()
