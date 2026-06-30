"""Tests for confidence-scored business-email inference (Slice E).

Hermetic: pure construction + an injected fake EmailFormatProvider (no network,
no vendor, no key). Covers the business-emails-only guard, the deterministic
confidence ordering, name/domain normalization, and the format-source seam.
"""

from jobfinder.jobsearch.email_format import (
    NullEmailFormatProvider,
    domain_matches,
    guess_emails,
    is_personal_domain,
    normalize_domain,
    split_name,
)


class FakeFormatProvider:
    """A fake provider that knows the format for specific domains only."""

    def __init__(self, known: dict[str, str]):
        self.known = known
        self.calls: list[str] = []

    def lookup_format(self, domain: str) -> str | None:
        self.calls.append(domain)
        return self.known.get(domain)


# --------------------------------------------------------------------------- #
# Normalization helpers.
# --------------------------------------------------------------------------- #
def test_normalize_domain_strips_scheme_path_and_at():
    assert normalize_domain("https://Acme.com/careers") == "acme.com"
    assert normalize_domain("@Acme.COM") == "acme.com"
    assert normalize_domain("jane@acme.com") == "acme.com"
    assert normalize_domain("acme.com:443") == "acme.com"


def test_normalize_domain_rejects_garbage():
    assert normalize_domain("") is None
    assert normalize_domain(None) is None
    assert normalize_domain("not a domain") is None
    assert normalize_domain("localhost") is None  # no dot → not a real mail domain


def test_split_name_first_last_and_mononym():
    assert split_name("Jane Smith") == ("jane", "smith")
    # Middle names ignored; first + last only.
    assert split_name("Jane Q. Public") == ("jane", "public")
    # Apostrophes/hyphens collapsed out of the token.
    assert split_name("Mary O'Neil") == ("mary", "oneil")
    assert split_name("Anne Smith-Jones") == ("anne", "smithjones")
    # A single token can't form first.last patterns.
    assert split_name("Cher") is None
    assert split_name("") is None


def test_split_name_folds_accents_to_ascii():
    # Regression: accented names were mangled (José García → ('jos','a')). They
    # must fold to the ASCII the company's mail system actually uses.
    assert split_name("José García") == ("jose", "garcia")
    assert split_name("Müller Schmidt") == ("muller", "schmidt")
    assert split_name("Ñoño Example") == ("nono", "example")
    assert guess_emails("José García", "acme.com")[0].email == "jose.garcia@acme.com"


def test_non_latin_name_yields_no_guess():
    # A name with no ASCII fold (e.g. CJK) is stripped to nothing → no guess,
    # rather than a corrupted address.
    assert split_name("张伟") is None
    assert guess_emails("张伟", "acme.com") == []


# --------------------------------------------------------------------------- #
# Business-emails-only guard.
# --------------------------------------------------------------------------- #
def test_personal_domains_never_produce_a_guess():
    assert is_personal_domain("gmail.com")
    assert is_personal_domain("jane@OUTLOOK.com")
    assert not is_personal_domain("acme.com")
    # guess_emails refuses a personal domain outright.
    assert guess_emails("Jane Smith", "gmail.com") == []
    assert guess_emails("Jane Smith", "jane@icloud.com") == []


def test_personal_domain_guard_covers_subdomains():
    # Regression: a sub-domain of a free-mail provider must still count personal.
    assert is_personal_domain("mail.gmail.com")
    assert guess_emails("Jane Smith", "mail.gmail.com") == []


def test_domain_matches_walks_parents():
    listed = {"acme.com"}
    assert domain_matches("acme.com", listed)
    assert domain_matches("careers.acme.com", listed)
    assert domain_matches("a.b.acme.com", listed)
    assert not domain_matches("notacme.com", listed)
    assert not domain_matches("acme.org", listed)
    # Must not match on a bare TLD even if (absurdly) listed.
    assert not domain_matches("acme.com", {"com"})


def test_is_suppressed_predicate_filters_at_producer():
    # guess_emails applies the do-not-contact predicate itself, so a suppressed
    # address is never constructed (structural guarantee).
    suppressed = {"jane.smith@acme.com"}
    guesses = guess_emails(
        "Jane Smith", "acme.com", is_suppressed=lambda e: e in suppressed
    )
    emails = {g.email for g in guesses}
    assert "jane.smith@acme.com" not in emails
    assert "jsmith@acme.com" in emails  # a non-suppressed guess survives


def test_is_suppressed_can_remove_everything():
    guesses = guess_emails("Jane Smith", "acme.com", is_suppressed=lambda e: True)
    assert guesses == []


def test_empty_or_malformed_domain_yields_no_guess():
    assert guess_emails("Jane Smith", None) == []
    assert guess_emails("Jane Smith", "not a domain") == []


def test_mononym_yields_no_guess():
    assert guess_emails("Cher", "acme.com") == []


# --------------------------------------------------------------------------- #
# Heuristic construction + ordering.
# --------------------------------------------------------------------------- #
def test_heuristic_guesses_are_business_and_confidence_ordered():
    guesses = guess_emails("Jane Smith", "Acme.com")
    assert guesses, "expected candidate emails"
    # All at the business domain; none personal.
    assert all(g.domain == "acme.com" for g in guesses)
    assert all(g.email.endswith("@acme.com") for g in guesses)
    # Strictly non-increasing confidence (deterministic ordering).
    confidences = [g.confidence for g in guesses]
    assert confidences == sorted(confidences, reverse=True)
    # first.last is the most likely heuristic guess.
    assert guesses[0].pattern == "first.last"
    assert guesses[0].email == "jane.smith@acme.com"
    assert all(g.provenance == "heuristic" for g in guesses)


def test_expected_patterns_present():
    emails = {g.pattern: g.email for g in guess_emails("Jane Smith", "acme.com")}
    assert emails["flast"] == "jsmith@acme.com"
    assert emails["first"] == "jane@acme.com"
    assert emails["firstl"] == "janes@acme.com"
    assert emails["f.last"] == "j.smith@acme.com"


# --------------------------------------------------------------------------- #
# The injected format-source seam.
# --------------------------------------------------------------------------- #
def test_null_provider_is_pure_heuristic():
    a = guess_emails("Jane Smith", "acme.com")
    b = guess_emails("Jane Smith", "acme.com", provider=NullEmailFormatProvider())
    assert [g.email for g in a] == [g.email for g in b]
    assert all(g.provenance == "heuristic" for g in b)


def test_format_source_promotes_confirmed_pattern_to_top():
    provider = FakeFormatProvider({"acme.com": "flast"})
    guesses = guess_emails("Jane Smith", "acme.com", provider=provider)
    assert provider.calls == ["acme.com"]
    top = guesses[0]
    assert top.pattern == "flast"
    assert top.email == "jsmith@acme.com"
    assert top.confidence > 0.9
    assert top.provenance == "format-source"
    # The losers are demoted well below the confirmed pattern.
    assert all(g.confidence < top.confidence for g in guesses[1:])


def test_unknown_confirmed_pattern_falls_back_to_heuristic():
    # A provider that returns a pattern we can't build is ignored (stays heuristic).
    provider = FakeFormatProvider({"acme.com": "weird_unknown_pattern"})
    guesses = guess_emails("Jane Smith", "acme.com", provider=provider)
    assert guesses[0].pattern == "first.last"
    assert all(g.provenance == "heuristic" for g in guesses)


def test_provider_miss_is_heuristic():
    provider = FakeFormatProvider({"other.com": "flast"})
    guesses = guess_emails("Jane Smith", "acme.com", provider=provider)
    assert all(g.provenance == "heuristic" for g in guesses)
