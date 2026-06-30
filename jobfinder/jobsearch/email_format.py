"""Confidence-scored *business* email inference (Slice E).

Given a person's name and a company's email domain, construct the likely
business email addresses — ``first.last@acme.com``, ``flast@acme.com``,
``first@acme.com`` … — each with a confidence score, so the user can pick the
best guess for a (human-approved, later-slice) outreach. This slice builds and
ranks the candidates; it sends nothing.

Two hard rules, enforced here and never bypassed:

1. **Business emails only.** A personal-mail domain (gmail.com, outlook.com, …)
   is refused outright — we never construct or guess a personal address.
2. **The do-not-contact list is always honored.** Suppression is applied by the
   caller (``contacts``/the CLI), but this module exposes the predicate so a
   suppressed person/domain yields *no* guesses anywhere.

Confidence comes from a deterministic prior over the common corporate patterns
(``first.last`` is by far the most common, then ``flast`` …). That prior can be
*sharpened* by an injected ``EmailFormatProvider`` — a seam that mirrors
``EnrichmentClient``/``AtsClient``: network is injected, the default
``NullEmailFormatProvider`` knows nothing (so CI stays hermetic and a run with no
provider is pure-heuristic), and a future "company email format" source
(hunter.io-style, or an LLM) is a drop-in behind the same protocol, degrading to
``None`` without a key. When the provider returns a known pattern for a domain,
that pattern is promoted to a high confidence and the rest demoted, and the
guess records ``provenance="format-source"`` instead of ``"heuristic"``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Container
from typing import Protocol, runtime_checkable

from jobfinder.jobsearch.models import EmailGuess

# Personal / free-mail domains we will NEVER construct an address for. A name at
# one of these is a personal mailbox, not a business contact — out of scope by the
# operating model (business emails only). Lower-cased exact-domain match.
PERSONAL_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "ymail.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "msn.com",
        "icloud.com",
        "me.com",
        "mac.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
        "pm.me",
        "gmx.com",
        "zoho.com",
        "yandex.com",
        "mail.com",
        "fastmail.com",
        "hey.com",
    }
)

# The common corporate local-part patterns, with a deterministic confidence prior
# reflecting roughly how often each is the real format. ``first.last`` dominates
# real corporate mail; single-letter/initial forms and bare ``first`` follow. The
# priors are intentionally well below 1.0 — these are *guesses* until a format
# source confirms one. Order here is the canonical display order on ties.
_PATTERN_PRIORS: tuple[tuple[str, float], ...] = (
    ("first.last", 0.45),
    ("flast", 0.20),
    ("first", 0.12),
    ("firstl", 0.08),
    ("first_last", 0.06),
    ("f.last", 0.04),
    ("lastf", 0.03),
    ("last", 0.02),
)

# Confidence assigned to the pattern a format source positively identifies, and
# the (low) confidence left on the other patterns once one is confirmed.
_CONFIRMED_CONFIDENCE = 0.95
_DEMOTED_CONFIDENCE = 0.02

# Derived once from the canonical table (not rebuilt per call): the set of
# buildable pattern names, and the canonical display order used to break
# confidence ties deterministically.
_PATTERN_NAMES = frozenset(p for p, _ in _PATTERN_PRIORS)
_PATTERN_ORDER = {p: i for i, (p, _) in enumerate(_PATTERN_PRIORS)}

_NAME_CLEAN_RE = re.compile(r"[^a-z\s'-]")
_DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)+$")


@runtime_checkable
class EmailFormatProvider(Protocol):
    """Looks up a company's known email format for a domain.

    Returns the local-part *pattern name* (one of the keys in ``_PATTERN_PRIORS``,
    e.g. ``"first.last"``) when the format is known, else ``None``. Implementations
    must return ``None`` (never raise) when they have no data or aren't configured,
    so the caller can treat "no format source" uniformly and stay heuristic.
    """

    def lookup_format(self, domain: str) -> str | None: ...


class NullEmailFormatProvider:
    """The default provider: no format source bound, so nothing is ever known.

    Keeps runs pure-heuristic and CI hermetic (no network, no vendor, no key) —
    exactly like ``NullEnrichmentClient``. A real provider is a drop-in behind the
    same protocol.
    """

    def lookup_format(self, domain: str) -> str | None:
        return None


def normalize_domain(domain: str | None) -> str | None:
    """Lower-case and strip a domain to its bare host, or ``None`` if unusable.

    Accepts a bare domain (``"Acme.com"``), an ``@domain`` form, or a full URL
    (``"https://acme.com/careers"``) and reduces it to ``"acme.com"``. Returns
    ``None`` when nothing domain-shaped remains, so callers fail closed rather than
    construct a malformed address.
    """
    if not domain:
        return None
    text = domain.strip().lower()
    text = re.sub(r"^[a-z]+://", "", text)  # strip scheme
    text = text.split("/", 1)[0]  # strip path
    text = text.rsplit("@", 1)[-1]  # strip any local-part / leading @
    text = text.split(":", 1)[0]  # strip port
    text = text.strip().strip(".")
    return text if _DOMAIN_RE.match(text) else None


def parent_domains(domain: str) -> list[str]:
    """``domain`` and each parent domain, most-specific first, excluding the TLD.

    "a.b.acme.com" → ["a.b.acme.com", "b.acme.com", "acme.com"]. Used so a match
    on a registrable domain also covers its sub-domains (blocking "acme.com"
    blocks "careers.acme.com"). The bare TLD is excluded so listing "acme.com"
    can never be interpreted as listing all of ".com". ``domain`` is assumed
    already normalized (lower-case bare host)."""
    labels = domain.split(".")
    return [".".join(labels[i:]) for i in range(len(labels) - 1)]


def domain_matches(domain: str, listed: Container[str]) -> bool:
    """True if ``domain`` itself OR any parent domain is in ``listed``.

    A match on a registrable domain must also cover its sub-domains: blocking
    "acme.com" blocks "careers.acme.com", and "gmail.com" being personal makes
    "mail.gmail.com" personal too (see :func:`parent_domains`)."""
    return any(d in listed for d in parent_domains(domain))


def is_personal_domain(domain: str | None) -> bool:
    """True if ``domain`` (or a parent domain) is personal/free-mail (business guard).

    Sub-domains of a personal provider count as personal too, so
    "mail.gmail.com" can't slip past the business-emails-only guard."""
    norm = normalize_domain(domain)
    return norm is not None and domain_matches(norm, PERSONAL_EMAIL_DOMAINS)


def split_name(name: str) -> tuple[str, str] | None:
    """Split a full name into (first, last) ASCII local-part tokens.

    Accents are folded to ASCII first (José → jose, Müller → muller) so an
    accented name yields the address the company's mail system actually uses,
    rather than being mangled into a wrong-but-plausible guess. Then drops
    anything but letters/space/hyphen/apostrophe, collapses hyphens/apostrophes
    out of each token (so "O'Neil" → "oneil", "Smith-Jones" → "smithjones"), and
    uses the FIRST token as the first name and the LAST token as the surname
    (middle names/initials are ignored for address construction). Returns ``None``
    when fewer than two usable tokens remain (a mononym can't form ``first.last``
    patterns reliably).
    """
    # NFKD-decompose then drop combining marks: é → e, ñ → n, ü → u. Non-Latin
    # scripts (e.g. CJK) have no ASCII fold and are stripped by _NAME_CLEAN_RE,
    # so such names correctly yield no guess rather than a corrupted one.
    folded = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in folded if not unicodedata.combining(c))
    cleaned = _NAME_CLEAN_RE.sub(" ", ascii_name.lower())
    tokens = [re.sub(r"['-]", "", t) for t in cleaned.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return None
    return tokens[0], tokens[-1]


def _local_part(pattern: str, first: str, last: str) -> str | None:
    """Build the local part for a pattern, or ``None`` if a pattern needs more."""
    builders = {
        "first.last": lambda: f"{first}.{last}",
        "flast": lambda: f"{first[0]}{last}",
        "first": lambda: first,
        "firstl": lambda: f"{first}{last[0]}",
        "first_last": lambda: f"{first}_{last}",
        "f.last": lambda: f"{first[0]}.{last}",
        "lastf": lambda: f"{last}{first[0]}",
        "last": lambda: last,
    }
    builder = builders.get(pattern)
    return builder() if builder else None


def guess_emails(
    name: str,
    domain: str | None,
    *,
    provider: EmailFormatProvider | None = None,
    is_suppressed: Callable[[str], bool] | None = None,
) -> list[EmailGuess]:
    """Construct confidence-ranked business-email guesses for a person at a domain.

    Returns an empty list (rather than raising) when the name can't be split into
    first/last, the domain is missing/malformed, or the domain is personal — the
    three "can't honestly guess a business email" cases. When a ``provider`` is
    given and positively identifies the domain's format, that pattern is promoted
    to high confidence and tagged ``provenance="format-source"``; otherwise every
    guess carries its heuristic prior.

    ``is_suppressed`` is the do-not-contact guard, applied HERE at the single
    producer of every business address so the "always honored" guarantee is
    structural — every caller (the CLI today, any future digest/UI/outreach
    surface) gets pre-filtered guesses and cannot accidentally leak a suppressed
    address by forgetting to re-filter. A returned guess is therefore never on the
    do-not-contact list.

    Results are sorted by confidence descending, with the canonical pattern order
    breaking ties so the output is deterministic.
    """
    norm = normalize_domain(domain)
    if norm is None or domain_matches(norm, PERSONAL_EMAIL_DOMAINS):
        return []
    parts = split_name(name)
    if parts is None:
        return []
    first, last = parts

    confirmed = provider.lookup_format(norm) if provider is not None else None
    # Only honor a confirmed pattern we actually know how to build.
    if confirmed is not None and confirmed not in _PATTERN_NAMES:
        confirmed = None

    guesses: list[EmailGuess] = []
    for pattern, prior in _PATTERN_PRIORS:
        local = _local_part(pattern, first, last)
        if not local:
            continue
        email = f"{local}@{norm}"
        if is_suppressed is not None and is_suppressed(email):
            continue  # do-not-contact: never emit a suppressed address.
        if confirmed is not None:
            confidence = (
                _CONFIRMED_CONFIDENCE if pattern == confirmed else _DEMOTED_CONFIDENCE
            )
            provenance = "format-source"
        else:
            confidence = prior
            provenance = "heuristic"
        guesses.append(
            EmailGuess(
                email=email,
                pattern=pattern,
                confidence=confidence,
                provenance=provenance,
                domain=norm,
            )
        )
    # Confidence desc; canonical pattern order breaks ties (deterministic output).
    guesses.sort(key=lambda g: (-g.confidence, _PATTERN_ORDER[g.pattern]))
    return guesses
