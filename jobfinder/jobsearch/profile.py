"""The target-role profile that every match is scored against.

This is pure configuration: the VP-of-AI role taxonomy, the negative filters
that disqualify obvious mismatches, the AI/data/analytics vocabulary, and the
fit-dimension weights. It mirrors ``jobfinder.fit.CandidateProfile`` in spirit
(a frozen description of what the candidate wants) but is specific to *job
postings* — titles and keywords rather than firmographics.

Matching against these patterns lives in ``jobfinder.jobsearch.match``; this
module only declares *what* to look for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Slice A scores four dimensions; the rest are declared so the breakdown lists
# them, but score NEUTRAL until a later slice fills them in (mirrors fit.py's
# "missing data is neutral, never punitive"). The four live weights sum to the
# share of the 0-100 total that Slice A can actually judge; neutral dimensions
# contribute their weight * 0.5. Weights sum to 1.0 across all dimensions.
DIMENSION_WEIGHTS: dict[str, float] = {
    "title_seniority": 0.30,  # VP/Head/Chief/SVP at an AI/data/analytics function
    "ai_scope": 0.30,  # AI / data / ML / analytics subject-matter relevance
    "location": 0.10,  # remote / preferred-geo fit
    "recency": 0.05,  # newer postings first
    # --- neutral until later slices (declared for an honest, stable breakdown) ---
    "exec_scope": 0.10,  # owns org/budget/strategy (needs description text)
    "company_fit": 0.10,  # industry/stage/AI-intensity (needs firmographics)
    "contactability": 0.05,  # hiring-manager/recruiter discoverability (Slice E)
}

# Dimensions Slice A can actually evaluate. The others score neutral (0.5).
LIVE_DIMENSIONS = ("title_seniority", "ai_scope", "location", "recency")

# Tier thresholds on the 0-100 score (from the product spec).
TIER_A_MIN = 80.0
TIER_B_MIN = 60.0


def _ci(*patterns: str) -> tuple[re.Pattern[str], ...]:
    """Compile case-insensitive whole-pattern matchers."""
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


@dataclass(frozen=True)
class TargetProfile:
    """What roles the candidate is hunting for, and how to weight fit.

    ``primary_title_patterns`` are the bullseye VP/Head/Chief AI-leadership
    titles; ``secondary_title_patterns`` are acceptable only when the rest of
    the fit is strong (Senior Director / Director-of-AI-strategy tier).
    ``ai_keywords`` is the subject vocabulary for the AI-scope dimension.
    ``disqualifying_patterns`` always reject (intern, contract, academic).
    ``ic_role_patterns`` reject *only* when no leadership title rescues them (so
    "Principal ML Engineer" drops but "VP, AI & ML Engineering" survives — see
    ``match``). ``preferred_locations``/``remote_ok`` drive the location dimension.
    """

    primary_title_patterns: tuple[re.Pattern[str], ...]
    secondary_title_patterns: tuple[re.Pattern[str], ...]
    disqualifying_patterns: tuple[re.Pattern[str], ...]
    ic_role_patterns: tuple[re.Pattern[str], ...]
    seniority_patterns: tuple[re.Pattern[str], ...]
    ai_keywords: frozenset[str]
    preferred_locations: tuple[str, ...] = ()
    remote_ok: bool = True
    weights: dict[str, float] = field(default_factory=lambda: dict(DIMENSION_WEIGHTS))


# The default profile for this job search: VP-of-AI and adjacent data/analytics
# leadership. Titles are matched whole-word/phrase, case-insensitively, against a
# posting's normalized title in ``match``.
VP_AI_PROFILE = TargetProfile(
    # Bullseye executive AI/data/analytics leadership titles. Word boundaries keep
    # "VP" from matching inside other words; the role-noun alternation accepts the
    # common phrasings ("VP of AI", "VP, Artificial Intelligence", "VP AI & Data").
    primary_title_patterns=_ci(
        r"\b(?:VP|SVP|EVP|vice president)\b.{0,30}\b(?:AI|artificial intelligence|"
        r"machine learning|\bML\b|data science|data and AI|data & AI|AI and data|"
        r"AI & data|analytics|generative AI|gen ?AI|AI platform|data products?|"
        r"enterprise AI)\b",
        # "VP of Data" as a head noun (data leadership), but NOT off-target
        # "Data <X>" functions (privacy/center/governance/etc.) — those fall
        # through to the generic-seniority partial.
        r"\b(?:VP|SVP|EVP|vice president)\b.{0,30}\bdata\b(?!\s+(?:privacy|"
        r"protection|center|centre|governance|entry|warehouse|quality|"
        r"steward|stewardship|security|compliance))",
        r"\bhead of\b.{0,30}\b(?:AI|artificial intelligence|machine learning|"
        r"\bML\b|data science|data and AI|data & AI|AI/ML|analytics|generative AI)\b",
        r"\bchief\b.{0,20}\b(?:AI|artificial intelligence|data|analytics|"
        r"data and AI|data & AI)\b.{0,20}\bofficer\b",
        r"\b(?:CAIO|CDAO)\b",
    ),
    # Acceptable when company/role quality is high (scored, then capped lower in
    # match so a strong secondary can still reach tier B but rarely tier A).
    secondary_title_patterns=_ci(
        r"\b(?:senior|sr\.?)\s+director\b.{0,30}\b(?:AI|artificial intelligence|"
        r"machine learning|\bML\b|data science|analytics|data)\b",
        r"\bdirector\b.{0,30}\b(?:AI strategy|AI product|AI transformation|"
        r"data science|machine learning)\b",
        r"\b(?:executive director|AI transformation lead|AI/ML platform lead)\b",
    ),
    # Always-disqualifying titles — junior/temporary/academic, regardless of any
    # leadership word (no "Intern VP" rescue case exists in practice).
    disqualifying_patterns=_ci(
        r"\b(?:intern|internship|apprentice|co-op)\b",
        r"\b(?:junior|jr\.?|entry[- ]level)\b",
        r"\b(?:contract|contractor|temporary|temp|staffing)\b",
        r"\b(?:professor|postdoc|post-doc|lecturer|phd researcher)\b",
    ),
    # Individual-contributor role nouns — rejected ONLY when no leadership
    # qualifier (seniority_patterns) is also present, so "ML Engineer" drops but
    # "VP, ML Engineering" survives. ``match`` applies the rescue.
    ic_role_patterns=_ci(
        r"\b(?:engineer|scientist|developer|analyst|researcher|consultant|"
        r"specialist|administrator|associate)\b",
    ),
    # Leadership qualifiers that RESCUE an otherwise-IC title and confirm
    # seniority. These are true *leadership* nouns only — a bare seniority
    # adjective like "Principal" or "Senior" does NOT rescue, so "Principal ML
    # Engineer"/"Senior ML Engineer" stay disqualified ICs while "VP, ML
    # Engineering"/"Director of Data Science" survive (see match._disqualified).
    seniority_patterns=_ci(
        r"\b(?:VP|SVP|EVP|vice president|head of|chief|CAIO|CDAO|director)\b",
    ),
    ai_keywords=frozenset(
        {
            "ai",
            "artificial",
            "intelligence",
            "ml",
            "machine",
            "learning",
            "data",
            "analytics",
            "science",
            "generative",
            "genai",
            "llm",
            "nlp",
            "platform",
            "products",
            "enterprise",
            "transformation",
            "deep",
        }
    ),
    preferred_locations=("remote", "united states", "us", "usa"),
    remote_ok=True,
)
