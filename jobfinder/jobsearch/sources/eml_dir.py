"""Read a directory of saved ``.eml`` job-alert files into RawPostings.

This is Slice A's ingestion: the user saves/exports LinkedIn job-alert emails to
a folder (or a Gmail filter does it), and we read every ``.eml`` there. A live
Gmail API reader is a later slice that produces the same ``RawPosting`` output,
so nothing downstream changes when it lands.
"""

from __future__ import annotations

from pathlib import Path

from jobfinder.jobsearch.models import RawPosting
from jobfinder.jobsearch.sources.linkedin_email import parse_alert_email


def read_eml_dir(path: str | Path) -> list[RawPosting]:
    """Parse every ``*.eml`` in ``path`` (sorted by name) into RawPostings.

    Files that aren't job alerts (or don't parse) contribute nothing rather than
    erroring, so a mixed mail folder is tolerated. Raises ``NotADirectoryError``
    if ``path`` isn't a directory, so a typo'd path fails loudly instead of
    silently yielding zero jobs.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    postings: list[RawPosting] = []
    for eml in sorted(directory.glob("*.eml")):
        postings.extend(parse_alert_email(eml.read_bytes()))
    return postings
