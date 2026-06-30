"""Job-posting sources for the job-search tool.

Each source turns some external input into ``RawPosting`` records:

- ``linkedin_email`` parses a LinkedIn job-alert email (offline, pure).
- ``eml_dir`` reads a directory of saved ``.eml`` alert files.

ATS boards are read with the core ``jobfinder.sources.ats.AtsClient`` and adapted
to ``RawPosting`` in ``jobfinder.jobsearch.normalize`` (no duplicate fetcher).
"""
