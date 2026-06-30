"""``python -m jobfinder.jobsearch`` entrypoint."""

import sys

from jobfinder.jobsearch.cli import main

if __name__ == "__main__":
    sys.exit(main())
