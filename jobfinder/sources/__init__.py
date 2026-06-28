"""External data-source adapters (SEC EDGAR, ATS providers, ...).

Each adapter is responsible for fetching raw source material and shaping it
into typed records. Adapters do not interpret signals — that belongs to the
modules under `jobfinder.signals`.
"""
