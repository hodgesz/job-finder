"""Job-finder: surfacing hidden-market opportunities from public signals.

This package is intentionally a modular monolith. Specialist "agents"
(SEC 8-K, Form D, scoring, ...) live as in-process modules and are only
extracted into standalone A2A services once a real service boundary is
justified. See the schemas module for the shared domain contract that
every module reads and writes.
"""
