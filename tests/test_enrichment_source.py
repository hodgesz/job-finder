"""Tests for the firmographic enrichment seam (offline; fetcher injected)."""

from jobfinder.sources.enrichment import (
    Enrichment,
    EnrichmentClient,
    NullEnrichmentClient,
)


def test_null_client_fills_nothing():
    # The default keeps live runs free and hermetic: no data, ever.
    client = NullEnrichmentClient()
    result = client.enrich(cik="320193", name="Apple Inc.")
    assert result == Enrichment()


def test_null_client_satisfies_protocol():
    # The runtime-checkable Protocol is what the assembler depends on, so the
    # default must satisfy it (and a future vendor client just has to as well).
    assert isinstance(NullEnrichmentClient(), EnrichmentClient)


def test_concrete_client_uses_injected_fetcher_offline():
    # A stand-in vendor client proves the seam works with no network: it reads a
    # fake fetcher and returns structured stage/size, exactly as a real Clearbit/
    # PeopleDataLabs client would behind its own injected Fetcher.
    calls: list[str] = []

    class FakeVendorClient:
        def __init__(self, fetch):
            self._fetch = fetch

        def enrich(self, *, cik, name):
            body = self._fetch(f"https://vendor.example/lookup?cik={cik}")
            stage, count = body.split(",")
            return Enrichment(funding_stage=stage, employee_count=int(count))

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return "series_b,180"

    client = FakeVendorClient(fake_fetch)
    assert isinstance(client, EnrichmentClient)
    result = client.enrich(cik="42", name="Acme")
    assert result == Enrichment(funding_stage="series_b", employee_count=180)
    assert calls == ["https://vendor.example/lookup?cik=42"]
