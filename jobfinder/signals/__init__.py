"""Signal-extraction modules.

Each module turns raw source records (from `jobfinder.sources`) into typed
`Signal` objects from `jobfinder.schemas`. Modules are pure in-process code
for now; they are the units we may later extract into A2A services.
"""
