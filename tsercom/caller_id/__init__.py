"""Initializes the tsercom.caller_id package and exposes its public API.

This package contains classes and utilities related to identifying and managing
callers or clients within the Tsercom system.
"""

# Import key classes from submodules to make them available at the package level.
from tsercom.caller_id.caller_id_map import CallerIdMap
from tsercom.caller_id.caller_identifier_waiter import CallerIdentifierWaiter
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.caller_id.client_id_fetcher import ClientIdFetcher

# Defines the public interface of the tsercom.caller_id package.
# When 'from tsercom.caller_id import *' is used, only these names are imported.
__all__ = [
    "CallerIdMap",
    "CallerIdentifierWaiter",
    "CallerIdentifier",
    "ClientIdFetcher",
]
