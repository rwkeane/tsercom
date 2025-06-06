"""Defines the base ServiceInfo class and related types."""

import dataclasses
from typing import (
    List,
    TypeVar,
)


@dataclasses.dataclass
class ServiceInfo:
    """Represents information about a discovered service instance.

    This class is typically used to store details obtained from mDNS
    (Multicast DNS) queries, such as name, port, IP addresses, and unique
    mDNS instance name.
    """

    name: str
    port: int
    addresses: List[str]
    mdns_name: str


# Define ServiceInfoT here for common use
ServiceInfoT = TypeVar("ServiceInfoT", bound="ServiceInfo")
