import dataclasses
from typing import List


@dataclasses.dataclass
class ServiceInfo:
    """Represents information about a discovered service instance.

    This class is typically used to store details obtained from mDNS (Multicast DNS)
    queries, such as the service's name, port, IP addresses, and its unique
    mDNS instance name.
    """

    # Original __init__ order: name: str, port: int, addresses: List[str], mdns_name: str
    name: str
    port: int
    addresses: List[str]
    mdns_name: str
