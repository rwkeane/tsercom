"""Defines ServiceInfo, a data class holding information about a discovered network service instance, typically obtained via mDNS."""
from typing import List


class ServiceInfo:
    """Represents information about a discovered service instance.

    This class is typically used to store details obtained from mDNS (Multicast DNS)
    queries, such as the service's name, port, IP addresses, and its unique
    mDNS instance name.
    """

    def __init__(
        self, name: str, port: int, addresses: List[str], mdns_name: str
    ) -> None:
        """Initializes a ServiceInfo instance.

        Args:
            name: The human-readable name of the service.
            port: The network port on which the service is available.
            addresses: A list of IP addresses (as strings) for the service.
            mdns_name: The unique mDNS instance name (e.g., "MyService._http._tcp.local.").
        """
        self.__mdns_name: str = mdns_name
        self.__name: str = name
        self.__port: int = port
        self.__addresses: List[str] = addresses

    @property
    def mdns_name(self) -> str:
        """Gets the unique mDNS instance name of the service.

        Returns:
            The mDNS instance name string.
        """
        return self.__mdns_name

    @property
    def name(self) -> str:
        """Gets the human-readable or service-specific name.

        Returns:
            The service name string.
        """
        return self.__name

    @property
    def port(self) -> int:
        """Gets the network port number of the service.

        Returns:
            The port number as an integer.
        """
        return self.__port

    @property
    def addresses(self) -> List[str]:
        """Gets the list of IP addresses for the service.

        Returns:
            A list of IP address strings.
        """
        return self.__addresses
