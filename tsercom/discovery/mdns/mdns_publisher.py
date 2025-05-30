"""Defines the MdnsPublisher abstract base class for publishing services via mDNS."""

from abc import ABC, abstractmethod


class MdnsPublisher(ABC):
    """Abstract base class for mDNS service publishers.

    Defines a common interface for publishing service information to
    the local network via mDNS.
    """

    @abstractmethod
    def publish(self) -> None:
        """Publishes the service instance via mDNS.

        Implementations should handle the necessary steps to make the service
        discoverable on the network according to their mDNS library and
        configuration.
        """
        pass
