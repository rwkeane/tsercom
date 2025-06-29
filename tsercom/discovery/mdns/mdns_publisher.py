"""MdnsPublisher ABC for publishing services via mDNS."""

from abc import ABC, abstractmethod


class MdnsPublisher(ABC):
    """Abstract base class for mDNS service publishers.

    Defines a common interface for publishing service information to
    the local network via mDNS.
    """

    @abstractmethod
    async def publish(self) -> None:
        """Publish the service instance via mDNS.

        Implementations should handle the necessary steps to make the service
        discoverable on the network according to their mDNS library and
        configuration.
        """
