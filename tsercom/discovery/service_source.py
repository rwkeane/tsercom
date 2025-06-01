"""Defines the interface for service discovery mechanisms."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo

TServiceInfo = TypeVar("TServiceInfo", bound=ServiceInfo)


class ServiceSource(Generic[TServiceInfo], ABC):
    """
    Abstract base class for service discovery mechanisms.

    A `ServiceSource` is responsible for finding services of a specific type
    and notifying its client when new services are discovered.
    """

    class Client(ABC):
        """
        Interface for clients of a `ServiceSource`.

        Implementers of this interface are notified when new services are discovered.
        """

        @abstractmethod
        async def _on_service_added(
            self,
            connection_info: TServiceInfo,
            caller_id: CallerIdentifier,
        ) -> None:
            """
            Called when a new service instance is discovered.

            Args:
                connection_info: Detailed information about the discovered service.
                caller_id: The unique identifier for the discovered service instance.
            """
            pass

    @abstractmethod
    async def start_discovery(self, client: "ServiceSource.Client") -> None:
        """
        Starts the service discovery process.

        The `ServiceSource` will begin looking for services and will notify the
        provided `client` when new services are found.

        Args:
            client: The client object to be notified of discovered services.
        """
        pass
