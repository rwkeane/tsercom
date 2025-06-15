"""Defines the interface for service discovery mechanisms."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo

ServiceInfoT = TypeVar("ServiceInfoT", bound=ServiceInfo)


# pylint: disable=R0903 # Abstract service source interface
class ServiceSource(Generic[ServiceInfoT], ABC):
    """Abstract base for service discovery. Finds services, notifies client."""

    # pylint: disable=R0903 # Abstract listener interface
    class Client(ABC):
        """Interface for `ServiceSource` clients. Notified of new services."""

        @abstractmethod
        async def _on_service_added(
            self,
            connection_info: ServiceInfoT,
            caller_id: CallerIdentifier,
        ) -> None:
            """Called when a new service instance is discovered.

            Args:
                connection_info: Info about the discovered service.
                caller_id: Unique ID for the discovered service instance.
            """
            raise NotImplementedError(
                "ServiceSource.Client._on_service_added must be implemented by subclasses."
            )

        @abstractmethod
        async def _on_service_removed(
            self,
            service_name: str,
            caller_id: CallerIdentifier,
        ) -> None:
            """Callback invoked when a previously discovered service is removed.

            Args:
                service_name: The mDNS instance name of the removed service.
                caller_id: The unique ID that was associated with the service
                           when it was added.
            """
            raise NotImplementedError(
                "ServiceSource.Client._on_service_removed must be implemented by subclasses."
            )

    @abstractmethod
    async def start_discovery(self, client: "ServiceSource.Client") -> None:
        """Starts service discovery.

        `ServiceSource` looks for services, notifies `client` of finds.

        Args:
            client: Client to be notified of discovered services.
        """
