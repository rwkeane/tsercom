"""MdnsListener ABC and client interface for mDNS service discovery."""

from abc import ABC, abstractmethod
from typing import Dict, List

from zeroconf import ServiceListener, Zeroconf


# MdnsListener now inherits from zeroconf.ServiceListener
class MdnsListener(ServiceListener):
    """ABC for mDNS service listeners.

    Extends `zeroconf.ServiceListener` and defines an async lifecycle
    (`start`, `close`) and a client notification interface.
    Subclasses will implement the async methods and ServiceListener methods.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Starts listening for mDNS services.
        """
        raise NotImplementedError(
            "MdnsListener.start must be implemented by subclasses."
        )

    # add_service, update_service, remove_service are part of ServiceListener.
    # Subclasses (like RecordListener) will override them with `async def`
    # and use # type: ignore[override] if MyPy complains.
    # We declare them here as abstract to ensure subclasses implement them,
    # matching the synchronous signatures from ServiceListener.
    @abstractmethod
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a new service is discovered."""
        raise NotImplementedError()

    @abstractmethod
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is removed."""
        raise NotImplementedError()

    @abstractmethod
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        raise NotImplementedError()

    @abstractmethod
    async def close(self) -> None:
        """Stops listening and cleans up resources."""
        raise NotImplementedError()

    class Client(ABC):
        """Interface for `MdnsListener` clients.

        Notified when full service details (name, port, addresses, TXT record)
        for a monitored service type are discovered/updated.
        """

        @abstractmethod
        async def _on_service_added(
            self,
            name: str,  # mDNS instance name of the service.
            port: int,  # Service port number.
            addresses: List[bytes],  # Raw binary IP addresses.
            txt_record: Dict[bytes, bytes | None],  # Parsed TXT record.
        ) -> None:
            """Callback for new or updated service discovery.

            Args:
                name: Unique mDNS name (e.g., "MyDevice._myservice._tcp.local").
                port: Network port of the service.
                addresses: List of raw binary IP addresses (A/AAAA records).
                txt_record: Dict of service's TXT record (metadata).
                            Keys are bytes, values are bytes or None.
            """
            # This method must be implemented by concrete client classes.
            raise NotImplementedError(
                "MdnsListener.Client._on_service_added must be implemented by subclasses."
            )

        @abstractmethod
        async def _on_service_removed(
            self, name: str, service_type: str, record_listener_uuid: str
        ) -> None:
            """Callback for service removal.

            Args:
                name: Unique mDNS name (e.g., "MyDevice._myservice._tcp.local").
                service_type: The type of the service that was removed.
                record_listener_uuid: The UUID of the RecordListener instance.
            """
