"""MdnsListener ABC and client interface for mDNS service discovery."""

from abc import ABC, abstractmethod
from typing import Dict, List
from zeroconf import ServiceListener

# AsyncServiceListener does not seem to exist in zeroconf.asyncio
# The listener for AsyncServiceBrowser is expected to be an object
# with async methods: update_service, remove_service, add_service,
# but it must still formally be a subclass of zeroconf.ServiceListener.


# pylint: disable=W0223 # Relies on InstanceListener fulfilling ServiceListener contract
class MdnsListener(ServiceListener, ABC):
    """ABC for mDNS service listeners.

    Extends `zeroconf.ServiceListener` and defines an `ABC` for async mDNS listeners.
    Defines `Client` interface for notifying clients about discovered raw
    service details. Subclasses should implement async methods for service
    discovery callbacks (update_service, remove_service, add_service)
    if they are to be used with `zeroconf.asyncio.AsyncServiceBrowser`.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Starts listening for mDNS services.
        """
        raise NotImplementedError(
            "MdnsListener.start must be implemented by subclasses."
        )

    @abstractmethod
    async def close(self) -> None:
        """
        Stops listening for mDNS services and cleans up resources.
        """
        raise NotImplementedError(
            "MdnsListener.close must be implemented by subclasses."
        )

    # Listener methods expected by AsyncServiceBrowser (and ServiceBrowser).
    # These are defined in zeroconf.ServiceListener synchronously.
    # Subclasses like RecordListener will implement these as async,
    # which is compatible with AsyncServiceBrowser.
    # MyPy will complain about overriding sync with async if we redeclare them here
    # as abstract async methods. So, we rely on ServiceListener's definitions
    # and ensure RecordListener implements them as async.
    # @abstractmethod
    # async def update_service(self, zc: "AsyncZeroconf", type_: str, name: str) -> None: # type: ignore # noqa: F821
    #     pass
    #
    # @abstractmethod
    # async def remove_service(self, zc: "AsyncZeroconf", type_: str, name: str) -> None: # type: ignore # noqa: F821
    #     pass
    #
    # @abstractmethod
    # async def add_service(self, zc: "AsyncZeroconf", type_: str, name: str) -> None: # type: ignore # noqa: F821
    #     pass

    # pylint: disable=R0903 # Internal helper/callback class for zeroconf
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
