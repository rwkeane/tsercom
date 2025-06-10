"""MdnsListener ABC and client interface for mDNS service discovery."""

from abc import ABC, abstractmethod
from typing import Dict, List

from zeroconf import ServiceListener


# pylint: disable=W0223 # Relies on InstanceListener fulfilling ServiceListener contract
class MdnsListener(ServiceListener):
    """ABC for mDNS service listeners.

    Extends `zeroconf.ServiceListener`, defines `Client` interface for
    notifying clients about discovered raw service details. Subclasses should
    implement `zeroconf.ServiceListener` methods and use `Client` interface.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Starts listening for mDNS services.
        """
        raise NotImplementedError(
            "MdnsListener.start must be implemented by subclasses."
        )

    # pylint: disable=R0903 # Internal helper/callback class for zeroconf
    class Client(ABC):
        """Interface for `MdnsListener` clients.

        Notified when full service details (name, port, addresses, TXT record)
        for a monitored service type are discovered/updated.
        """

        @abstractmethod
        def _on_service_added(
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
        def _on_service_removed(
            self, name: str, service_type: str, record_listener_uuid: str
        ) -> None:
            """Callback for service removal.

            Args:
                name: Unique mDNS name (e.g., "MyDevice._myservice._tcp.local").
                service_type: The type of the service that was removed.
                record_listener_uuid: The UUID of the RecordListener instance.
            """
