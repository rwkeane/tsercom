from abc import ABC, abstractmethod
from typing import Dict, List
from zeroconf import ServiceListener


class MdnsListener(ServiceListener):
    class Client(ABC):
        """Interface for clients of `RecordListener`.

        Implementers are notified when full service details (name, port, addresses,
        TXT record) for a service of the monitored type are discovered or updated.
        """

        @abstractmethod
        def _on_service_added(
            self,
            name: str,  # The mDNS instance name of the service.
            port: int,  # The service port number.
            addresses: List[bytes],  # List of raw binary IP addresses.
            txt_record: Dict[
                bytes, bytes | None
            ],  # Parsed TXT record as a dictionary.
        ) -> None:
            """Callback invoked when a new service is discovered or an existing one is updated.

            Args:
                name: The unique mDNS instance name of the service (e.g., "MyDevice._myservice._tcp.local.").
                port: The network port on which the service is available.
                addresses: A list of raw IP addresses (in binary format) associated with the service.
                           These typically come from A or AAAA records.
                txt_record: A dictionary representing the service's TXT record,
                            containing key-value metadata. Keys are bytes, values are bytes or None.
            """
            # This method must be implemented by concrete client classes.
            raise NotImplementedError(
                "RecordListener.Client._on_service_added must be implemented by subclasses."
            )
