"""Defines InstanceListener, responsible for discovering mDNS service instances.

It listens for mDNS records (SRV, A/AAAA, TXT) via a RecordListener,
constructs ServiceInfo objects from these records, and notifies a client
upon the discovery of a complete service instance.
"""

from abc import ABC, abstractmethod
from functools import partial
import socket
from typing import Callable, Dict, Generic, List, Optional  # Removed TypeVar
import logging

from tsercom.discovery.mdns.mdns_listener import MdnsListener
from tsercom.discovery.service_info import (
    ServiceInfo,
    TServiceInfo,
)  # Import TServiceInfo
from tsercom.discovery.mdns.record_listener import RecordListener
from tsercom.threading.aio.aio_utils import run_on_event_loop


class InstanceListener(Generic[TServiceInfo], MdnsListener.Client):
    """Listens for mDNS service instances of a specified type and notifies a client.

    This class uses a `RecordListener` to monitor for mDNS records. When
    complete service information (SRV, A/AAAA, TXT records) is available,
    it constructs a `ServiceInfo` object (or a subclass `TServiceInfo`),
    and notifies its registered `Client` about the newly discovered service instance.
    """

    class Client(ABC):  # Removed Generic[TServiceInfo]
        """Interface for clients of `InstanceListener`.

        Implementers of this interface are notified when a complete service
        instance, matching the type monitored by the `InstanceListener`,
        is discovered.
        """

        @abstractmethod
        async def _on_service_added(
            self, connection_info: TServiceInfo
        ) -> None:
            """Callback invoked when a new service instance is discovered.

            Args:
                connection_info: Detailed information about the discovered
                                 service, of type `TServiceInfo`.
            """
            # This method must be implemented by concrete client classes.
            raise NotImplementedError(
                "InstanceListener.Client._on_service_added must be implemented by subclasses."
            )

    def __init__(
        self,
        client: "InstanceListener.Client",  # Removed [TServiceInfo]
        service_type: str,
        *,
        mdns_listener_factory: Optional[
            Callable[[MdnsListener.Client, str], MdnsListener]
        ] = None,
    ) -> None:
        """Initializes the InstanceListener.

        Args:
            client: An object implementing the `InstanceListener.Client` interface.
                    It will receive notifications about discovered services.
            service_type: The mDNS service type string to listen for
                          (e.g., "_my_service._tcp.local.").

        Raises:
            ValueError: If `client` is None.
            TypeError: If `client` is not a subclass of `InstanceListener.Client`,
                       or if `service_type` is not a string.
        """
        if client is None:
            raise ValueError(
                "Client argument cannot be None for InstanceListener."
            )
        # isinstance is used here, which is generally preferred for ABCs.
        if not isinstance(
            client, InstanceListener.Client
        ):  # Check against the non-generic Client still works due to MRO
            raise TypeError(
                f"Client must be an instance of InstanceListener.Client, got {type(client).__name__}."
            )
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be a string, got {type(service_type).__name__}."
            )

        self.__client: InstanceListener.Client = (
            client  # Removed [TServiceInfo]
        )
        # This InstanceListener acts as the client to the MdnsListener.

        self.__listener: MdnsListener  # Declare type once for __listener
        if mdns_listener_factory is None:
            # Default factory creates RecordListener
            def default_mdns_listener_factory(
                listener_client: MdnsListener.Client, s_type: str
            ) -> MdnsListener:
                # RecordListener is already imported at the top of the file.
                return RecordListener(listener_client, s_type)

            self.__listener = default_mdns_listener_factory(self, service_type)
        else:
            # Use provided factory
            self.__listener = mdns_listener_factory(self, service_type)

    def __populate_service_info(
        # This method aggregates information from disparate mDNS records (SRV, A/AAAA, TXT)
        # to build a cohesive ServiceInfo object representing a discovered service.
        self,
        record_name: str,  # Typically the mDNS instance name
        port: int,
        addresses: List[bytes],  # List of raw IP addresses (binary format)
        txt_record: Dict[bytes, bytes | None],  # Raw TXT record data
    ) -> Optional[ServiceInfo]:
        """Constructs a `ServiceInfo` object from raw mDNS record data.

        Args:
            record_name: The mDNS instance name of the service.
            port: The service port number.
            addresses: A list of raw IP addresses (in binary format, e.g., from A/AAAA records).
            txt_record: A dictionary representing the TXT record data.

        Returns:
            A `ServiceInfo` object if successful, or `None` if essential information
            (like convertible IP addresses) is missing.
        """
        if not addresses:
            logging.warning(
                f"No IP addresses available for service '{record_name}' at port {port}. Cannot populate ServiceInfo."
            )
            return None

        # TODO(developer/issue_id): Enhance IP address conversion to support IPv6.
        # The current use of socket.inet_ntoa() is IPv4-specific.
        # To handle IPv6, we would need to know the address family (AF_INET or AF_INET6)
        # for each addr_bytes. This might require changes in how RecordListener
        # provides these addresses (e.g., as tuples of (family, bytes), by
        # attempting to guess based on length which is fragile, or by trying both
        # inet_ntoa and inet_ntop).
        # Example for IPv6: socket.inet_ntop(socket.AF_INET6, addr_bytes)
        addresses_out: List[str] = []
        for addr_bytes in addresses:
            try:
                # This assumes IPv4 addresses from `socket.inet_ntoa`.
                # For IPv6, `socket.inet_ntop(socket.AF_INET6, addr_bytes)` would be needed.
                # Current code implies IPv4 from `inet_ntoa`.
                address_str = socket.inet_ntoa(addr_bytes)
                addresses_out.append(address_str)
            except (socket.error, TypeError, ValueError) as e:
                logging.warning(
                    f"Failed to convert address bytes to string for service '{record_name}': {e}"
                )
                continue  # Skip this address and try the next one.

        if not addresses_out:
            logging.warning(
                f"Failed to convert any binary IP addresses to string format for service '{record_name}', port {port}."
            )
            return None

        # The key 'name' is a common convention but not a strict standard.
        readable_name_str: str
        name_key_bytes = b"name"  # Use bytes literal for dict key
        txt_value_bytes = txt_record.get(name_key_bytes)

        if txt_value_bytes is not None:
            try:
                readable_name_str = txt_value_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # If decoding fails, fall back to the record_name.
                logging.warning(
                    f"Failed to decode TXT record 'name' for '{record_name}'. Using record name as fallback."
                )
                readable_name_str = record_name
        else:
            # If 'name' key is not in TXT record, use the mDNS instance name.
            readable_name_str = record_name

        # Subclasses might override _convert_service_info to create a TServiceInfo instance.
        return ServiceInfo(readable_name_str, port, addresses_out, record_name)

    def _convert_service_info(
        self, service_info: ServiceInfo, txt_record: Dict[bytes, bytes | None]
    ) -> TServiceInfo:
        """Converts a base `ServiceInfo` object to the specific `TServiceInfo` type.

        This method is intended to be overridden by subclasses if `TServiceInfo`
        is a more specific type than `ServiceInfo` and requires additional
        parsing of the `txt_record` or other data to populate its fields.
        The default implementation performs a passthrough, which is only safe
        if `TServiceInfo` is `ServiceInfo` itself.

        Args:
            service_info: The base `ServiceInfo` object created from common mDNS records.
            txt_record: The raw TXT record data, which might contain additional
                        information for populating `TServiceInfo`.

        Returns:
            An instance of `TServiceInfo`. If `TServiceInfo` is more specific
            than `ServiceInfo`, this method *must* be overridden.
        """
        # This cast is safe if TServiceInfo is indeed ServiceInfo.
        # If TServiceInfo is a subclass, this method MUST be overridden in the
        # subclass of InstanceListener to correctly create and return TServiceInfo.
        # The type: ignore is kept here because this method is a hook for subclasses
        # and its "correctness" depends on how TServiceInfo is defined by the subclass.
        return service_info  # type: ignore[return-value]

    def _on_service_added(
        self,
        record_name: str,  # mDNS instance name
        port: int,  # Service port from SRV record
        addresses: List[bytes],  # IP addresses from A/AAAA records (binary)
        txt_record: Dict[bytes, bytes | None],  # Parsed TXT record
    ) -> None:
        """Callback from `RecordListener` when all necessary mDNS records for a service are available.

        This method implements the `RecordListener.Client` interface. It populates
        a `ServiceInfo` object, converts it to `TServiceInfo` (potentially via
        subclass override), and then notifies its own client by scheduling the
        `_on_service_added` call on the event loop.

        Args:
            record_name: The mDNS instance name of the service.
            port: The port number of the service.
            addresses: A list of raw binary IP addresses for the service.
            txt_record: A dictionary representing the service's TXT record.
        """
        base_service_info = self.__populate_service_info(
            record_name, port, addresses, txt_record
        )
        if base_service_info is None:
            # If essential info couldn't be populated, do not proceed.
            return

        # This step allows subclasses to create more specialized info objects.
        typed_service_info = self._convert_service_info(
            base_service_info, txt_record
        )

        # The client's _on_service_added method is expected to be a coroutine.
        run_on_event_loop(
            partial(self.__client._on_service_added, typed_service_info)
        )
