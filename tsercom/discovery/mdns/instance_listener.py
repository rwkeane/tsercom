"""mDNS instance listener, builds on RecordListener."""

import logging
import socket
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Generic, List, Optional

from tsercom.discovery.mdns.mdns_listener import MdnsListener
from tsercom.discovery.mdns.record_listener import RecordListener
from tsercom.discovery.service_info import (
    ServiceInfo,
    ServiceInfoT,
)
from tsercom.threading.aio.aio_utils import run_on_event_loop


MdnsListenerFactory = Callable[[MdnsListener.Client, str], MdnsListener]


class InstanceListener(Generic[ServiceInfoT], MdnsListener.Client):
    """Listens for mDNS service instances and notifies a client.

    Uses `RecordListener` for mDNS records. When complete service info
    (SRV, A/AAAA, TXT) is available, constructs `ServiceInfo` (or subclass)
    and notifies its `Client`.
    """

    # pylint: disable=R0903 # Abstract listener client interface
    class Client(ABC):  # Removed Generic[TServiceInfo]
        """Interface for `InstanceListener` clients.

        Notified when a complete service instance, matching the type
        monitored by `InstanceListener`, is discovered.
        """

        @abstractmethod
        async def _on_service_added(
            self, connection_info: ServiceInfoT
        ) -> None:
            """Callback invoked when a new service instance is discovered.

            Args:
                connection_info: Info about discovered service (`ServiceInfoT`).
            """
            # This method must be implemented by concrete client classes.
            raise NotImplementedError(
                "Client._on_service_added must be implemented by subclasses."
            )

        @abstractmethod
        async def _on_service_removed(self, service_name: str) -> None:
            """Callback invoked when a service instance is removed.

            Args:
                service_name: The mDNS instance name of the removed service.
            """
            # This method must be implemented by concrete client classes.
            raise NotImplementedError(
                "Client._on_service_removed must be implemented by subclasses."
            )

    def __init__(
        self,
        client: "InstanceListener.Client",
        service_type: str,
        *,
        mdns_listener_factory: Optional[MdnsListenerFactory] = None,
    ) -> None:
        """Initializes the InstanceListener.

        Args:
            client: `InstanceListener.Client` for service notifications.
            service_type: mDNS service type (e.g., "_my_service._tcp.local.").

        Raises:
            ValueError: If `client` is None.
            TypeError: If client or service_type has an unexpected type.
        """
        if client is None:
            raise ValueError("Client cannot be None for InstanceListener.")
        if not isinstance(client, InstanceListener.Client):
            # Long error message
            raise TypeError(
                f"Client must be InstanceListener.Client, got {type(client).__name__}."
            )
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be str, got {type(service_type).__name__}."
            )

        self.__client: InstanceListener.Client = client
        # This InstanceListener acts as client to MdnsListener.

        # pylint: disable=W0238 # Listener managed by this class (its lifecycle)
        self.__listener: MdnsListener
        if mdns_listener_factory is None:
            # Default factory creates RecordListener
            def default_mdns_listener_factory(
                listener_client: MdnsListener.Client, s_type: str
            ) -> MdnsListener:
                return RecordListener(listener_client, s_type)

            self.__listener = default_mdns_listener_factory(self, service_type)
        else:
            # Use provided factory
            self.__listener = mdns_listener_factory(self, service_type)

        self.__listener.start()

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
            record_name: mDNS instance name.
            port: Service port number.
            addresses: List of raw binary IP addresses (from A/AAAA records).
            txt_record: Dict representing TXT record data.

        Returns:
            `ServiceInfo` or `None` if essential info (e.g., IPs) missing.
        """
        if not addresses:
            logging.warning(
                "No IP addresses for service '%s' port %s. Cannot populate.",
                record_name,
                port,
            )
            return None

        # TODO(dev): Enhance IP address conversion for IPv6.
        # socket.inet_ntoa() is IPv4-specific. Need address family info.
        # Example IPv6: socket.inet_ntop(socket.AF_INET6, addr_bytes)
        addresses_out: List[str] = []
        for addr_bytes in addresses:
            try:
                # Assumes IPv4 from `socket.inet_ntoa`.
                address_str = socket.inet_ntoa(addr_bytes)
                addresses_out.append(address_str)
            except (socket.error, TypeError, ValueError) as e:
                logging.warning(
                    "Failed to convert address bytes for '%s': %s",
                    record_name,
                    e,
                )
                continue  # Skip this address.

        if not addresses_out:
            logging.warning(
                "No binary IPs converted to string for '%s', port %s.",
                record_name,
                port,
            )
            return None

        # 'name' key is common convention, not strict standard.
        readable_name_str: str
        name_key_bytes = b"name"
        txt_value_bytes = txt_record.get(name_key_bytes)

        if txt_value_bytes is not None:
            try:
                readable_name_str = txt_value_bytes.decode("utf-8")
            except UnicodeDecodeError:
                logging.warning(
                    "Failed to decode TXT 'name' for '%s'. Using record name.",
                    record_name,
                )
                readable_name_str = record_name
        else:
            # If 'name' not in TXT, use mDNS instance name.
            readable_name_str = record_name

        # Subclasses might override _convert_service_info.
        return ServiceInfo(readable_name_str, port, addresses_out, record_name)

    def _convert_service_info(
        self, service_info: ServiceInfo, _txt_record: Dict[bytes, bytes | None]
    ) -> ServiceInfoT:
        """Converts base `ServiceInfo` to specific `ServiceInfoT`.

        Override if `ServiceInfoT` is more specific and needs extra parsing
        of `_txt_record`. Default is passthrough.

        Args:
            service_info: Base `ServiceInfo` from mDNS records.
            _txt_record: Raw TXT data for populating `ServiceInfoT`. (Unused)

        Returns:
            Instance of `ServiceInfoT`. Must be overridden if `ServiceInfoT`
            is a subclass of `ServiceInfo`.
        """
        # This cast is safe if ServiceInfoT is indeed ServiceInfo.
        # If ServiceInfoT is a subclass, this method MUST be overridden in the
        # subclass of InstanceListener to correctly create and return ServiceInfoT.
        # The type: ignore is kept here because this method is a hook for subclasses
        # and its "correctness" depends on how ServiceInfoT is defined by the subclass.
        return service_info  # type: ignore[return-value]

    def _on_service_added(  # Parameter 'name' now matches MdnsListener.Client
        self,
        name: str,  # mDNS instance name
        port: int,  # Service port from SRV record
        addresses: List[bytes],  # IP addresses from A/AAAA records (binary)
        txt_record: Dict[bytes, bytes | None],  # Parsed TXT record
    ) -> None:
        """Callback from `RecordListener` with mDNS records for a service.

        Implements `MdnsListener.Client`. Populates `ServiceInfo`, converts
        to `ServiceInfoT`, then notifies its own client via `_on_service_added`
        on the event loop.

        Args:
            name: mDNS instance name.
            port: Service port number.
            addresses: List of raw binary IP addresses.
            txt_record: Dict of the service's TXT record.
        """
        base_service_info = self.__populate_service_info(
            name, port, addresses, txt_record
        )
        if base_service_info is None:
            # If essential info couldn't be populated, do not proceed.
            return

        # This step allows subclasses to create more specialized info objects.
        typed_service_info = self._convert_service_info(
            base_service_info, txt_record
        )

        # Client's _on_service_added is expected to be a coroutine.
        # pylint: disable=W0212 # Calling client's notification method
        run_on_event_loop(
            partial(self.__client._on_service_added, typed_service_info)
        )

    def _on_service_removed(  # Matches MdnsListener.Client
        self,
        name: str,
        service_type: str,
        record_listener_uuid: str,  # pylint: disable=unused-argument
    ) -> None:
        """Callback from `RecordListener` when a service is removed.

        Implements `MdnsListener.Client`. Notifies its own client via
        `_on_service_removed` on the event loop.

        Args:
            name: mDNS instance name of the removed service.
            service_type: The type of the service removed (unused).
            record_listener_uuid: UUID of the RecordListener (unused).
        """
        # Client's _on_service_removed is expected to be a coroutine.
        # pylint: disable=W0212 # Calling client's notification method
        run_on_event_loop(partial(self.__client._on_service_removed, name))
