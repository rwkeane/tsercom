from abc import ABC, abstractmethod
from typing import Dict, List
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
import logging


class RecordListener(ServiceListener):
    """A low-level listener for mDNS service records using the `zeroconf` library.

    This class implements the `zeroconf.ServiceListener` interface. It monitors
    for mDNS records of a specified service type and notifies a registered `Client`
    when services are added or updated, providing detailed record information.
    """

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

    def __init__(
        self, client: "RecordListener.Client", service_type: str
    ) -> None:
        """Initializes the RecordListener.

        Args:
            client: An object implementing the `RecordListener.Client` interface.
            service_type: The base mDNS service type to listen for (e.g., "_myservice").
                          It will be appended with "._tcp.local." internally.

        Raises:
            ValueError: If `client` or `service_type` is None, or if `service_type`
                        does not start with an underscore.
            TypeError: If arguments are not of the expected types.
        """
        if client is None:
            raise ValueError(
                "Client argument cannot be None for RecordListener."
            )
        # Changed from isinstance to hasattr to check for method implementation (duck typing for ABC)
        if not hasattr(client, "_on_service_added") or not callable(
            getattr(client, "_on_service_added")
        ):
            raise TypeError(
                f"Client must implement the RecordListener.Client interface (e.g., _on_service_added method), got {type(client).__name__}."
            )

        if service_type is None:
            raise ValueError(
                "service_type argument cannot be None for RecordListener."
            )
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be a string, got {type(service_type).__name__}."
            )
        # mDNS service types conventionally start with an underscore.
        if not service_type.startswith("_"):
            raise ValueError(
                f"service_type must start with an underscore (e.g., '_my_service'), got '{service_type}'."
            )

        self.__client: RecordListener.Client = client
        # Determine the expected type string for zeroconf
        if service_type.endswith("._tcp.local.") or service_type.endswith(
            "._udp.local."
        ):
            self.__expected_type: str = service_type
        elif service_type.endswith("._tcp") or service_type.endswith(
            "._udp"
        ):  # e.g. _my_service._tcp
            self.__expected_type: str = f"{service_type}.local."
        else:  # e.g. _my_service
            self.__expected_type: str = f"{service_type}._tcp.local."

        self.__mdns: Zeroconf = Zeroconf()
        logging.info(
            f"Starting mDNS scan for services of type: {self.__expected_type}"
        )
        self.__browser: ServiceBrowser = ServiceBrowser(
            self.__mdns, self.__expected_type, self
        )

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a service's information (e.g., TXT record) is updated.

        This method is part of the `zeroconf.ServiceListener` interface.
        It retrieves the updated service information and notifies the client.

        Args:
            zc: The `Zeroconf` instance.
            type_: The type of the service that was updated.
            name: The mDNS instance name of the service that was updated.
        """
        logging.info(
            f"Service updated or added (via update_service): type='{type_}', name='{name}'"
        )
        if type_ != self.__expected_type:
            logging.debug(
                f"Ignoring service update for '{name}' of unexpected type '{type_}'. Expected '{self.__expected_type}'."
            )
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            logging.error(
                f"Failed to get service info for updated service '{name}' of type '{type_}'."
            )
            return

        if info.port is None:  # Port is essential for service communication.
            logging.error(
                f"No port found for updated service '{name}' of type '{type_}'."
            )
            return

        if not info.addresses:  # Addresses are essential.
            logging.warning(
                f"No addresses found for updated service '{name}' of type '{type_}'."
            )
            # Depending on requirements, might return here or proceed without addresses.
            # For now, we proceed as `_on_service_added` handles address list.

        # info.name is the instance name, info.properties is the TXT record.
        self.__client._on_service_added(
            name,
            info.port,
            info.addresses,
            info.properties,  # Pass `name` from args, not info.name
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a service is removed from the network.

        This method is part of the `zeroconf.ServiceListener` interface.
        Currently, it only logs the removal.

        Args:
            zc: The `Zeroconf` instance.
            type_: The type of the service that was removed.
            name: The mDNS instance name of the service that was removed.
        """
        logging.info(f"Service removed: type='{type_}', name='{name}'")
        # Placeholder for any client notification or cleanup logic for service removal.
        # For example, self.__client._on_service_removed(name, type_)
        pass

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a new service is discovered.

        This method is part of the `zeroconf.ServiceListener` interface.
        It retrieves the new service's information and notifies the client.
        Note: `zeroconf` may also call `update_service` for new services. This
        handler ensures services are processed if `add_service` is the first callback.

        Args:
            zc: The `Zeroconf` instance.
            type_: The type of the service that was discovered.
            name: The mDNS instance name of the service that was discovered.
        """
        logging.info(
            f"Service added (via add_service): type='{type_}', name='{name}'"
        )
        if type_ != self.__expected_type:
            logging.debug(
                f"Ignoring added service '{name}' of unexpected type '{type_}'. Expected '{self.__expected_type}'."
            )
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            logging.error(
                f"Failed to get service info for added service '{name}' of type '{type_}'."
            )
            return

        if info.port is None:  # Port is essential.
            logging.error(
                f"No port found for added service '{name}' of type '{type_}'."
            )
            return

        if not info.addresses:  # Addresses are essential.
            logging.warning(
                f"No addresses found for added service '{name}' of type '{type_}'."
            )
            # As with update_service, deciding whether to proceed or return.

        self.__client._on_service_added(
            name,
            info.port,
            info.addresses,
            info.properties,  # Pass `name` from args
        )
