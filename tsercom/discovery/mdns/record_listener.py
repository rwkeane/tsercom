"""Listener for mDNS service records using zeroconf."""

import logging
import uuid

from zeroconf import ServiceBrowser, Zeroconf

from tsercom.discovery.mdns.mdns_listener import MdnsListener


class RecordListener(MdnsListener):
    """Low-level mDNS service listener using `zeroconf`.

    Implements `zeroconf.ServiceListener`. Monitors for mDNS records of a
    specified type, notifies client on add/update with record details.
    """

    def __init__(self, client: MdnsListener.Client, service_type: str) -> None:
        """Initializes the RecordListener.

        Args:
            client: Implements `MdnsListener.Client` interface.
            service_type: Base mDNS service type (e.g., "_myservice").
                          Appended with "._tcp.local." internally.

        Raises:
            ValueError: If args invalid or service_type missing leading '_'.
            TypeError: If args are not of expected types.
        """
        if client is None:
            raise ValueError("Client cannot be None for RecordListener.")

        if service_type is None:
            raise ValueError("service_type cannot be None for RecordListener.")
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be str, got {type(service_type).__name__}."
            )
        # mDNS service types usually start with an underscore.
        if not service_type.startswith("_"):
            # Long error message
            raise ValueError(
                f"service_type must start with '_', got '{service_type}'."
            )

        self.__client: MdnsListener.Client = client
        self._uuid_str = str(
            uuid.uuid4()
        )  # Unique ID for this listener instance
        self.__expected_type: str  # Declare type once
        # Determine the expected type string for zeroconf
        if service_type.endswith("._tcp.local.") or service_type.endswith(
            "._udp.local."
        ):
            self.__expected_type = service_type
        elif service_type.endswith("._tcp") or service_type.endswith(
            "._udp"
        ):  # e.g. _my_service._tcp
            self.__expected_type = f"{service_type}.local."
        else:  # e.g. _my_service
            self.__expected_type = f"{service_type}._tcp.local."

        self.__mdns: Zeroconf = Zeroconf()
        logging.info(
            "Starting mDNS scan for services of type: %s", self.__expected_type
        )
        # pylint: disable=W0238 # __browser is used by zeroconf for its lifecycle
        self.__browser: ServiceBrowser | None = None

    def start(self) -> None:
        # pylint: disable=W0238 # __browser is used by zeroconf for its lifecycle
        self.__browser = ServiceBrowser(
            self.__mdns, self.__expected_type, self
        )

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a service's info (e.g., TXT) is updated.

        Part of `zeroconf.ServiceListener`. Retrieves updated service info
        and notifies the client.

        Args:
            zc: `Zeroconf` instance.
            type_: Service type updated.
            name: mDNS name of updated service.
        """
        logging.info(
            "Service updated or added (via update_service): type='%s', name='%s'",
            type_,
            name,
        )
        if type_ != self.__expected_type:
            logging.debug(
                "Ignoring update for '%s', type '%s'. Expected '%s'.",
                name,
                type_,
                self.__expected_type,
            )
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            logging.error(
                "Failed to get info for updated service '%s' type '%s'.",
                name,
                type_,
            )
            return

        if info.port is None:  # Port is essential
            logging.error(
                "No port for updated service '%s' type '%s'.", name, type_
            )
            return

        if not info.addresses:  # Addresses are essential
            logging.warning(
                "No addresses for updated service '%s' type '%s'.", name, type_
            )
            # Decide if to proceed; currently proceeds.

        # info.name is instance name, info.properties is TXT.
        # pylint: disable=W0212 # Calling listener's notification method
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
        logging.info("Service removed: type='%s', name='%s'", type_, name)
        self.__client._on_service_removed(name, type_, self._uuid_str)

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a new service is discovered.

        Part of `zeroconf.ServiceListener`. Retrieves new service info,
        notifies client. `zeroconf` might also call `update_service` for new
        services; this handles if `add_service` is the first callback.

        Args:
            zc: `Zeroconf` instance.
            type_: Discovered service type.
            name: mDNS name of discovered service.
        """
        logging.info(
            "Service added (via add_service): type='%s', name='%s'",
            type_,
            name,
        )
        if type_ != self.__expected_type:
            logging.debug(
                "Ignoring added service '%s', type '%s'. Expected '%s'.",
                name,
                type_,
                self.__expected_type,
            )
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            logging.error(
                "Failed to get info for added service '%s' type '%s'.",
                name,
                type_,
            )
            return

        if info.port is None:  # Port is essential
            logging.error(
                "No port for added service '%s' type '%s'.", name, type_
            )
            return

        if not info.addresses:  # Addresses are essential
            logging.warning(
                "No addresses for added service '%s' type '%s'.", name, type_
            )
            # As with update_service, decide whether to proceed or return.

        # pylint: disable=W0212 # Calling listener's notification method
        self.__client._on_service_added(
            name,
            info.port,
            info.addresses,
            info.properties,  # Pass `name` from args
        )
