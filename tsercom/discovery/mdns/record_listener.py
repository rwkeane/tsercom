"""Listener for mDNS service records using zeroconf."""

import asyncio  # Added import
import logging
import uuid
from zeroconf import Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncZeroconf,
)
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
        super().__init__()

        self.__client: MdnsListener.Client = client
        self._uuid_str = str(uuid.uuid4())
        self.__expected_type: str
        if service_type.endswith("._tcp.local.") or service_type.endswith(
            "._udp.local."
        ):
            self.__expected_type = service_type
        elif service_type.endswith("._tcp") or service_type.endswith("._udp"):
            self.__expected_type = f"{service_type}.local."
        else:
            self.__expected_type = f"{service_type}._tcp.local."

        self.__mdns: AsyncZeroconf = AsyncZeroconf()
        logging.info(
            "Starting mDNS scan for services of type: %s (AsyncZeroconf with AsyncServiceBrowser, default interfaces/IPVersion)",
            self.__expected_type,
        )
        self.__browser: AsyncServiceBrowser | None = None

    async def start(self) -> None:
        # Use AsyncServiceBrowser, pass the underlying sync Zeroconf instance from AsyncZeroconf
        self.__browser = AsyncServiceBrowser(
            self.__mdns.zeroconf, [self.__expected_type], listener=self
        )

    # --- ServiceListener interface methods ---

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a service's info (e.g., TXT) is updated."""
        logging.info(
            "Sync update_service called: type='%s', name='%s'. Scheduling async handler.",
            type_,
            name,
        )
        asyncio.create_task(self._handle_update_service(type_, name))

    async def _handle_update_service(self, type_: str, name: str) -> None:
        """Async handler for service updates."""
        logging.info(
            "Async handler _handle_update_service started for: type='%s', name='%s'",
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

        # Use self.__mdns (AsyncZeroconf instance) for async operations
        info = await self.__mdns.async_get_service_info(type_, name)
        if info is None:
            logging.error(
                "Failed to get info for updated service '%s' type '%s'.",
                name,
                type_,
            )
            return

        if info.port is None:
            logging.error(
                "No port for updated service '%s' type '%s'.", name, type_
            )
            return

        if not info.addresses:
            logging.warning(
                "No addresses for updated service '%s' type '%s'.", name, type_
            )
            # Decide if to proceed; currently proceeds.

        # pylint: disable=W0212 # Calling listener's notification method
        await self.__client._on_service_added(
            name,
            info.port,
            info.addresses,
            info.properties,  # Pass `name` from args, not info.name
        )
        logging.info(
            "Async handler _handle_update_service completed for: type='%s', name='%s'",
            type_,
            name,
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a service is removed from the network."""
        logging.info( # Existing log
            "Sync remove_service called: type='%s', name='%s'. Scheduling async handler.",
            type_,
            name,
        )
        # Log entry added as per instruction, though it's similar to above.
        logging.info("[REC_LISTENER] remove_service (sync) called for name: %s, type: %s", name, type_)
        asyncio.create_task(self._handle_remove_service(type_, name))

    async def _handle_remove_service(self, type_: str, name: str) -> None:
        """Async handler for service removal."""
        logging.info(
            "[REC_LISTENER] _handle_remove_service (async) started for name: %s, type: %s",
            name,
            type_,
        )
        # Note: The original `remove_service` didn't check type_ == self.__expected_type
        # but it's good practice for handlers. However, the client notification might
        # be for any service if the listener was registered for multiple types by some means.
        # For now, let's assume client is interested in removal of this specific type.
        # if type_ != self.__expected_type: # Consider if this check is needed here
        #     logging.debug("Ignoring removal for '%s', type '%s'. Expected '%s'.", name, type_, self.__expected_type)
        #     return

        # pylint: disable=W0212 # Calling listener's notification method
        logging.info(
            "[REC_LISTENER] _handle_remove_service: About to call client._on_service_removed for %s", name
        )
        await self.__client._on_service_removed(name, type_, self._uuid_str)
        logging.info(
            "[REC_LISTENER] _handle_remove_service: Returned from client._on_service_removed for %s", name
        )
        logging.info( # Existing log
            "Async handler _handle_remove_service completed for: type='%s', name='%s'",
            type_,
            name,
        )

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a new service is discovered."""
        logging.info(
            "Sync add_service called: type='%s', name='%s'. Scheduling async handler.",
            type_,
            name,
        )
        asyncio.create_task(self._handle_add_service(type_, name))

    async def _handle_add_service(self, type_: str, name: str) -> None:
        """Async handler for service additions."""
        logging.info(
            "Async handler _handle_add_service started for: type='%s', name='%s'",
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

        # Use self.__mdns (AsyncZeroconf instance) for async operations
        info = await self.__mdns.async_get_service_info(type_, name)
        if info is None:
            logging.error(
                "Failed to get info for added service '%s' type '%s'.",
                name,
                type_,
            )
            return

        if info.port is None:
            logging.error(
                "No port for added service '%s' type '%s'.", name, type_
            )
            return

        if not info.addresses:
            logging.warning(
                "No addresses for added service '%s' type '%s'.", name, type_
            )
            # As with update_service, decide whether to proceed or return.

        # pylint: disable=W0212 # Calling listener's notification method
        await self.__client._on_service_added(
            name,
            info.port,
            info.addresses,
            info.properties,
        )
        logging.info(
            "Async handler _handle_add_service completed for: type='%s', name='%s'",
            type_,
            name,
        )

    async def close(self) -> None:
        # Closes the mDNS listener and the underlying AsyncZeroconf instance.
        if self.__browser:
            logging.info(
                "Cleaning up AsyncServiceBrowser in RecordListener for %s",
                self.__expected_type,
            )
            # For AsyncServiceBrowser, cancellation is typically handled when
            # the AsyncZeroconf instance it's tied to is closed via async_close().
            # AsyncServiceBrowser itself does not have a cancel() or async_cancel() method.
            # We just set it to None here as its tasks will be cancelled by AsyncZeroconf.
            self.__browser = None

        if self.__mdns:
            try:
                await (
                    self.__mdns.async_close()
                )  # AsyncZeroconf handles closing its sync_zeroconf
            except Exception as e:  # pylint: disable=broad-except
                logging.error(
                    "Error during AsyncZeroconf.async_close(): %s",
                    e,
                    exc_info=True,
                )
        logging.info("RecordListener closed for %s", self.__expected_type)
