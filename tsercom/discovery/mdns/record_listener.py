"""Listener for mDNS service records using zeroconf."""

import asyncio # Added import asyncio
import logging
import uuid
from zeroconf import Zeroconf  # Import Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncZeroconf,
)
from tsercom.discovery.mdns.mdns_listener import MdnsListener


class RecordListener(MdnsListener):
    """Low-level mDNS service listener using `zeroconf`.

    Implements the listener interface for `zeroconf.asyncio.AsyncServiceBrowser`
    by providing async methods for service updates, removals, and additions.
    Monitors for mDNS records of a specified type, notifies client on
    add/update with record details.
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
        # Use AsyncServiceBrowser
        self.__browser = AsyncServiceBrowser(
            self.__mdns.zeroconf, [self.__expected_type], listener=self
        )

    # Synchronous methods called by AsyncServiceBrowser, which then schedule async tasks
    def update_service(self, zc: "Zeroconf", type_: str, name: str) -> None:  # type: ignore[override]
        asyncio.create_task(self._async_update_service(type_, name))

    async def _async_update_service(self, type_: str, name: str) -> None:
        """Async implementation for service update.

        Specified by `MdnsListener` (and expected by `AsyncServiceBrowser`).
        Retrieves updated service info and notifies the client.

        Args:
            type_: Service type updated.
            name: mDNS name of updated service.
        """
        # The zc parameter from the sync wrapper is not used here.
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

    def remove_service(self, zc: "Zeroconf", type_: str, name: str) -> None:  # type: ignore[override]
        asyncio.create_task(self._async_remove_service(type_, name))

    async def _async_remove_service(self, type_: str, name: str) -> None:
        """Async implementation for service removal."""
        # The zc parameter is from the sync callback, not used directly in async logic here.
        logging.info("Service removed: type='%s', name='%s'", type_, name)
        await self.__client._on_service_removed(name, type_, self._uuid_str)

    def add_service(self, zc: "Zeroconf", type_: str, name: str) -> None:  # type: ignore[override]
        asyncio.create_task(self._async_add_service(type_, name))

    async def _async_add_service(self, type_: str, name: str) -> None:
        """Async implementation for new service discovery.

        Specified by `MdnsListener` (and expected by `AsyncServiceBrowser`).
        Retrieves new service info, notifies client. `zeroconf` might also call
        `update_service` for new services; this handles if `add_service` is
        the first callback.

        Args:
            type_: Discovered service type.
            name: mDNS name of discovered service.
        """
        # The zc parameter from the sync wrapper is not used here.
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

    async def close(self) -> None:
        # Closes the mDNS listener and the underlying AsyncZeroconf instance.
        if self.__browser:
            logging.info(
                "Cleaning up AsyncServiceBrowser in RecordListener for %s",
                self.__expected_type,
            )
            # Note: If an error occurs in async_cancel or async_close,
            # it might prevent subsequent cleanup steps.
            # Consider try/except for each await if necessary.
            await self.__browser.async_cancel()
            self.__browser = None

        if self.__mdns:
            try:
                await self.__mdns.async_close()
            except Exception as e:  # pylint: disable=broad-except
                logging.error(
                    "Error during AsyncZeroconf.async_close() in RecordListener: %s",
                    e,
                    exc_info=True,
                )
        logging.info("RecordListener closed for %s", self.__expected_type)
