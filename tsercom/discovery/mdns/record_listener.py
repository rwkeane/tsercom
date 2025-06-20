"""Listener for mDNS service records using zeroconf."""

import asyncio
import logging
import uuid
from typing import Optional
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

    def __init__(
        self,
        client: MdnsListener.Client,
        service_type: str,
        zc_instance: Optional[AsyncZeroconf] = None,
    ) -> None:
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

        self.__mdns: AsyncZeroconf
        self.__is_shared_zc: bool = zc_instance is not None
        if zc_instance:
            self.__mdns = zc_instance
            logging.info(
                "Using shared AsyncZeroconf for RecordListener, type: %s",
                self.__expected_type,
            )
        else:
            self.__mdns = AsyncZeroconf()
            logging.info(
                "Created new AsyncZeroconf for RecordListener, type: %s (AsyncServiceBrowser with default IPVersion)",
                self.__expected_type,
            )

        self.__browser: AsyncServiceBrowser | None = None

    async def start(self) -> None:
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

        await self.__client._on_service_added(
            name,
            info.port,
            info.addresses,
            info.properties,
        )
        logging.info(
            "Async handler _handle_update_service completed for: type='%s', name='%s'",
            type_,
            name,
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called by `zeroconf` when a service is removed from the network."""
        logging.info(
            "Sync remove_service called: type='%s', name='%s'. Scheduling async handler.",
            type_,
            name,
        )
        logging.info(
            "[REC_LISTENER] remove_service (sync) called for name: %s, type: %s",
            name,
            type_,
        )
        asyncio.create_task(self._handle_remove_service_wrapper(type_, name))

    async def _handle_remove_service_wrapper(
        self, type_: str, name: str
    ) -> None:
        """Wrapper to ensure a small sleep after task creation from sync context."""
        await self._handle_remove_service(type_, name)
        await asyncio.sleep(
            0
        )  # Yield control to allow the task to potentially start

    async def _handle_remove_service(self, type_: str, name: str) -> None:
        """Async handler for service removal."""
        logging.info(
            "[REC_LISTENER] _handle_remove_service (async) started for name: %s, type: %s",
            name,
            type_,
        )

        logging.info(
            "[REC_LISTENER] _handle_remove_service: About to call client._on_service_removed for %s",
            name,
        )
        await self.__client._on_service_removed(name, type_, self._uuid_str)
        logging.info(
            "[REC_LISTENER] _handle_remove_service: Returned from client._on_service_removed for %s",
            name,
        )
        logging.info(
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
            self.__browser = (
                None  # Browser is cancelled by closing AsyncZeroconf
            )

        if not self.__is_shared_zc and self.__mdns:
            logging.info(
                "Closing owned AsyncZeroconf instance for RecordListener, type: %s",
                self.__expected_type,
            )
            try:
                await self.__mdns.async_close()
            except Exception as e:
                logging.error(
                    "Error during owned AsyncZeroconf.async_close() for %s: %s",
                    self.__expected_type,
                    e,
                    exc_info=True,
                )
        elif self.__is_shared_zc:
            logging.info(
                "Not closing shared AsyncZeroconf instance for RecordListener, type: %s",
                self.__expected_type,
            )

        logging.info(
            "RecordListener close method finished for %s", self.__expected_type
        )
