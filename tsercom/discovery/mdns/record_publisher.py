"""Publishes mDNS service records using zeroconf."""

import logging
from typing import Dict, Optional

from zeroconf import ServiceInfo, IPVersion
from zeroconf.asyncio import AsyncZeroconf

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.util.ip import get_all_addresses

_logger = logging.getLogger(__name__)


class RecordPublisher(MdnsPublisher):
    """Publishes a service to mDNS with specific record details.

    Uses `zeroconf` to construct/register `ServiceInfo`, making service
    discoverable on LAN (IPv4).
    """

    def __init__(
        self,
        name: str,  # The mDNS instance name (e.g., "MyDevice")
        type_: str,  # The base service type (e.g., "_myservice")
        port: int,
        properties: Optional[Dict[bytes, bytes | None]] = None,
        zc_instance: Optional[AsyncZeroconf] = None,  # Added
    ) -> None:
        """Initializes the RecordPublisher.

        Args:
            name: Instance name (e.g., "MyDevice"). Forms part of full mDNS name
                  (e.g., "MyDevice._myservice._tcp.local").
            type_: Base service type (e.g., "_http"). Must start with '_'.
            port: Network port service is on.
            properties: Optional dict for TXT record (bytes: bytes/None).
                        Defaults to {}.

        Raises:
            ValueError: If `type_` is None or not `startswith("_")`.
            TypeError: If args are not of expected types (implicit check).
        """
        if type_ is None or not type_.startswith("_"):
            # Long error message
            raise ValueError(
                f"Service type_ must start with an underscore (e.g., '_myservice'), got '{type_}'."
            )

        if properties is None:
            properties = {}

        self.__ptr: str = f"{type_}._tcp.local."
        self.__srv: str = f"{name}.{self.__ptr}"
        self.__port: int = port
        self.__txt: Dict[bytes, bytes | None] = properties
        self.__shared_zc: Optional[AsyncZeroconf] = zc_instance
        self.__owned_zc: Optional[AsyncZeroconf] = None
        self._zc: AsyncZeroconf | None = (
            None  # This will point to either shared_zc or owned_zc
        )
        self._service_info: ServiceInfo | None = None

        # Logging the service being published for traceability.
        # Replacing print with logging for better practice, assuming logger is configured elsewhere.

    async def publish(self) -> None:
        if self._zc:
            _logger.info(
                "Service %s already published. Re-registering.", self.__srv
            )
            # Optionally, unregister first or update. For now, assume
            # re-registering is desired or Zeroconf handles it.

        self._service_info = ServiceInfo(
            type_=self.__ptr,
            name=self.__srv,
            addresses=get_all_addresses(),
            port=self.__port,
            properties=self.__txt,
        )

        if self.__shared_zc:
            self._zc = self.__shared_zc
            _logger.info(
                "Using shared AsyncZeroconf instance for %s.", self.__srv
            )
        else:
            _logger.info(
                "Creating new AsyncZeroconf instance for %s.", self.__srv
            )
            self.__owned_zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
            self._zc = self.__owned_zc

        if self._zc:  # Should always be true if logic above is correct
            await self._zc.async_register_service(self._service_info)
            _logger.info("Service %s registered.", self.__srv)
        else:
            _logger.error(
                "AsyncZeroconf instance not available for %s, cannot register service.",
                self.__srv,
            )

    async def close(self) -> None:
        """Closes the Zeroconf instance and unregisters services."""
        if self._zc and self._service_info:
            try:
                _logger.info(
                    "Attempting to unregister service: %s (using %s ZC instance)",
                    self.__srv,
                    "shared" if self.__shared_zc else "owned",
                )
                await self._zc.async_unregister_service(self._service_info)
                _logger.info(
                    "Service %s successfully unregistered.", self.__srv
                )
            except Exception as e:
                _logger.error(
                    "Error unregistering service %s: %s",
                    self.__srv,
                    e,
                    exc_info=True,
                )
        elif not self._service_info:
            _logger.warning(
                "No service_info to unregister for %s, skipping unregister.",
                self.__srv,
            )

        if self.__owned_zc:
            _logger.info(
                "Attempting to close owned Zeroconf instance for %s.",
                self.__srv,
            )
            try:
                await self.__owned_zc.async_close()
                _logger.info(
                    "Owned Zeroconf instance for %s successfully closed.",
                    self.__srv,
                )
            except Exception as e:
                _logger.error(
                    "Error closing owned Zeroconf for %s: %s",
                    self.__srv,
                    e,
                    exc_info=True,
                )
            finally:
                self.__owned_zc = None
        elif self.__shared_zc:
            _logger.info(
                "Not closing shared Zeroconf instance for %s.", self.__srv
            )

        # Clear references
        self._zc = None
        self._service_info = (
            None  # Already cleared in original if block, but good to ensure
        )

        if (
            not self._zc and not self.__owned_zc
        ):  # Check if everything is cleaned up
            _logger.debug(
                "Zeroconf for %s already closed or not initialized.",
                self.__srv,
            )
