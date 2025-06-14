"""Publishes mDNS service records using zeroconf."""

import asyncio  # Needed for asyncio.sleep
import logging
from typing import Dict, Optional

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.util.ip import get_all_addresses

_logger = logging.getLogger(__name__)


class RecordPublisher(MdnsPublisher):
    """Publishes a service to mDNS with specific record details.

    Uses `zeroconf.asyncio` to construct/register `ServiceInfo`, making service
    discoverable on LAN.
    """

    def __init__(
        self,
        name: str,
        type_: str,
        port: int,
        properties: Optional[Dict[bytes, bytes | None]] = None,
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
        """
        if type_ is None or not type_.startswith("_"):
            raise ValueError(
                f"Service type_ must start with an underscore (e.g., '_myservice'), got '{type_}'."
            )

        if properties is None:
            properties = {}

        self.__ptr: str = f"{type_}._tcp.local."
        self.__srv: str = f"{name}.{self.__ptr}"
        self.__port: int = port
        self.__txt: Dict[bytes, bytes | None] = properties
        self._aiozc: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None

    async def publish(self) -> None:
        """Registers the service with mDNS using AsyncZeroconf.

        If already published, it will attempt to unregister the existing
        service before re-registering the new one.
        """
        if self._aiozc is None:
            self._aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only)
            _logger.info("AsyncZeroconf instance created for %s.", self.__srv)
        else:
            _logger.info(
                "AsyncZeroconf instance for %s already exists. Checking for re-registration.",
                self.__srv,
            )
            if self._service_info:
                try:
                    await self._aiozc.async_unregister_service(
                        self._service_info
                    )
                    _logger.info(
                        "Previously registered service %s unregistered before re-publishing.",
                        self.__srv,
                    )
                except Exception as e:  # pylint: disable=broad-except
                    _logger.warning(
                        "Error unregistering previous service %s: %s. Proceeding with re-registration.",
                        self.__srv,
                        e,
                    )

        # Create/update service info to pick up any IP changes.
        self._service_info = ServiceInfo(
            type_=self.__ptr,
            name=self.__srv,
            addresses=get_all_addresses(),  # Important to get current IPs
            port=self.__port,
            properties=self.__txt,
        )

        if self._aiozc:  # Should always be true here
            await self._aiozc.async_register_service(
                self._service_info, allow_name_change=True
            )
            _logger.info("Service %s registered.", self.__srv)
        else:
            # This case should ideally not be reached if logic is correct
            _logger.error(
                "AsyncZeroconf instance not available for publishing %s.",
                self.__srv,
            )

    async def close(self) -> None:
        """Closes the AsyncZeroconf instance and unregisters services."""
        if self._aiozc is not None:
            _logger.debug(
                "Closing AsyncZeroconf for %s and unregistering.", self.__srv
            )
            try:
                if self._service_info:
                    await self._aiozc.async_unregister_service(
                        self._service_info
                    )
                    _logger.info("Service %s unregistered.", self.__srv)
                    # Add a small delay to allow for network propagation before closing zeroconf
                    await asyncio.sleep(0.1)

                await self._aiozc.async_close()
                _logger.debug(
                    "AsyncZeroconf instance for %s closed.", self.__srv
                )
            except Exception as e:  # pylint: disable=broad-except
                _logger.error(
                    "Error closing AsyncZeroconf for %s: %s", self.__srv, e
                )
            finally:
                self._aiozc = None
                self._service_info = None
        else:
            _logger.debug(
                "AsyncZeroconf for %s already closed or not initialized.",
                self.__srv,
            )
