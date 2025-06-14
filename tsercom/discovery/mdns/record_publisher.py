"""Publishes mDNS service records using zeroconf."""

import asyncio  # Needed for asyncio.sleep
import logging
from typing import Dict, Optional

from zeroconf import IPVersion, ServiceInfo  # Keep ServiceInfo and IPVersion
from zeroconf.asyncio import AsyncZeroconf  # For AsyncZeroconf

# Removed import from zeroconf.aio

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
        self._aiozc: AsyncZeroconf | None = None  # Renamed and typed
        self._service_info: ServiceInfo | None = None

        # Logging the service being published for traceability.
        # Replacing print with logging for better practice, assuming logger is configured elsewhere.

    async def publish(self) -> None:  # Signature changed
        if self._aiozc is not None:  # Check if AsyncZeroconf instance exists
            _logger.info(
                "AsyncZeroconf instance for %s already exists. Re-registering service.",
                self.__srv,
            )
            # If already published, unregister before re-registering to ensure clean state
            # This assumes _service_info from a previous publish call is still valid
            if self._service_info:
                try:
                    # Corrected: Call as a method of self._aiozc
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
            # else:
            # _logger.warning("AsyncZeroconf exists but no previous service_info. This might be an inconsistent state.")
        else:
            # Create AsyncZeroconf instance if it doesn't exist
            self._aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only)
            _logger.info("AsyncZeroconf instance created for %s.", self.__srv)

        # Create or update service info. It's good to recreate it to pick up any IP changes.
        self._service_info = ServiceInfo(
            type_=self.__ptr,
            name=self.__srv,
            addresses=get_all_addresses(),  # Fetches current IPs
            port=self.__port,
            properties=self.__txt,
        )

        # No need for self._loop anymore
        # self._zc is now self._aiozc

        # Use async_register_service
        # Corrected: Call as a method of self._aiozc
        await self._aiozc.async_register_service(
            self._service_info, allow_name_change=True
        )
        _logger.info(
            "Service %s registered using self._aiozc.async_register_service.",
            self.__srv,
        )

    async def close(self) -> None:
        """Closes the AsyncZeroconf instance and unregisters services."""
        if self._aiozc is not None:  # Check if AsyncZeroconf instance exists
            _logger.debug(
                "Closing AsyncZeroconf for %s and unregistering.", self.__srv
            )
            try:
                if self._service_info:  # Ensure service was registered
                    # Use async_unregister_service
                    # Corrected: Call as a method of self._aiozc
                    await self._aiozc.async_unregister_service(
                        self._service_info
                    )
                    _logger.info(
                        "Service %s unregistered using self._aiozc.async_unregister_service.",
                        self.__srv,
                    )
                    # Add a small delay to allow for network propagation before closing zeroconf
                    await asyncio.sleep(0.1)
                # Use async_close for AsyncZeroconf
                await self._aiozc.async_close()
                _logger.debug(
                    "AsyncZeroconf instance for %s closed.", self.__srv
                )
            # pylint: disable=W0718 # Catch all exceptions to keep publish loop alive
            except Exception as e:
                # Long log line
                _logger.error(
                    "Error closing AsyncZeroconf for %s: %s", self.__srv, e
                )
            finally:
                self._aiozc = None  # Reset AsyncZeroconf instance
                # self._loop is already removed
                self._service_info = None  # Reset service info
        else:
            # Long log line
            _logger.debug(
                "AsyncZeroconf for %s already closed or not initialized.",
                self.__srv,
            )
