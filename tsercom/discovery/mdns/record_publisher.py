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
        service_info_at_start_of_close = self._service_info
        unregistration_attempted = False
        unregistration_succeeded = False

        try:
            if self._zc: # Check if _zc is valid before trying to use it
                if service_info_at_start_of_close:
                    _logger.info(f"Attempting to unregister service {self.__srv} using service_info: {service_info_at_start_of_close} (ZC type: {'shared' if self.__shared_zc else 'owned'})")
                    unregistration_attempted = True
                    await self._zc.async_unregister_service(service_info_at_start_of_close)
                    unregistration_succeeded = True
                    _logger.info(f"Service {self.__srv} unregistered successfully.")
                else:
                    _logger.warning(f"No self._service_info found for {self.__srv} at start of close method. Skipping unregistration call.")
                    # If there's no service_info, unregistration wasn't needed for this object's state from publish perspective.
                    unregistration_succeeded = True # Considered successful as no action was pending for this _service_info
            else:
                _logger.warning(f"No active Zeroconf instance (_zc) for {self.__srv}. Cannot unregister.")
                # If _zc is None, we can't unregister, so treat as "nothing to do" for unregistration.
                unregistration_succeeded = True


            # Close owned zeroconf instance if it exists
            if self.__owned_zc:
                _logger.info(f"Closing owned AsyncZeroconf instance for {self.__srv}.")
                await self.__owned_zc.async_close()
                _logger.info(f"Owned AsyncZeroconf instance for {self.__srv} closed.")

        except Exception as e:
            # Log detailed error for unregistration or closing owned_zc
            if unregistration_attempted and not unregistration_succeeded:
                _logger.error(f"CRITICAL: Exception during async_unregister_service for {self.__srv}. Service may still be registered. Error: {e}", exc_info=True)
            else: # Error during closing owned_zc or other unexpected error if _zc was None initially
                _logger.error(f"Exception during close operation for {self.__srv}. Error: {e}", exc_info=True)
        finally:
            # Only nullify _service_info if unregistration was successful or wasn't needed.
            if unregistration_succeeded:
                self._service_info = None
            else:
                _logger.warning(f"self._service_info for {self.__srv} was NOT cleared because unregistration failed or was not confirmed.")

            if self.__owned_zc: # Ensure owned_zc is cleared if it was set
                self.__owned_zc = None
            self._zc = None # Clear the active reference in all cases after attempts

            # Final debug log for state
            if not self._service_info and not self._zc and not self.__owned_zc:
                _logger.debug(f"RecordPublisher for {self.__srv} fully cleaned up (service_info, _zc, _owned_zc are None).")
            else:
                _logger.debug(f"RecordPublisher for {self.__srv} post-close state: _service_info is {'None' if not self._service_info else 'Present'}, _zc is {'None' if not self._zc else 'Present'}, _owned_zc is {'None' if not self.__owned_zc else 'Present'}")
