"""Publishes mDNS service records using zeroconf."""

import logging

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.util.ip import get_all_addresses

_logger = logging.getLogger(__name__)

# Note on mDNS Name Reuse with Shared AsyncZeroconf:
# When using a shared AsyncZeroconf instance across multiple RecordPublisher
# instances or for rapid unregister/re-register sequences of the same service
# instance name, `python-zeroconf` might exhibit caching behaviors or require
# a delay for the name to be fully released. Attempting to re-register the
# exact same service instance name immediately after unregistration on the same
# shared AsyncZeroconf instance can lead to `NonUniqueNameException`.
# For service "updates" in such scenarios, consider using slightly varied
# instance names (e.g., appending a version identifier) if immediate
# re-registration with the same logical service identity is critical.


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
        properties: dict[bytes, bytes | None] | None = None,
        zc_instance: AsyncZeroconf | None = None,  # Added
    ) -> None:
        """Initialize the RecordPublisher.

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
                f"Service type_ must start with an underscore (e.g., '_myservice'), "
                f"got '{type_}'."
            )

        if properties is None:
            properties = {}

        self.__ptr: str = f"{type_}._tcp.local."
        self.__srv: str = f"{name}.{self.__ptr}"
        self.__port: int = port
        self.__txt: dict[bytes, bytes | None] = properties
        self.__shared_zc: AsyncZeroconf | None = zc_instance
        self.__owned_zc: AsyncZeroconf | None = None
        self._zc: AsyncZeroconf | None = (
            None  # This will point to either shared_zc or owned_zc
        )
        self._service_info: ServiceInfo | None = None

        # Logging the service being published for traceability.
        # Replacing print with logging for better practice, assuming logger is
        # configured elsewhere.

    async def publish(self) -> None:
        """Publish the service to make it discoverable via mDNS.

        Constructs a `zeroconf.ServiceInfo` object with the configured
        details (name, type, port, properties, addresses) and registers it
        with an `AsyncZeroconf` instance. If an `AsyncZeroconf` instance was
        provided during initialization (shared), it's used. Otherwise, a new
        `AsyncZeroconf` instance is created and managed by this publisher.
        """
        if self._zc:
            _logger.info("Service %s already published. Re-registering.", self.__srv)
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
            _logger.info("Using shared AsyncZeroconf instance for %s.", self.__srv)
        else:
            _logger.info("Creating new AsyncZeroconf instance for %s.", self.__srv)
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
        """Close the Zeroconf instance and unregister services."""
        service_info_at_start_of_close = self._service_info
        unregistration_attempted = False
        unregistration_succeeded = False

        try:
            if self._zc:  # Check if _zc is valid before trying to use it
                if service_info_at_start_of_close:
                    _logger.info(
                        "Attempting to unregister service %s using service_info: %s "
                        "(ZC type: %s)",
                        self.__srv,
                        service_info_at_start_of_close,
                        ("shared" if self.__shared_zc else "owned"),
                    )
                    unregistration_attempted = True
                    await self._zc.async_unregister_service(
                        service_info_at_start_of_close
                    )
                    unregistration_succeeded = True
                    _logger.info("Service %s unregistered successfully.", self.__srv)
                else:
                    _logger.warning(
                        "No self._service_info found for %s at start of close "
                        "method. Skipping unregistration call.",
                        self.__srv,
                    )
                    # If there's no service_info, unregistration wasn't needed
                    # for this object's state from publish perspective.
                    unregistration_succeeded = True  # Considered successful as no
                    # action was pending for this
                    # _service_info
            else:
                _logger.warning(
                    "No active Zeroconf instance (_zc) for %s. Cannot unregister.",
                    self.__srv,
                )
                # If _zc is None, we can't unregister, so treat as "nothing to do"
                # for unregistration.
                unregistration_succeeded = True

            # Close owned zeroconf instance if it exists
            if self.__owned_zc:
                _logger.info("Closing owned AsyncZeroconf instance for %s.", self.__srv)
                await self.__owned_zc.async_close()
                _logger.info("Owned AsyncZeroconf instance for %s closed.", self.__srv)

        except Exception as e:
            # Log detailed error for unregistration or closing owned_zc
            if unregistration_attempted and not unregistration_succeeded:
                _logger.error(
                    "CRITICAL: Exception during async_unregister_service for %s. "
                    "Service may still be registered. Error: %s",
                    self.__srv,
                    e,
                    exc_info=True,
                )
            else:  # Error during closing owned_zc or other unexpected error if _zc
                # was None initially
                _logger.error(
                    "Exception during close operation for %s. Error: %s",
                    self.__srv,
                    e,
                    exc_info=True,
                )
        finally:
            # Only nullify _service_info if unregistration was successful or
            # wasn't needed.
            if unregistration_succeeded:
                self._service_info = None
            else:
                _logger.warning(
                    "self._service_info for %s was NOT cleared because "
                    "unregistration failed or was not confirmed.",
                    self.__srv,
                )

            if self.__owned_zc:  # Ensure owned_zc is cleared if it was set
                self.__owned_zc = None
            self._zc = None  # Clear the active reference in all cases after attempts

            # Final debug log for state
            if not self._service_info and not self._zc and not self.__owned_zc:
                _logger.debug(
                    "RecordPublisher for %s fully cleaned up (service_info, _zc, "
                    "_owned_zc are None).",
                    self.__srv,
                )
            else:
                _logger.debug(
                    "RecordPublisher for %s post-close state: _service_info is %s, "
                    "_zc is %s, _owned_zc is %s",
                    self.__srv,
                    ("None" if not self._service_info else "Present"),
                    ("None" if not self._zc else "Present"),
                    ("None" if not self.__owned_zc else "Present"),
                )


# === Developer Note: mDNS Name Reuse with python-zeroconf ===
# Observations during testing (e.g., in
# `discovery_e2etest.py::test_instance_update_reflects_changes`) suggest that
# when using a shared `AsyncZeroconf` instance, or even with separate instances
# in rapid succession, `python-zeroconf` can exhibit sensitivities
# to unregistering a service instance name and then immediately re-registering
# the exact same name. This can lead to `zeroconf._exceptions.NonUniqueNameException`
# or `zeroconf._exceptions.NotRunningException` if the shared instance's state
# becomes problematic after the first unregistration.
#
# Workarounds for tests or applications needing to simulate service "updates"
# by replacing an old service with a new one under the same logical identity include:
#   1. Using slightly different mDNS instance names for the "updated" service
#      (e.g., appending a version suffix like "_v2").
#   2. Ensuring the old `AsyncZeroconf` instance used for the original service
#      is completely closed, and a new `AsyncZeroconf` instance is used for
#      the listener and the "updated" service publisher, if strict name reuse is
#      attempted. This was the pattern adopted to fix
#      `test_instance_update_reflects_changes`.
#   3. Introducing significant delays between unregistration and re-registration,
#      though the necessary duration can be unreliable.
#
# This behavior appears related to internal caching or state management within
# `python-zeroconf` concerning service name lifecycle.
# =====================================================================
