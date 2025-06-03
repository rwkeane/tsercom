import logging  # For _logger
from typing import Dict, Optional
from asyncio import AbstractEventLoop  # For type hinting
from zeroconf import IPVersion, ServiceInfo, Zeroconf # type: ignore[import-not-found]

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.util.ip import get_all_addresses

_logger = logging.getLogger(__name__)


class RecordPublisher(MdnsPublisher):
    """Publishes a service instance to mDNS using specific record details.

    This class takes detailed service parameters (name, type, port, properties)
    and uses the `zeroconf` library to construct and register a `ServiceInfo`
    object, making the service discoverable on the local network. It specifically
    publishes services using IPv4.
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
            name: The specific instance name for the service being published.
                  This will be part of the full mDNS service name
                  (e.g., "MyDevice._myservice._tcp.local.").
            type_: The base type of the service (e.g., "_http", "_myservice").
                   It must start with an underscore.
            port: The network port on which the service is available.
            properties: An optional dictionary for the TXT record, where keys
                        are bytes and values are bytes or None. Defaults to an
                        empty dictionary if None.

        Raises:
            ValueError: If `type_` is None or does not start with an underscore.
            TypeError: If arguments are not of the expected types (implicitly checked
                       by zeroconf or Python, but good to be aware).
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
        self._zc: Zeroconf | None = None
        self._loop: AbstractEventLoop | None = None
        self._service_info: ServiceInfo | None = (
            None  # To store for unregistering
        )

        # Logging the service being published for traceability.
        # Replacing print with logging for better practice, assuming logger is configured elsewhere.

    async def publish(self) -> None:
        if self._zc:
            _logger.info(
                "Service %s already published. Re-registering.", self.__srv
            )
            # Optionally, unregister first or update if supported,
            # for now, we assume re-registering is the desired behavior or that
            # Zeroconf handles re-registration of the same ServiceInfo correctly.
            # await self.close() # This would unregister and close before re-registering

        self._service_info = ServiceInfo(  # Store service_info
            type_=self.__ptr,
            name=self.__srv,
            addresses=get_all_addresses(),
            port=self.__port,
            properties=self.__txt,
        )

        import asyncio

        self._loop = asyncio.get_running_loop()

        self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        await self._loop.run_in_executor(
            None, self._zc.register_service, self._service_info
        )
        _logger.info("Service %s registered.", self.__srv)

    async def close(self) -> None:
        """Closes the Zeroconf instance and unregisters services."""
        if self._zc and self._loop:
            _logger.debug(
                "Closing Zeroconf instance for %s and unregistering service.",
                self.__srv,
            )
            try:
                if (
                    self._service_info
                ):  # Ensure service was registered before trying to unregister
                    await self._loop.run_in_executor(
                        None, self._zc.unregister_service, self._service_info
                    )
                    _logger.info("Service %s unregistered.", self.__srv)
                await self._loop.run_in_executor(None, self._zc.close)
                _logger.debug("Zeroconf instance for %s closed.", self.__srv)
            except Exception as e:
                _logger.error(
                    "Error closing Zeroconf for %s: %s", self.__srv, e
                )
            finally:
                self._zc = None
                self._loop = None
                self._service_info = None
        else:
            _logger.debug(
                "Zeroconf instance for %s already closed or not initialized.",
                self.__srv,
            )
