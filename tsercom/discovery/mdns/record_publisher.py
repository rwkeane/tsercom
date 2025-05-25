from typing import Dict, Optional
from zeroconf import IPVersion, ServiceInfo, Zeroconf

from tsercom.util.ip import get_all_addresses


class RecordPublisher:
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

        # Logging the service being published for traceability.
        # Replacing print with logging for better practice, assuming logger is configured elsewhere.

    def publish(self) -> None:
        """Publishes the service to mDNS.

        Constructs a `zeroconf.ServiceInfo` object with the configured details
        and registers it with a `Zeroconf` instance. The service is published
        using IPVersion.V4Only.
        """
        # `addresses` are fetched dynamically to get all current IPv4 addresses of the host.
        service_info = ServiceInfo(
            type_=self.__ptr,  # The service type (e.g., "_myservice._tcp.local.")
            name=self.__srv,  # The full service name (e.g., "MyDevice._myservice._tcp.local.")
            addresses=get_all_addresses(
                ip_version=IPVersion.V4Only
            ),  # Get IPv4 addresses
            port=self.__port,
            properties=self.__txt,
            # Optional: server name can be set here if needed, e.g., f"{socket.gethostname()}.local."
        )

        zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        zeroconf.register_service(service_info)

        # Logging successful publication.
