import datetime
from typing import Callable, Dict, Optional
from uuid import getnode as get_mac

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.discovery.mdns.record_publisher import RecordPublisher


class InstancePublisher:
    """Publishes a service instance using mDNS (Multicast DNS).

    This class handles the creation of an mDNS instance name (if not provided)
    and prepares a TXT record with basic information like a readable name
    and a publication timestamp. It then uses `RecordPublisher` to announce
    the service instance on the local network.
    """

    def __init__(
        self,
        port: int,
        service_type: str,
        readable_name: str | None = None,
        instance_name: str | None = None,
        *,
        mdns_publisher_factory: Optional[
            Callable[
                [str, str, int, Optional[Dict[bytes, bytes | None]]],
                MdnsPublisher,
            ]
        ] = None,
    ) -> None:
        """Initializes the InstancePublisher.

        Args:
            port: The network port on which the service is available.
            service_type: The mDNS service type string (e.g., "_my_service._tcp.local.").
            readable_name: An optional human-readable name for the service.
                           This will be included in the TXT record if provided.
            instance_name: An optional specific mDNS instance name. If None,
                           a unique name is generated based on port and MAC address,
                           truncated to 15 characters.

        Raises:
            ValueError: If `port` or `service_type` is None (though type hints
                        should prevent this, explicit checks are for runtime safety).
            TypeError: If arguments are not of the expected types.
            RuntimeError: If `_make_txt_record` fails internally.
        """
        if port is None:
            raise ValueError(
                "port argument cannot be None for InstancePublisher."
            )
        if not isinstance(port, int):
            raise TypeError(
                f"port must be an integer, got {type(port).__name__}."
            )

        if service_type is None:
            raise ValueError(
                "service_type argument cannot be None for InstancePublisher."
            )
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be a string, got {type(service_type).__name__}."
            )

        if readable_name is not None and not isinstance(readable_name, str):
            raise TypeError(
                f"readable_name must be a string or None, got {type(readable_name).__name__}."
            )

        if instance_name is not None and not isinstance(instance_name, str):
            raise TypeError(
                f"instance_name must be a string or None, got {type(instance_name).__name__}."
            )

        self.__name: str | None = readable_name

        # The name is based on port and MAC address to provide some uniqueness,
        # and truncated to meet mDNS length recommendations/requirements if necessary.
        effective_instance_name: str
        if instance_name is None:
            mac_address = get_mac()
            generated_name = f"{port}{mac_address}"
            # mDNS instance names are often recommended to be relatively short.
            # Truncating to 15 chars is an arbitrary choice here, specific requirements may vary.
            if len(generated_name) > 15:
                effective_instance_name = generated_name[:15]
            else:
                effective_instance_name = generated_name
        else:
            effective_instance_name = instance_name

        txt_record = self._make_txt_record()
        # This check is defensive; _make_txt_record should always return a dict.
        if txt_record is None:  # Should ideally not be reachable
            raise RuntimeError(
                "_make_txt_record failed to produce a TXT record."
            )

        if mdns_publisher_factory is None:
            # Default factory creates RecordPublisher
            def default_mdns_publisher_factory(eff_inst_name: str, s_type: str, p: int, txt: Optional[Dict[bytes, bytes | None]]) -> MdnsPublisher:
                # RecordPublisher is already imported at the top of the file.
                return RecordPublisher(eff_inst_name, s_type, p, txt)
            self.__record_publisher: MdnsPublisher = default_mdns_publisher_factory(
                effective_instance_name, service_type, port, txt_record
            )
        else:
            # Use provided factory
            self.__record_publisher: MdnsPublisher = mdns_publisher_factory(
                effective_instance_name, service_type, port, txt_record
            )

    def _make_txt_record(self) -> dict[bytes, bytes | None]:
        """Creates the TXT record dictionary for the mDNS announcement.

        The TXT record includes the publication timestamp and, if provided,
        the human-readable name of the service.

        Returns:
            A dictionary where keys and values are bytes, suitable for mDNS TXT records.
        """
        properties: dict[bytes, bytes | None] = {
            b"published_on": self.__get_current_date_time_bytes()
        }

        if self.__name is not None:
            properties[b"name"] = self.__name.encode("utf-8")

        return properties

    def publish(self) -> None:
        """Publishes the service instance using mDNS.

        This method delegates to the underlying `RecordPublisher` to make the
        service visible on the network.
        """
        self.__record_publisher.publish()

    def __get_current_date_time_bytes(self) -> bytes:
        """Gets the current date and time formatted as a UTF-8 encoded string.

        Returns:
            The current timestamp as bytes (e.g., "YYYY-MM-DD HH:MM:SS.ffffff").
        """
        now = datetime.datetime.now()
        # Format includes microseconds for higher precision if needed.
        as_str = now.strftime("%F %T.%f")
        return as_str.encode("utf-8")
