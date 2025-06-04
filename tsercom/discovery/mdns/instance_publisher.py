"""InstancePublisher for mDNS service announcement with TXT prep."""

import datetime
import logging
from typing import Callable, Dict, Optional
from uuid import getnode as get_mac

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.discovery.mdns.record_publisher import RecordPublisher

_logger = logging.getLogger(__name__)


class InstancePublisher:
    """Publishes a service instance using mDNS.

    Handles mDNS instance name creation (if not provided) and prepares
    a TXT record (name, timestamp). Uses `RecordPublisher` to announce.
    """

    # pylint: disable=R0913,R0912 # Handles complex initialization logic for service properties
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
            port: Network port of the service.
            service_type: mDNS service type (e.g., "_my_service._tcp.local.").
            readable_name: Optional human-readable name for TXT record.
            instance_name: Optional mDNS instance name. If None, a unique name
                           is generated (port + MAC, truncated to 15 chars).

        Raises:
            ValueError: If port/service_type is None or invalid.
            TypeError: If arguments have unexpected types.
            RuntimeError: If _make_txt_record fails.
        """
        if port is None:
            raise ValueError("port cannot be None for InstancePublisher.")
        if not isinstance(port, int):
            raise TypeError(f"port must be int, got {type(port).__name__}.")

        if service_type is None:
            raise ValueError(
                "service_type cannot be None for InstancePublisher."
            )
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be str, got {type(service_type).__name__}."
            )

        # Ensure service_type is base type (e.g., "_myservice") for RecordPublisher
        base_service_type = service_type
        suffix_to_remove = "._tcp.local."
        if base_service_type.endswith(suffix_to_remove):
            base_service_type = base_service_type[: -len(suffix_to_remove)]

        if readable_name is not None and not isinstance(readable_name, str):
            raise TypeError(
                f"readable_name must be str or None, got {type(readable_name).__name__}."
            )

        if instance_name is not None and not isinstance(instance_name, str):
            raise TypeError(
                f"instance_name must be str or None, got {type(instance_name).__name__}."
            )

        self.__name: str | None = readable_name

        # Name based on port/MAC for uniqueness, truncated for mDNS if needed.
        effective_instance_name: str
        if instance_name is None:
            mac_address = get_mac()
            generated_name = f"{port}{mac_address}"
            # TODO(dev): Verify 15-char truncation for generated names.
            # mDNS labels can be up to 63 chars. Adjust if needed.
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
                "_make_txt_record failed to produce TXT record."
            )

        self.__record_publisher: MdnsPublisher  # Declare type once
        if mdns_publisher_factory is None:
            # Default factory creates RecordPublisher
            def default_mdns_publisher_factory(
                eff_inst_name: str,
                s_type: str,
                p: int,
                txt: Optional[Dict[bytes, bytes | None]],
            ) -> MdnsPublisher:
                return RecordPublisher(eff_inst_name, s_type, p, txt)

            self.__record_publisher = default_mdns_publisher_factory(
                effective_instance_name, base_service_type, port, txt_record
            )
        else:
            # Use provided factory
            self.__record_publisher = mdns_publisher_factory(
                effective_instance_name, base_service_type, port, txt_record
            )

    def _make_txt_record(self) -> dict[bytes, bytes | None]:
        """Creates TXT record dict for mDNS. Incl. pub timestamp & opt. name.

        Returns:
            Dict (bytes: bytes/None) for mDNS TXT record.
        """
        properties: dict[bytes, bytes | None] = {
            b"published_on": self.__get_current_date_time_bytes()
        }

        if self.__name is not None:
            properties[b"name"] = self.__name.encode("utf-8")

        return properties

    async def publish(self) -> None:
        """Publishes the service instance using mDNS.

        This method delegates to the underlying `RecordPublisher` to make the
        service visible on the network.
        """
        await self.__record_publisher.publish()

    async def close(self) -> None:
        """Closes the underlying record publisher if it supports closing."""
        if hasattr(self.__record_publisher, "close") and callable(
            getattr(self.__record_publisher, "close")
        ):
            try:
                # Assuming the close method of the publisher is async
                await self.__record_publisher.close()
            # pylint: disable=W0718 # Catch all exceptions to keep publish loop alive
            except Exception as e:
                _logger.error(
                    "Error while closing the record publisher: %s",
                    e,
                    exc_info=True,
                )
        else:
            # Long but readable debug message
            _logger.debug(
                "Record publisher does not have a close method or it's not callable."
            )

    def __get_current_date_time_bytes(self) -> bytes:
        """Gets current date/time as UTF-8 encoded string.

        Returns:
            Timestamp as bytes (e.g., "YYYY-MM-DD HH:MM:SS.ffffff").
        """
        now = datetime.datetime.now()
        # Format includes microseconds for precision.
        as_str = now.strftime("%F %T.%f")
        return as_str.encode("utf-8")
