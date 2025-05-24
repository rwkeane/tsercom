import datetime
from uuid import getnode as get_mac

from tsercom.discovery.mdns.record_publisher import RecordPublisher


class InstancePublisher:
    """
    Publishes an instance of the specified type.
    """

    def __init__(
        self,
        port: int,
        service_type: str,
        readable_name: str | None = None,
        instance_name: str | None = None,
    ):
        if port is None: # Though type hint implies non-optional, being explicit
            raise ValueError("port argument cannot be None for InstancePublisher.")
        if not isinstance(port, int):
            raise TypeError(f"port must be an integer, got {type(port).__name__}.")

        if service_type is None: # Though type hint implies non-optional, being explicit
            raise ValueError("service_type argument cannot be None for InstancePublisher.")
        if not isinstance(service_type, str):
            raise TypeError(f"service_type must be a string, got {type(service_type).__name__}.")

        if readable_name is not None and not isinstance(readable_name, str):
            raise TypeError(f"readable_name must be a string or None, got {type(readable_name).__name__}.")

        if instance_name is not None and not isinstance(instance_name, str):
            raise TypeError(f"instance_name must be a string or None, got {type(instance_name).__name__}.")

        self.__name = readable_name
        if instance_name is None:
            mac_address = get_mac()
            instance_name = f"{port}{mac_address}"
            if len(instance_name) > 15:
                instance_name = instance_name[:15]

        txt_record = self._make_txt_record()
        if txt_record is None: # Should not happen based on _make_txt_record logic
            raise RuntimeError("_make_txt_record failed to produce a TXT record.")
        self.__record_publisher = RecordPublisher(
            instance_name, service_type, port, txt_record
        )

    def _make_txt_record(self) -> dict[bytes, bytes | None]:
        properties: dict[bytes, bytes | None] = {
            "published_on".encode(
                "utf-8"
            ): self.__get_current_date_time_bytes()
        }

        if self.__name is not None:
            properties["name".encode("utf-8")] = self.__name.encode("utf-8")

        return properties

    def publish(self) -> None:
        self.__record_publisher.publish()

    def __get_current_date_time_bytes(self) -> bytes:
        now = datetime.datetime.now()
        as_str = now.strftime("%F %T.%f")
        return as_str.encode("utf-8")
