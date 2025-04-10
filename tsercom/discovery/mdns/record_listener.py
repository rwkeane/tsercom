from abc import ABC, abstractmethod
from typing import Dict, List
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf


class RecordListener(ServiceListener):
    """
    Low-level listener class to interface with mDNS, for discovering services.
    """

    class Client(ABC):
        @abstractmethod
        def _on_service_added(
            self,
            name: str,
            port: int,
            addesses: List[bytes],
            txt_record: Dict[bytes, bytes | None],
        ) -> None:
            raise NotImplementedError("This method must be implemented!")

    def __init__(self, client: Client, service_type: str):
        assert client is not None and issubclass(
            type(client), RecordListener.Client
        ), client
        # fmt: off
        assert (
            service_type is not None and service_type[0] == "_"
        ), service_type
        # fmt: on

        self.__client = client
        self.__expected_type = f"{service_type}._tcp.local."

        self.__mdns = Zeroconf()
        connection_str = f"{service_type}._tcp.local."
        print("Scanning for:", connection_str)
        self.__browser = ServiceBrowser(self.__mdns, connection_str, self)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print("Updated service of type ", type_, ", name ", name)
        if type_ != self.__expected_type:
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            print("ERROR: Invalid service records found!")
            return

        if info.port is None:
            print("ERROR: No port found!")
            return

        self.__client._on_service_added(
            info.name, info.port, info.addresses, info.properties
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print("Removed service of type ", type_, ", name ", name)
        pass

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print("Discovered service of type ", type_, ", name ", name)
        if type_ != self.__expected_type:
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            print("ERROR: Invalid service records found!")
            return

        if info.port is None:
            print("ERROR: No port found!")
            return

        self.__client._on_service_added(
            info.name, info.port, info.addresses, info.properties
        )
