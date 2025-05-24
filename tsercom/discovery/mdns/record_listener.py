from abc import ABC, abstractmethod
from typing import Dict, List
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
import logging


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
        if client is None:
            raise ValueError("Client argument cannot be None for RecordListener.")
        if not issubclass(type(client), RecordListener.Client):
            raise TypeError(
                f"Client must be a subclass of RecordListener.Client, got {type(client).__name__}."
            )

        if service_type is None:
            raise ValueError("service_type argument cannot be None for RecordListener.")
        if not isinstance(service_type, str):
            # Added for robustness, as service_type[0] would fail if not a string.
            raise TypeError(
                f"service_type must be a string, got {type(service_type).__name__}."
            )
        if not service_type.startswith("_"):
            raise ValueError(
                f"service_type must start with an underscore (e.g., '_my_service'), got '{service_type}'."
            )

        self.__client = client
        self.__expected_type = f"{service_type}._tcp.local."

        self.__mdns = Zeroconf()
        connection_str = f"{service_type}._tcp.local."
        logging.info(f"Scanning for: {connection_str}")
        self.__browser = ServiceBrowser(self.__mdns, connection_str, self)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        logging.info(f"Updated service of type {type_}, name {name}")
        if type_ != self.__expected_type:
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            logging.error(f"Invalid service records found for service {name} of type {type_} during update.")
            return

        if info.port is None:
            logging.error(f"No port found for service {name} of type {type_} during update.")
            return

        self.__client._on_service_added(
            info.name, info.port, info.addresses, info.properties
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        logging.info(f"Removed service of type {type_}, name {name}")
        pass

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        logging.info(f"Discovered service of type {type_}, name {name}")
        if type_ != self.__expected_type:
            return

        info = zc.get_service_info(type_, name)
        if info is None:
            logging.error(f"Invalid service records found for service {name} of type {type_} during add.")
            return

        if info.port is None:
            logging.error(f"No port found for service {name} of type {type_} during add.")
            return

        self.__client._on_service_added(
            info.name, info.port, info.addresses, info.properties
        )
