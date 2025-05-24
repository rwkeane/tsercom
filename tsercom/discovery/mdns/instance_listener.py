from abc import ABC, abstractmethod
from functools import partial
import socket
from typing import Dict, Generic, List, TypeVar
import logging

from tsercom.discovery.service_info import ServiceInfo
from tsercom.discovery.mdns.record_listener import RecordListener
from tsercom.threading.aio.aio_utils import run_on_event_loop

TServiceInfo = TypeVar("TServiceInfo", bound=ServiceInfo)


class InstanceListener(Generic[TServiceInfo], RecordListener.Client):
    """
    Searches for instances of the specified type.
    """

    class Client(ABC):
        @abstractmethod
        async def _on_service_added(
            self, connection_info: TServiceInfo
        ) -> None:
            raise NotImplementedError("This method must be implemented!")

    def __init__(self, client: "InstanceListener.Client", service_type: str):
        if client is None:
            raise ValueError("Client argument cannot be None for InstanceListener.")
        if not issubclass(type(client), InstanceListener.Client):
            raise TypeError(
                f"Client must be a subclass of InstanceListener.Client, got {type(client).__name__}."
            )
        if not isinstance(service_type, str):
            raise TypeError(
                f"service_type must be a string, got {type(service_type).__name__}."
            )

        self.__client = client

        self.__listener = RecordListener(self, service_type)

    def __populate_service_info(
        self,
        record_name: str,
        port: int,
        addresses: List[bytes],
        txt_record: Dict[bytes, bytes | None],
    ) -> ServiceInfo | None:
        if len(addresses) == 0:
            logging.warning(f"No addresses available for service at port {port}. Cannot populate ServiceInfo.")
            return None

        addresses_out = []
        for i in range(len(addresses)):
            try:
                address = socket.inet_ntoa(addresses[i])
                addresses_out.append(address)
            except Exception:
                continue

        if len(addresses_out) == 0:
            # Log the first binary address if available, otherwise just port.
            # addresses[0] is bytes, so it might not be directly human-readable without further processing.
            # For logging, it's better to indicate the failure clearly.
            first_address_info = f"first address (binary): {addresses[0]}" if addresses else "unknown address"
            logging.warning(
                f"Failed to convert any binary addresses to string format for service at {first_address_info}, port {port}."
            )
            return None

        name_encoded = "name".encode("utf-8")
        if name_encoded in txt_record:  # b'str' is byte casted 'str'
            readable_name: bytes = txt_record[name_encoded]  # type: ignore
            readable_name = readable_name.decode("utf-8")  # type: ignore
        else:
            readable_name = record_name  # type: ignore

        return ServiceInfo(readable_name, port, addresses_out, record_name)  # type: ignore

    def _convert_service_info(
        self, service_info: ServiceInfo, txt_record: Dict[bytes, bytes | None]
    ) -> TServiceInfo:
        return service_info  # type: ignore

    def _on_service_added(
        self,
        record_name: str,
        port: int,
        addresses: List[bytes],
        txt_record: Dict[bytes, bytes | None],
    ) -> None:
        service_info = self.__populate_service_info(
            record_name, port, addresses, txt_record
        )
        if service_info is None:
            return
        service_info = self._convert_service_info(service_info, txt_record)
        run_on_event_loop(
            partial(self.__client._on_service_added, service_info)
        )
