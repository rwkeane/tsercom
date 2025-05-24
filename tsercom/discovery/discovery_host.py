from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Generic, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.mdns.instance_listener import InstanceListener
from tsercom.discovery.service_info import ServiceInfo
from tsercom.threading.aio.aio_utils import run_on_event_loop

TServiceInfo = TypeVar("TServiceInfo", bound=ServiceInfo)


class DiscoveryHost(
    Generic[TServiceInfo],
    InstanceListener[TServiceInfo].Client,  # type: ignore
):
    """
    Helper object, to wrap CallerId management (for reconnection) with service
    discovery.
    """

    class Client(ABC):
        @abstractmethod
        async def _on_service_added(
            self, connection_info: TServiceInfo, caller_id: CallerIdentifier
        ) -> None:
            pass

    @overload
    def __init__(self, *, service_type: str):
        pass

    @overload
    def __init__(
        self,
        *,
        instance_listener_factory: Callable[
            [InstanceListener.Client], InstanceListener[TServiceInfo]
        ],
    ):
        pass

    def __init__(
        self,
        *,
        service_type: Optional[str] = None,
        instance_listener_factory: Optional[
            Callable[[InstanceListener.Client], InstanceListener[TServiceInfo]]
        ] = None,
    ) -> None:
        if not ((service_type is not None) ^ (instance_listener_factory is not None)):
            raise ValueError(
                "Exactly one of 'service_type' or 'instance_listener_factory' must be provided."
            )

        self.__service_type = service_type
        self.__instance_listener_factory = instance_listener_factory

        self.__discoverer: InstanceListener[TServiceInfo] | None = None
        self.__client: DiscoveryHost.Client | None = None

        self.__caller_id_map: Dict[str, CallerIdentifier] = {}

    def start_discovery(self, client: "DiscoveryHost.Client") -> None:
        """
        Starts discovery. Results are returned by a call to the |client|.
        """
        # client validation will be done in __start_discovery_impl
        run_on_event_loop(partial(self.__start_discovery_impl, client))

    async def __start_discovery_impl(
        self, client: "DiscoveryHost.Client"
    ) -> None:
        if client is None:
            raise ValueError("Client argument cannot be None for start_discovery.")
        # It's good practice to also check the type, though type hints help.
        # For this exercise, sticking to explicit None check as per original assert focus.
        # if not issubclass(type(client), DiscoveryHost.Client):
        #     raise TypeError(f"Client must be a subclass of DiscoveryHost.Client, got {type(client).__name__}.")

        if self.__discoverer is not None:
            raise RuntimeError("Discovery has already been started.")

        self.__client = client
        if self.__instance_listener_factory is not None:
            self.__discoverer = self.__instance_listener_factory(self)
        else:
            assert self.__service_type is not None
            self.__discoverer = InstanceListener[TServiceInfo](
                self, self.__service_type
            )

    async def _on_service_added(self, connection_info: TServiceInfo) -> None:
        # print("ENDPOINT FOUND") # Handled by logging subtask
        if self.__client is None:
            raise RuntimeError("Client not set; discovery may not have been started correctly.")
        caller_id: CallerIdentifier = None  # type: ignore
        if connection_info.mdns_name in self.__caller_id_map:
            caller_id = self.__caller_id_map[connection_info.mdns_name]
        else:
            caller_id = CallerIdentifier()
            self.__caller_id_map[connection_info.mdns_name] = caller_id

        await self.__client._on_service_added(connection_info, caller_id)
