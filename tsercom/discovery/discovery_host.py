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
        """Interface for clients wishing to receive discovery notifications.

        Clients of `DiscoveryHost` must implement this interface to be notified
        when new services are discovered.
        """

        @abstractmethod
        async def _on_service_added(
            self, connection_info: TServiceInfo, caller_id: CallerIdentifier
        ) -> None:
            """Called when a new service is discovered.

            Args:
                connection_info: Information about the discovered service.
                caller_id: The unique identifier assigned to this service instance.
            """
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
        """Initializes the DiscoveryHost.

        Args:
            service_type: The mDNS service type string to listen for (e.g., "_my_service._tcp.local.").
                          Exactly one of `service_type` or `instance_listener_factory` must be provided.
            instance_listener_factory: A callable that creates an `InstanceListener`.
                                       This is an alternative to providing `service_type` for more
                                       custom listener configurations. Exactly one of `service_type` or
                                       `instance_listener_factory` must be provided.

        Raises:
            ValueError: If neither or both `service_type` and `instance_listener_factory` are provided.
        """
        # Ensure that either a service type or a factory is provided, but not both.
        if not ((service_type is not None) ^ (instance_listener_factory is not None)):
            raise ValueError(
                "Exactly one of 'service_type' or 'instance_listener_factory' must be provided."
            )

        self.__service_type = service_type
        self.__instance_listener_factory = instance_listener_factory

        # Initialize internal state variables.
        self.__discoverer: InstanceListener[TServiceInfo] | None = None
        self.__client: DiscoveryHost.Client | None = None

        self.__caller_id_map: Dict[str, CallerIdentifier] = {}

    def start_discovery(self, client: "DiscoveryHost.Client") -> None:
        """Starts the service discovery process.

        Discovered services will be reported to the provided `client` object
        via its `_on_service_added` method. This method schedules the actual
        discovery startup on the event loop.

        Args:
            client: An object implementing the `DiscoveryHost.Client` interface
                    that will receive notifications about discovered services.
        """
        # client validation will be done in __start_discovery_impl
        run_on_event_loop(partial(self.__start_discovery_impl, client))

    async def __start_discovery_impl(
        self, client: "DiscoveryHost.Client"
    ) -> None:
        """Internal implementation for starting discovery; runs on the event loop.

        Args:
            client: The client object that will receive discovery notifications.

        Raises:
            ValueError: If the provided `client` is None.
            RuntimeError: If discovery has already been started.
        """
        # Validate the client argument.
        if client is None:
            raise ValueError("Client argument cannot be None for start_discovery.")
        # It's good practice to also check the type, though type hints help.
        # For this exercise, sticking to explicit None check as per original assert focus.
        # if not issubclass(type(client), DiscoveryHost.Client):
        #     raise TypeError(f"Client must be a subclass of DiscoveryHost.Client, got {type(client).__name__}.")

        # Ensure discovery isn't started multiple times.
        if self.__discoverer is not None:
            raise RuntimeError("Discovery has already been started.")

        # Store the client and initialize the InstanceListener.
        self.__client = client
        if self.__instance_listener_factory is not None:
            self.__discoverer = self.__instance_listener_factory(self)
        else:
            assert self.__service_type is not None
            self.__discoverer = InstanceListener[TServiceInfo](
                self, self.__service_type
            )

    async def _on_service_added(self, connection_info: TServiceInfo) -> None:
        """Handles a new service instance reported by the InstanceListener.

        This method is called by the underlying `InstanceListener` when a new
        service is found. It assigns a `CallerIdentifier` to the service
        and then notifies the `DiscoveryHost`'s client.

        Args:
            connection_info: Information about the discovered service instance.
        """
        # print("ENDPOINT FOUND") # Handled by logging subtask
        # Ensure that a client is registered to receive the notification.
        if self.__client is None:
            raise RuntimeError("Client not set; discovery may not have been started correctly.")
        # Obtain or create a CallerIdentifier for the discovered service.
        caller_id: CallerIdentifier
        if connection_info.mdns_name in self.__caller_id_map:
            caller_id = self.__caller_id_map[connection_info.mdns_name]
        else:
            caller_id = CallerIdentifier()
            self.__caller_id_map[connection_info.mdns_name] = caller_id

        # Notify the registered client about the new service.
        await self.__client._on_service_added(connection_info, caller_id)
