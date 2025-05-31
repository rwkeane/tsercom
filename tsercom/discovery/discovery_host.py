"""Manages discovery of network services using mDNS. It utilizes an InstanceListener and notifies a client upon discovering services, associating them with CallerIdentifiers."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Generic, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.mdns.instance_listener import InstanceListener
from tsercom.discovery.service_info import ServiceInfo
from tsercom.threading.aio.aio_utils import run_on_event_loop

# Generic type for service information, bound by the base ServiceInfo class.
TServiceInfo = TypeVar("TServiceInfo", bound=ServiceInfo)

# Removed module-level type aliases that caused issues with TServiceInfo binding.


class DiscoveryHost(
    Generic[TServiceInfo], InstanceListener.Client[TServiceInfo]
):
    """Manages service discovery using mDNS and CallerIdentifier association.

    This class listens for service instances of a specified type (or using a
    custom listener factory) and notifies a client upon discovery. It also
    manages `CallerIdentifier` instances for discovered services to facilitate
    reconnection and consistent identification. It acts as a client to an
    `InstanceListener` to receive raw service discovery events.
    """

    class Client(ABC, Generic[TServiceInfo]):
        """Interface for clients wishing to receive discovery notifications from `DiscoveryHost`.

        Implementers of this interface are notified when new services, relevant
        to the `DiscoveryHost`'s configuration, are discovered.
        """

        @abstractmethod
        async def _on_service_added(
            self, connection_info: TServiceInfo, caller_id: CallerIdentifier
        ) -> None:
            """Callback invoked when a new service instance is discovered and processed.

            Args:
                connection_info: Detailed information about the discovered service,
                                 of type `TServiceInfo`.
                caller_id: The unique `CallerIdentifier` assigned or retrieved
                           for this service instance.
            """
            pass

    @overload
    def __init__(self, *, service_type: str):
        """Initializes DiscoveryHost with a specific mDNS service type."""
        ...

    @overload
    def __init__(
        self,
        *,
        instance_listener_factory: Callable[  # Reverted to full type hint
            [InstanceListener[TServiceInfo].Client],
            InstanceListener[TServiceInfo],
        ],
    ):
        """Initializes DiscoveryHost with a factory for creating an InstanceListener."""
        ...

    def __init__(
        self,
        *,
        service_type: Optional[str] = None,
        instance_listener_factory: Optional[
            Callable[
                [InstanceListener.Client[TServiceInfo]],
                InstanceListener[TServiceInfo],
            ]
        ] = None,
    ) -> None:
        """Initializes the DiscoveryHost.

        This constructor is overloaded. It must be called with exactly one of the
        keyword arguments `service_type` or `instance_listener_factory`.

        Args:
            service_type: The mDNS service type string to listen for (e.g.,
                          "_my_service._tcp.local.").
            instance_listener_factory: A callable that creates an `InstanceListener`.
                                       This allows for more custom listener configurations,
                                       such as using different mDNS libraries or settings.
                                       The factory will be provided with `self` (this
                                       `DiscoveryHost` instance) as the client for the listener.

        Raises:
            ValueError: If neither or both `service_type` and
                        `instance_listener_factory` are provided.
        """
        # Ensure exclusive provision of either service_type or instance_listener_factory.
        if not (
            (service_type is not None)
            ^ (instance_listener_factory is not None)
        ):
            raise ValueError(
                "Exactly one of 'service_type' or 'instance_listener_factory' must be provided."
            )

        self.__service_type: Optional[str] = service_type
        self.__instance_listener_factory: Optional[
            Callable[
                [InstanceListener.Client[TServiceInfo]],
                InstanceListener[TServiceInfo],
            ]
        ] = instance_listener_factory

        # The actual mDNS instance listener; initialized in start_discovery_impl.
        self.__discoverer: Optional[InstanceListener[TServiceInfo]] = None
        self.__client: Optional[DiscoveryHost.Client[TServiceInfo]] = None

        # Maps mDNS instance names to their assigned CallerIdentifiers.
        self.__caller_id_map: Dict[str, CallerIdentifier] = {}

    def start_discovery(
        self, client: "DiscoveryHost.Client[TServiceInfo]"
    ) -> None:
        """Starts the service discovery process.

        This method schedules the actual discovery startup (`__start_discovery_impl`)
        on the event loop. Discovered services will be reported to the provided
        `client` object via its `_on_service_added` method.

        Args:
            client: An object implementing the `DiscoveryHost.Client` interface
                    that will receive notifications about discovered services.
        """
        # The actual startup logic, including client validation, is on the event loop.
        run_on_event_loop(partial(self.__start_discovery_impl, client))

    async def __start_discovery_impl(
        self,
        client: "DiscoveryHost.Client[TServiceInfo]",
    ) -> None:
        """Internal implementation for starting discovery; runs on the event loop.

        Initializes and starts the `InstanceListener` using either the provided
        `service_type` or `instance_listener_factory`.

        Args:
            client: The client object that will receive discovery notifications.

        Raises:
            ValueError: If the provided `client` is None.
            RuntimeError: If discovery has already been started.
        """
        if client is None:
            raise ValueError(
                "Client argument cannot be None for start_discovery."
            )

        if self.__discoverer is not None:
            raise RuntimeError("Discovery has already been started.")

        self.__client = client
        if self.__instance_listener_factory is not None:
            self.__discoverer = self.__instance_listener_factory(self)
        else:
            # This assertion is safe due to the __init__ constructor logic.
            assert (
                self.__service_type is not None
            ), "Service type must be set if no factory is provided."
            self.__discoverer = InstanceListener[TServiceInfo](
                self, self.__service_type
            )
        # TODO(developer/issue_id): Verify if self.__discoverer (InstanceListener)
        # requires an explicit start() method to be called after instantiation.
        # If so, it should be called here. For example:
        # if hasattr(self.__discoverer, "start") and callable(self.__discoverer.start):
        #     # await self.__discoverer.start() # If async
        #     # self.__discoverer.start() # If sync
        #     pass # Actual call depends on InstanceListener's API

    async def _on_service_added(self, connection_info: TServiceInfo) -> None:
        """Callback from `InstanceListener` when a new service instance is found.

        This method implements the `InstanceListener.Client` interface. It assigns
        a `CallerIdentifier` to the newly discovered service (or retrieves an existing
        one if the service was seen before) and then notifies this `DiscoveryHost`'s
        client via its `_on_service_added` method.

        Args:
            connection_info: Information about the discovered service instance,
                             of type `TServiceInfo`.

        Raises:
            RuntimeError: If the `DiscoveryHost`'s client is not set (e.g.,
                          if `start_discovery` was not called or failed).
        """
        if self.__client is None:
            # This indicates a programming error, discovery should be started with a client.
            raise RuntimeError(
                "DiscoveryHost client not set; discovery may not have been started correctly."
            )

        # Use the mDNS name as a key to uniquely identify service instances for CallerId mapping.
        service_mdns_name = connection_info.mdns_name

        caller_id: CallerIdentifier
        if service_mdns_name in self.__caller_id_map:
            caller_id = self.__caller_id_map[service_mdns_name]
        else:
            caller_id = CallerIdentifier.random()  # Use random() for new IDs
            self.__caller_id_map[service_mdns_name] = caller_id

        await self.__client._on_service_added(connection_info, caller_id)


# TODO(developer/issue_id): Implement a stop_discovery() method.
# This method should handle stopping the self.__discoverer (InstanceListener)
# if it has a stop() method, and potentially clear self.__client,
# self.__discoverer, and self.__caller_id_map to allow for restart or cleanup.
