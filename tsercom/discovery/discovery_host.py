"""Manages mDNS service discovery and notifies clients of services."""

import logging
from typing import Callable, Dict, Generic, Optional, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.mdns.instance_listener import (
    InstanceListener,
    MdnsListenerFactory,
)
from tsercom.discovery.service_info import ServiceInfoT
from tsercom.discovery.service_source import ServiceSource


# pylint: disable=R0903 # Implements ServiceSource and InstanceListener.Client
class DiscoveryHost(
    Generic[ServiceInfoT],
    ServiceSource[ServiceInfoT],
    InstanceListener.Client,
):
    """Manages mDNS service discovery and CallerIdentifier association.

    Listens for service instances (specified type or via custom factory)
    and notifies a client. Manages `CallerIdentifier` for discovered services
    for reconnection and ID. Acts as `InstanceListener.Client` for raw events.
    Implements `ServiceSource`.
    """

    @overload
    def __init__(self, *, service_type: str):
        """Initializes DiscoveryHost to listen for a specific mDNS service type.

        Args:
            service_type: The mDNS service type string to discover
                          (e.g., "_my_service._tcp.local.").
        """
        ...  # pylint: disable=W2301 # Ellipsis is part of overload definition

    @overload
    def __init__(
        self,
        *,
        instance_listener_factory: Callable[
            [InstanceListener.Client],
            InstanceListener[ServiceInfoT],
        ],
    ):
        """Initializes DiscoveryHost with a factory for a custom InstanceListener.

        This allows using a custom mDNS discovery mechanism or configuration.

        Args:
            instance_listener_factory: A callable that takes an
                `InstanceListener.Client` (which will be this DiscoveryHost instance)
                and returns an `InstanceListener[ServiceInfoT]` instance.
        """
        ...  # pylint: disable=W2301 # Ellipsis is part of overload definition

    @overload
    def __init__(
        self,
        *,
        mdns_listener_factory: MdnsListenerFactory,
    ):
        """Initializes DiscoveryHost with a factory for a custom InstanceListener.

        This allows using a custom mDNS discovery mechanism or configuration.

        Args:
            mdns_listener_factory: A callable to create a new MdnsListener.
        """
        ...  # pylint: disable=W2301 # Ellipsis is part of overload definition

    def __init__(
        self,
        *,
        service_type: Optional[str] = None,
        instance_listener_factory: Optional[
            Callable[
                [InstanceListener.Client],
                InstanceListener[ServiceInfoT],
            ]
        ] = None,
        mdns_listener_factory: Optional[MdnsListenerFactory] = None,
    ) -> None:
        """Initializes DiscoveryHost. Overloaded: use one keyword arg.

        Args:
            service_type: mDNS service type (e.g., "_my_service._tcp.local.").
            instance_listener_factory: Callable creating `InstanceListener`.
                Allows custom listener configs (e.g., different mDNS libs).
                Factory gets `self` as client for the listener.

        Raises:
            ValueError: If neither or both args provided.
        """
        # Ensure exclusive provision of service_type or factory
        if (service_type is None) == (instance_listener_factory is None):
            # Long error message
            raise ValueError(
                "Exactly one of 'service_type' or 'instance_listener_factory' must be provided."
            )
        if (instance_listener_factory is not None) and (
            mdns_listener_factory is not None
        ):
            # Long error message
            raise ValueError(
                "'instance_listener_factory' and 'mdns_listener_factory' cannot both be set."
            )

        self.__instance_listener_factory: Callable[
            [InstanceListener.Client],
            InstanceListener[ServiceInfoT],
        ]
        if instance_listener_factory is not None:
            self.__instance_listener_factory = instance_listener_factory
        else:
            assert service_type is not None

            def instance_factory(
                client: InstanceListener.Client,
            ) -> InstanceListener[ServiceInfoT]:
                return InstanceListener[ServiceInfoT](
                    client,
                    service_type,
                    mdns_listener_factory=mdns_listener_factory,
                )

            self.__instance_listener_factory = instance_factory

        # mDNS instance listener; initialized in __start_discovery_impl.
        self.__discoverer: Optional[InstanceListener[ServiceInfoT]] = None
        self.__client: Optional[ServiceSource.Client] = None

        # Maps mDNS instance names to CallerIdentifiers.
        self.__caller_id_map: Dict[str, CallerIdentifier] = {}

    async def start_discovery(self, client: ServiceSource.Client) -> None:
        """Starts service discovery.

        Calls async `__start_discovery_impl`. Discovered services reported
        to `client` via `_on_service_added`. Fulfills `ServiceSource` interface.

        Args:
            client: `ServiceSource.Client` for discovery notifications.
        """
        await self.__start_discovery_impl(client)

    async def __start_discovery_impl(
        self,
        client: ServiceSource.Client,
    ) -> None:
        """Internal implementation for starting discovery.

        Initializes and starts `InstanceListener` using `service_type` or
        `instance_listener_factory`.

        Args:
            client: Client for discovery notifications.

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
        try:
            self.__discoverer = self.__instance_listener_factory(self)
        except Exception as e:
            logging.error(f"Failed to initialize discovery listener: {e}")
            self.__discoverer = None
        # TODO(developer/issue_id): Verify if self.__discoverer (InstanceListener)
        # requires an explicit start() method to be called after instantiation.
        # If so, it should be called here. For example:
        # if hasattr(self.__discoverer, "start") and callable(self.__discoverer.start):
        #     # await self.__discoverer.start() # If async
        #     # self.__discoverer.start() # If sync
        #     pass  # Actual call depends on InstanceListener's API

    async def _on_service_added(self, connection_info: ServiceInfoT) -> None:  # type: ignore[override]
        """Callback from `InstanceListener` when a new service instance is found.

        Implements `InstanceListener.Client`. Assigns/retrieves `CallerIdentifier`
        for the new service. Notifies `DiscoveryHost`'s client
        (a `ServiceSource.Client`) via `_on_service_added`.

        Args:
            connection_info: Info about discovered service (`ServiceInfoT`).

        Raises:
            RuntimeError: If `DiscoveryHost` client not set (start_discovery issue).
        """
        if self.__client is None:
            # Programming error: discovery should have been started with a client.
            # Long error message
            raise RuntimeError(
                "DiscoveryHost client not set; discovery may not have been started correctly."
            )

        # Use mDNS instance name as key for CallerId mapping.
        service_mdns_name = connection_info.mdns_name

        caller_id: CallerIdentifier
        if service_mdns_name in self.__caller_id_map:
            caller_id = self.__caller_id_map[service_mdns_name]
        else:
            caller_id = CallerIdentifier.random()
            self.__caller_id_map[service_mdns_name] = caller_id

        # pylint: disable=protected-access # Internal callback to client
        # pylint: disable=W0212 # Calling listener's notification method
        await self.__client._on_service_added(connection_info, caller_id)

    async def _on_service_removed(self, service_name: str) -> None:
        """Handles a service being removed.

        Notifies the ServiceSource.Client that a service, identified by its
        mDNS name, is no longer available. It also removes the service's
        CallerIdentifier from internal tracking. This method is part of the
        InstanceListener.Client interface.

        Args:
            service_name: The mDNS instance name of the service that was removed.
        """
        logging.info("Service removed by DiscoveryHost: %s", service_name)
        caller_id = self.__caller_id_map.pop(service_name, None)

        if self.__client is None:
            logging.warning(
                "No client configured in DiscoveryHost; cannot notify of service removal: %s",
                service_name,
            )
            return

        if caller_id:
            # Check if the client has the _on_service_removed method
            if hasattr(self.__client, "_on_service_removed") and callable(
                getattr(self.__client, "_on_service_removed")
            ):
                # pylint: disable=protected-access, W0212
                await self.__client._on_service_removed(
                    service_name, caller_id
                )
            else:
                logging.warning(
                    "Client %s does not implement _on_service_removed, cannot notify for %s.",
                    type(self.__client).__name__,
                    service_name,
                )
        else:
            logging.warning(
                "Service %s removed, but was not found in DiscoveryHost's caller_id_map.",
                service_name,
            )


# TODO(developer/issue_id): Implement a stop_discovery() method.
# This method should handle stopping self.__discoverer (InstanceListener)
# if it has stop(), and potentially clear client, discoverer, and map.
