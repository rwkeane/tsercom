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
        ...

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
        ...

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
        ...

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
        num_modes_selected = sum(
            [
                bool(service_type),
                bool(instance_listener_factory),
                bool(mdns_listener_factory),
            ]
        )

        if num_modes_selected != 1:
            raise ValueError(
                "Exactly one of 'service_type', 'instance_listener_factory', "
                "or 'mdns_listener_factory' must be provided."
            )
        self.__instance_listener_factory: Callable[
            [InstanceListener.Client],
            InstanceListener[ServiceInfoT],
        ]
        if instance_listener_factory is not None:
            self.__instance_listener_factory = instance_listener_factory
        else:

            def instance_factory(
                client: InstanceListener.Client,
            ) -> InstanceListener[ServiceInfoT]:
                # If service_type was not provided during __init__ (e.g. using mdns_listener_factory mode),
                # provide a default service_type for the InstanceListener, as it requires a string.
                effective_service_type = (
                    service_type
                    if service_type is not None
                    else "_internal_default._tcp.local."
                )
                return InstanceListener[ServiceInfoT](
                    client,
                    effective_service_type,
                    mdns_listener_factory=mdns_listener_factory,
                )

            self.__instance_listener_factory = instance_factory

        self.__discoverer: Optional[InstanceListener[ServiceInfoT]] = None
        self.__client: Optional[ServiceSource.Client] = None
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
            raise ValueError("Client argument cannot be None for start_discovery.")

        if self.__discoverer is not None:
            raise RuntimeError("Discovery has already been started.")

        self.__client = client
        try:
            self.__discoverer = self.__instance_listener_factory(
                self
            )  # This might raise
            if self.__discoverer:
                await self.__discoverer.start()  # This might also raise
        except Exception as e:
            logging.error(
                "Failed to initialize or start discovery listener: %s",
                e,
                exc_info=True,
            )
            # If self.__discoverer was successfully created but start() failed,
            # it's good practice to try and clean it up if it has resources.
            if self.__discoverer:
                try:
                    # Assuming async_stop is idempotent and safe to call even if start didn't fully complete
                    await self.__discoverer.async_stop()
                except Exception as stop_e:
                    logging.error(
                        "Error trying to stop discoverer after start failed: %s",
                        stop_e,
                        exc_info=True,
                    )
            self.__discoverer = (
                None  # Ensure discoverer is None if any part of init/start fails
            )

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
            raise RuntimeError(
                "DiscoveryHost client not set; discovery may not have been started correctly."
            )

        service_mdns_name = connection_info.mdns_name

        caller_id: CallerIdentifier
        if service_mdns_name in self.__caller_id_map:
            caller_id = self.__caller_id_map[service_mdns_name]
        else:
            caller_id = CallerIdentifier.random()
            self.__caller_id_map[service_mdns_name] = caller_id

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
            if hasattr(self.__client, "_on_service_removed") and callable(
                getattr(self.__client, "_on_service_removed")
            ):

                await self.__client._on_service_removed(service_name, caller_id)
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

    async def stop_discovery(self) -> None:
        # Stops service discovery and cleans up resources.
        logging.info("Stopping DiscoveryHost...")
        if self.__discoverer is not None:
            if hasattr(self.__discoverer, "async_stop") and callable(
                getattr(self.__discoverer, "async_stop")
            ):
                await self.__discoverer.async_stop()
            elif hasattr(self.__discoverer, "stop") and callable(
                getattr(self.__discoverer, "stop")
            ):
                self.__discoverer.stop()
            self.__discoverer = None
        self.__client = None
        self.__caller_id_map.clear()
        logging.info("DiscoveryHost stopped.")
