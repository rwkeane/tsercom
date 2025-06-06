"""Manages dynamic service discovery and establishes connections to discovered services.

This module provides the `ServiceConnector` class, which acts as a bridge between
a `ServiceSource` (which discovers services) and a `ConnectionFactory` (which
establishes connections, typically gRPC channels). The `ServiceConnector` listens
for newly discovered services, attempts to connect to them, and notifies its
client upon successful connection. It also handles marking connections as failed
to allow for re-discovery and re-connection.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, Optional, Set, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_source import (
    ServiceInfoT as SourceServiceInfoT,
)
from tsercom.discovery.service_source import (
    ServiceSource,
)
from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.util.connection_factory import ConnectionFactory

ChannelTypeT = TypeVar("ChannelTypeT")


class ServiceConnector(
    Generic[SourceServiceInfoT, ChannelTypeT],
    ServiceSource.Client,
):
    """Monitors a `ServiceSource` and attempts to connect to discovered services.

    This class implements `ServiceSource.Client` to receive notifications about
    services being added. When a new service is discovered, `ServiceConnector`
    uses the provided `ConnectionFactory` to attempt to establish a connection
    (e.g., a gRPC channel) to it.

    If a connection is successfully established, the `ServiceConnector` notifies
    its own registered `Client` (an implementer of `ServiceConnector.Client`)
    via the `_on_channel_connected` callback, passing the service information,
    a `CallerIdentifier` for the service instance, and the established channel.

    It tracks active connections by `CallerIdentifier` to avoid duplicate
    connection attempts and allows marking specific clients as failed to enable
    re-discovery and connection. Operations are managed to run on a consistent
    asyncio event loop, captured during the first asynchronous operation.

    Type Args:
        ServiceInfoT: The type of service information object (subclass of
            `ServiceInfo`) that the `ServiceSource` provides.
        ChannelTypeT: The type of the communication channel object that the
            `ConnectionFactory` produces (e.g., `grpc.aio.Channel`).
    """

    class Client(ABC):
        """Interface for clients of `ServiceConnector`.

        Implementers of this interface are notified when the `ServiceConnector`
        successfully establishes a communication channel to a discovered service.
        """

        @abstractmethod
        async def _on_channel_connected(
            self,
            connection_info: SourceServiceInfoT,  # Use imported SourceServiceInfoT
            caller_id: CallerIdentifier,
            channel: ChannelTypeT,
        ) -> None:
            """Callback invoked when a channel to a discovered service is connected.

            Args:
                connection_info: The `ServiceInfoT` object containing details
                    about the discovered service (e.g., name, addresses, port).
                caller_id: The `CallerIdentifier` uniquely identifying the specific
                    instance of the discovered service.
                channel: The established communication channel (of type `ChannelTypeT`,
                    e.g., `grpc.aio.Channel`) to the service.
            """

    def __init__(
        self,
        client: "ServiceConnector.Client",
        connection_factory: ConnectionFactory[ChannelTypeT],
        service_source: ServiceSource[SourceServiceInfoT],
    ) -> None:
        """Initializes the ServiceConnector.

        Args:
            client: An instance that implements the `ServiceConnector.Client`
                interface. This client will receive notifications about
                successfully established channels.
            connection_factory: A `ConnectionFactory` instance responsible for
                creating communication channels (of type `ChannelTypeT`) to
                services based on their address and port.
            service_source: A `ServiceSource` instance that will provide
                information about discovered services (of type `SourceServiceInfoT`).
                The `ServiceConnector` registers itself as a client to this source.
        """
        self.__client: ServiceConnector.Client = client
        self.__service_source: ServiceSource[SourceServiceInfoT] = (
            service_source
        )
        self.__connection_factory: ConnectionFactory[ChannelTypeT] = (
            connection_factory
        )

        self.__callers: Set[CallerIdentifier] = set()  # Using typing.Set

        # The event loop is captured on the first relevant async operation,
        # typically in _on_service_added, to ensure subsequent operations
        # are scheduled on the same loop for consistency.
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

        super().__init__()  # Call ServiceSource.Client.__init__

    async def start(self) -> None:
        """Starts the service discovery process.

        This method initiates discovery by calling `start_discovery` on the
        configured `ServiceSource`, passing `self` as the client to receive
        callbacks when services are added or removed (though only `_on_service_added`
        is actively used by this implementation for connection logic).
        """
        await self.__service_source.start_discovery(self)

    async def mark_client_failed(self, caller_id: CallerIdentifier) -> None:
        """Marks a connected service instance (client from ServiceConnector\'s perspective) as failed.

        This removes the `caller_id` from the set of tracked active connections,
        allowing the `ServiceConnector` to attempt a new connection if the same
        service instance (with the same `caller_id`) is re-discovered by the
        `ServiceSource`.

        The operation is scheduled to run on the `ServiceConnector`\'s captured
        event loop to ensure thread-safety and proper async context.

        Args:
            caller_id: The `CallerIdentifier` of the service instance to be
                marked as failed.

        Raises:
            AssertionError: If `caller_id` was not previously tracked as an
                active connection (internal consistency check). This typically
                indicates a logic error or misuse.
            RuntimeError: If an event loop cannot be determined when attempting
                to schedule the operation (should not happen if `start` has been
                called and discovery is active).
        """
        if self.__event_loop is None:
            # Attempt to capture loop if not already. This might occur if mark_client_failed
            # is called before any service discovery callback has set the loop.
            logging.warning(
                "mark_client_failed called before event loop was captured by a discovery event."
            )
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                logging.error(
                    "Failed to get current event loop in mark_client_failed. Cannot proceed."
                )
                # Or raise RuntimeError if this is considered a critical failure path.
                return

        if not is_running_on_event_loop(self.__event_loop):
            # Schedule the implementation to run on the captured event loop.
            # The future returned by run_on_event_loop is not awaited here,
            # as mark_client_failed is fire-and-forget from the caller\'s perspective.
            run_on_event_loop(
                partial(self._mark_client_failed_impl, caller_id),
                self.__event_loop,
            )
            return

        # Already on the correct loop, execute directly.
        await self._mark_client_failed_impl(caller_id)

    async def _mark_client_failed_impl(
        self, caller_id: CallerIdentifier
    ) -> None:
        """Internal implementation for marking a client as failed.

        This method must be called on the `ServiceConnector`\'s event loop.
        It removes the `caller_id` from the set of active callers.

        Args:
            caller_id: The `CallerIdentifier` of the client to remove.
        """
        assert caller_id in self.__callers, (
            f"Attempted to mark unknown caller {caller_id} as failed. "
            f"Known callers: {self.__callers}"
        )
        self.__callers.remove(caller_id)
        logging.info(
            "Marked service with CallerIdentifier %s as failed. "
            "It can be re-discovered and re-connected.",
            caller_id,
        )

    async def _on_service_added(  # type: ignore[override]
        self,
        connection_info: SourceServiceInfoT,
        caller_id: CallerIdentifier,
    ) -> None:
        """Handles new service discovery events from the `ServiceSource`.

        This method is called by the `ServiceSource` when a new service instance
        is discovered. It captures the event loop on its first invocation.
        If the service (identified by `caller_id`) is not already tracked as an
        active connection, it attempts to establish a gRPC channel using the
        `ConnectionFactory`. If successful, it notifies its own client via
        `_on_channel_connected`.

        Args:
            connection_info: The `ServiceInfoT` object containing details about
                the newly discovered service.
            caller_id: The `CallerIdentifier` for the specific service instance.
        """
        if self.__event_loop is None:
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                logging.error(
                    "Failed to capture event loop in _on_service_added for service %s. "
                    "Cannot attempt connection.",
                    connection_info.name,
                )
                return
        elif not is_running_on_event_loop(self.__event_loop):
            # This indicates a potential issue, as callbacks from ServiceSource
            # should ideally be on a consistent loop or handled appropriately.
            # For now, assert to highlight this during development/testing.
            # In production, might re-schedule or log a critical warning.
            raise RuntimeError(
                "_on_service_added callback received on an unexpected event loop."
            )

        if caller_id in self.__callers:
            logging.debug(
                "Service with CallerIdentifier %s (Name: %s) already connected or connect attempt in progress. Skipping.",
                caller_id,
                connection_info.name,
            )
            return

        logging.info(
            "Service %s (CallerIdentifier: %s) discovered at %s:%s. Attempting connection.",
            connection_info.name,
            caller_id,
            connection_info.addresses,
            connection_info.port,
        )

        channel: Optional[ChannelTypeT] = (
            await self.__connection_factory.connect(
                connection_info.addresses, connection_info.port
            )
        )

        if channel is None:
            logging.warning(
                "Could not establish gRPC channel for service %s (CallerIdentifier: %s) at %s:%s.",
                connection_info.name,
                caller_id,
                connection_info.addresses,
                connection_info.port,
            )
            return

        logging.info(
            "Successfully established gRPC channel for service %s (CallerIdentifier: %s).",
            connection_info.name,
            caller_id,
        )

        self.__callers.add(caller_id)
        # pylint: disable=protected-access # Calling client's designated callback
        await self.__client._on_channel_connected(
            connection_info,
            caller_id,
            channel,
        )
