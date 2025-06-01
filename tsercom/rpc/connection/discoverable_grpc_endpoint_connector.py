from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, TypeVar, Optional
import asyncio
import typing

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.discovery.service_info import ServiceInfo
from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
import logging
import grpc

if typing.TYPE_CHECKING:
    from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory

# Generic type for service information, bound by the base ServiceInfo class.
TServiceInfo = TypeVar("TServiceInfo", bound=ServiceInfo)


class DiscoverableGrpcEndpointConnector(
    Generic[TServiceInfo], DiscoveryHost.Client  # Removed [TServiceInfo]
):
    """Connects to gRPC endpoints discovered via `DiscoveryHost`.

    This class acts as a client to `DiscoveryHost`. When a service is discovered,
    it attempts to establish a gRPC channel to that service using a provided
    `GrpcChannelFactory`. Successful connections (channel established) are then
    reported to its own registered `Client`. It also tracks active connections
    to avoid redundant connection attempts.
    """

    class Client(ABC):
        """Interface for clients of `DiscoverableGrpcEndpointConnector`.

        Implementers are notified when a gRPC channel to a discovered service
        has been successfully established.
        """

        @abstractmethod
        async def _on_channel_connected(
            self,
            connection_info: TServiceInfo,  # Detailed service information.
            caller_id: CallerIdentifier,  # Unique ID for the service instance.
            channel: "grpc.Channel",  # The established gRPC channel.
        ) -> None:
            """Callback invoked when a gRPC channel to a discovered service is connected.

            Args:
                connection_info: The `TServiceInfo` object for the discovered service.
                caller_id: The `CallerIdentifier` associated with the service instance.
                channel: The successfully established `grpc.Channel`.
            """
            pass

    def __init__(
        self,
        client: "DiscoverableGrpcEndpointConnector.Client",  # Removed [TServiceInfo]
        channel_factory: "GrpcChannelFactory",
        discovery_host: DiscoveryHost[TServiceInfo],
    ) -> None:
        """Initializes the DiscoverableGrpcEndpointConnector.

        Args:
            client: The client object that will receive notifications about
                    successfully connected channels.
            channel_factory: A `GrpcChannelFactory` used to create gRPC channels
                             to discovered services.
            discovery_host: The `DiscoveryHost` instance that will provide
                            discovered service information.
        """
        self.__client: DiscoverableGrpcEndpointConnector.Client = (
            client  # Removed [TServiceInfo]
        )
        self.__discovery_host: DiscoveryHost[TServiceInfo] = discovery_host
        self.__channel_factory: "GrpcChannelFactory" = channel_factory

        self.__callers: set[CallerIdentifier] = set[CallerIdentifier]()

        # Event loop captured during the first relevant async operation (_on_service_added).
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

        super().__init__()  # Calls __init__ of DiscoveryHost.Client

    async def start(self) -> None:
        """Starts the service discovery process.

        This initiates discovery by calling `start_discovery` on the configured
        `DiscoveryHost`. This instance (`self`) is passed as the client to
        receive `_on_service_added` callbacks from the `DiscoveryHost`.
        """
        await self.__discovery_host.start_discovery(self)

    async def mark_client_failed(self, caller_id: CallerIdentifier) -> None:
        """Marks a client associated with a `CallerIdentifier` as failed or unhealthy.

        This allows the connector to potentially re-establish a connection to this
        service if it's discovered again. The operation is performed on the
        event loop captured during initial operations.

        Args:
            caller_id: The `CallerIdentifier` of the client/service to mark as failed.

        Raises:
            AssertionError: If `caller_id` was not previously tracked.
        """
        if self.__event_loop is None:
            logging.warning(
                "mark_client_failed called before event loop was captured. This may indicate an issue if called before any service discovery."
            )
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                logging.error(
                    "Failed to get event loop in mark_client_failed."
                )
                return

        if not is_running_on_event_loop(self.__event_loop):
            run_on_event_loop(
                partial(self._mark_client_failed_impl, caller_id),
                self.__event_loop,
            )
            return

        await self._mark_client_failed_impl(caller_id)

    async def _mark_client_failed_impl(
        self, caller_id: CallerIdentifier
    ) -> None:
        # Assert that the caller_id was indeed being tracked.
        assert (
            caller_id in self.__callers
        ), f"Attempted to mark unknown caller_id {caller_id} as failed."
        # Remove the caller_id, allowing new connections if the service is re-discovered.
        self.__callers.remove(caller_id)
        logging.info(
            f"Marked client with caller_id {caller_id} as failed. It can now be re-discovered."
        )

    async def _on_service_added(
        self,
        connection_info: TServiceInfo,
        caller_id: CallerIdentifier,
    ) -> None:
        """Callback from `DiscoveryHost` when a new service is discovered.

        This method attempts to establish a gRPC channel to the discovered service.
        If successful, it notifies its own client via `_on_channel_connected`.
        It ensures operations run on a consistent event loop.

        Args:
            connection_info: The `TServiceInfo` for the discovered service.
            caller_id: The `CallerIdentifier` for the service instance.
        """
        # Capture or verify the event loop on the first call.
        if self.__event_loop is None:
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                logging.error(
                    "Failed to get event loop in _on_service_added. Cannot proceed with channel connection."
                )
                return
        else:
            # Ensure subsequent calls are on the same captured event loop.
            assert is_running_on_event_loop(
                self.__event_loop
            ), "Operations must run on the captured event loop."

        # Prevent duplicate connection attempts to the same service instance.
        if caller_id in self.__callers:
            logging.info(
                f"Service with Caller ID {caller_id} (Name: {connection_info.name}) already connected or connecting. Skipping."
            )
            return

        logging.info(
            f"Service added: {connection_info.name} (CallerID: {caller_id}). Attempting to establish gRPC channel."
        )

        # ChannelInfo was a typo, should be grpc.Channel from find_async_channel
        channel: Optional["grpc.Channel"] = (
            await self.__channel_factory.find_async_channel(
                connection_info.addresses, connection_info.port
            )
        )

        if channel is None:
            logging.warning(
                f"Could not establish gRPC channel for endpoint: {connection_info.name} at {connection_info.addresses}:{connection_info.port}."
            )
            return

        logging.info(
            f"Successfully established gRPC channel for: {connection_info.name} (CallerID: {caller_id})"
        )

        self.__callers.add(caller_id)
        # The client expects a grpc.Channel, not ChannelInfo wrapper here.
        await self.__client._on_channel_connected(
            connection_info, caller_id, channel
        )
