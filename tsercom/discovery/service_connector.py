from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, TypeVar, Optional, TYPE_CHECKING
import asyncio
import typing

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo
from tsercom.discovery.service_source import ServiceSource
from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
import logging

# Local application imports
from tsercom.util.connection_factory import ConnectionFactory  # Added import

if typing.TYPE_CHECKING:
    # Removed GrpcChannelFactory, ConnectionFactory is globally imported now for type hints.
    if TYPE_CHECKING:
        pass  # Keep for type hints if grpc.Channel is used as a concrete type argument
    pass

TServiceInfo = TypeVar("TServiceInfo", bound=ServiceInfo)
TChannelType = TypeVar("TChannelType")


class ServiceConnector(
    Generic[TServiceInfo, TChannelType], ServiceSource.Client
):
    """Connects to gRPC endpoints discovered via a `ServiceSource`.

    This class acts as a client to a `ServiceSource`. When a service is discovered,
    it attempts to establish a gRPC channel to that service using a provided
    `ConnectionFactory[grpc.Channel]`. Successful connections (channel established) are then
    reported to its own registered `Client`. It also tracks active connections
    to avoid redundant connection attempts.
    """

    class Client(ABC):
        """Interface for clients of `ServiceConnector`.

        Implementers are notified when a gRPC channel to a discovered service
        has been successfully established.
        """

        @abstractmethod
        async def _on_channel_connected(
            self,
            connection_info: TServiceInfo,
            caller_id: CallerIdentifier,
            channel: TChannelType,
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
        client: "ServiceConnector.Client",  # TODO(https://github.com/ClaudeTools/claude-tools-swe-prototype/issues/223): Should be ServiceConnector.Client[TChannelType]
        connection_factory: ConnectionFactory[TChannelType],
        service_source: ServiceSource[TServiceInfo],
    ) -> None:
        """Initializes the ServiceConnector.

        Args:
            client: The client object that will receive notifications about
                    successfully connected channels.
            connection_factory: A `ConnectionFactory[grpc.Channel]` used to create
                                gRPC channels to discovered services.
            service_source: The `ServiceSource` instance that will provide
                            discovered service information.
        """
        self.__client: ServiceConnector.Client = (
            client  # TODO(https://github.com/ClaudeTools/claude-tools-swe-prototype/issues/223): Should be ServiceConnector.Client[TChannelType]
        )
        self.__service_source: ServiceSource[TServiceInfo] = service_source
        self.__connection_factory: ConnectionFactory[TChannelType] = (
            connection_factory
        )

        self.__callers: set[CallerIdentifier] = set[CallerIdentifier]()

        # Event loop captured during the first relevant async operation (_on_service_added).
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

        super().__init__()

    async def start(self) -> None:
        """Starts the service discovery process.

        This initiates discovery by calling `start_discovery` on the configured
        `ServiceSource`. This instance (`self`) is passed as the client to
        receive `_on_service_added` callbacks from the `ServiceSource`.
        """
        await self.__service_source.start_discovery(self)

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
        assert (
            caller_id in self.__callers
        ), f"Attempted to mark unknown caller_id {caller_id} as failed."
        # Removing the caller_id allows new connections if the service is re-discovered.
        self.__callers.remove(caller_id)
        logging.info(
            f"Marked client with caller_id {caller_id} as failed. It can now be re-discovered."
        )

    async def _on_service_added(  # type: ignore[override]
        self,
        connection_info: TServiceInfo,
        caller_id: CallerIdentifier,
    ) -> None:
        """Callback from `ServiceSource` when a new service is discovered.

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
            if self.__event_loop is None:  # pragma: no cover
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

        channel: Optional[TChannelType] = (
            await self.__connection_factory.connect(
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
        # The client expects a T_ChannelType, not ChannelInfo wrapper here.
        await self.__client._on_channel_connected(
            connection_info,
            caller_id,
            channel,  # TODO: Fix this
        )
