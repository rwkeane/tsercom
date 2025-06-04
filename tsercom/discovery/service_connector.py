# pylint: disable=C0301
"""Manages service discovery and connection establishment."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, TypeVar, Optional
import asyncio
import logging  # Moved up

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo
from tsercom.discovery.service_source import ServiceSource
from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)

# Local application imports
from tsercom.util.connection_factory import ConnectionFactory


ServiceInfoT = TypeVar("ServiceInfoT", bound=ServiceInfo)
ChannelTypeT = TypeVar("ChannelTypeT")


class ServiceConnector(
    Generic[ServiceInfoT, ChannelTypeT], ServiceSource.Client
):
    """Connects to gRPC endpoints from a `ServiceSource`.

    Acts as a `ServiceSource` client. On discovery, attempts to establish
    a gRPC channel using a `ConnectionFactory`. Successful connections are
    reported to its own `Client`. Tracks active connections.
    """

    # pylint: disable=R0903 # Abstract listener interface
    class Client(ABC):
        """Interface for `ServiceConnector` clients.

        Notified when a gRPC channel to a discovered service is established.
        """

        @abstractmethod
        async def _on_channel_connected(
            self,
            connection_info: ServiceInfoT,
            caller_id: CallerIdentifier,
            channel: ChannelTypeT,
        ) -> None:
            """Callback when a gRPC channel to a service is connected.

            Args:
                connection_info: `ServiceInfoT` for the discovered service.
                caller_id: `CallerIdentifier` for the service instance.
                channel: The established `grpc.Channel`.
            """
            # pass # W0107 (unnecessary-pass) will be fixed by removing this

    def __init__(  # pylint: disable=C0301
        self,
        client: "ServiceConnector.Client",  # TODO(https://github.com/ClaudeTools/claude-tools-swe-prototype/issues/223): Should be ServiceConnector.Client[ChannelTypeT]
        connection_factory: ConnectionFactory[ChannelTypeT],
        service_source: ServiceSource[ServiceInfoT],
    ) -> None:
        """Initializes the ServiceConnector.

        Args:
            client: Client to receive notifications of connected channels.
            connection_factory: `ConnectionFactory` to create gRPC channels.
            service_source: `ServiceSource` for discovered service info.
        """
        self.__client: ServiceConnector.Client = (
            client  # pylint: disable=W0511 # TODO: fix type hint
        )
        self.__service_source: ServiceSource[ServiceInfoT] = service_source
        self.__connection_factory: ConnectionFactory[ChannelTypeT] = (
            connection_factory
        )

        self.__callers: set[CallerIdentifier] = set()

        # Event loop captured during first relevant async op (_on_service_added).
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

        super().__init__()

    async def start(self) -> None:
        """Starts service discovery.

        Calls `start_discovery` on `ServiceSource`, passing `self` as client
        to receive `_on_service_added` callbacks.
        """
        await self.__service_source.start_discovery(self)

    async def mark_client_failed(self, caller_id: CallerIdentifier) -> None:
        """Marks a client with `CallerIdentifier` as failed/unhealthy.

        Allows re-establishing connection if service is re-discovered.
        Operation is performed on the captured event loop.

        Args:
            caller_id: `CallerIdentifier` of the client/service to mark failed.

        Raises:
            AssertionError: If `caller_id` was not previously tracked.
        """
        if self.__event_loop is None:
            logging.warning(
                "mark_client_failed called before event loop capture. Potential issue."  # pylint: disable=C0301
            )
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                logging.error("Failed to get event loop in mark_client_failed.")
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
        assert caller_id in self.__callers, f"Unknown caller {caller_id}."
        # Removing caller_id allows new connections if service re-discovered.
        self.__callers.remove(caller_id)
        logging.info(
            "Marked client with caller_id %s as failed. Can be re-discovered.",
            caller_id,
        )

    async def _on_service_added(  # type: ignore[override]
        self,
        connection_info: ServiceInfoT,
        caller_id: CallerIdentifier,
    ) -> None:
        """Callback from `ServiceSource` when a new service is discovered.

        Attempts to establish gRPC channel. If successful, notifies client via
        `_on_channel_connected`. Ensures operations run on consistent event loop.

        Args:
            connection_info: `ServiceInfoT` for the discovered service.
            caller_id: `CallerIdentifier` for service.
        """
        # Capture or verify event loop on first call.
        if self.__event_loop is None:
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:  # pragma: no cover
                # pylint: disable=C0301 # Long error message
                logging.error(
                    "Failed to get event loop in _on_service_added. Cannot proceed."
                )
                return
        else:
            # Ensure subsequent calls are on the same captured event loop.
            assert is_running_on_event_loop(  # pylint: disable=C0301
                self.__event_loop
            ), "Operations must run on the captured event loop."

        # Prevent duplicate connection attempts to same service instance.
        if caller_id in self.__callers:
            logging.info(
                "Service with Caller ID %s (Name: %s) already connected. Skipping.",
                caller_id,
                connection_info.name,
            )
            return

        logging.info(
            "Service added: %s (CallerID: %s). Attempting gRPC channel.",
            connection_info.name,
            caller_id,
        )

        channel: Optional[ChannelTypeT] = (
            await self.__connection_factory.connect(
                connection_info.addresses, connection_info.port
            )
        )

        if channel is None:
            # pylint: disable=C0301 # Long warning message
            logging.warning(
                "Could not establish gRPC channel for endpoint: %s at %s:%s.",
                connection_info.name,
                connection_info.addresses,
                connection_info.port,
            )
            return

        logging.info(
            "Successfully established gRPC channel for: %s (CallerID: %s)",
            connection_info.name,
            caller_id,
        )

        self.__callers.add(caller_id)
        # The client expects a ChannelTypeT, not ChannelInfo wrapper here.
        # pylint: disable=W0212 # Calling listener's notification method
        await self.__client._on_channel_connected(
            connection_info,
            caller_id,
            channel,  # pylint: disable=W0511 # TODO: Fix this
        )
