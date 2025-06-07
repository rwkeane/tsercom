"""Implements the server-side specific `RuntimeDataHandler` logic.

This module defines `ServerRuntimeDataHandler`, a concrete implementation of
`RuntimeDataHandlerBase` tailored for Tsercom runtimes operating in a server role.
It is responsible for managing server-specific aspects of data flow and caller
registration. A key characteristic is that it typically acts as the source of
truth for time, providing a synchronized clock for its clients, often through
a `TimeSyncServer`.
"""

from typing import Generic, Optional, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.time_sync_server import TimeSyncServer

EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT")


class ServerRuntimeDataHandler(
    Generic[DataTypeT, EventTypeT],
    RuntimeDataHandlerBase[DataTypeT, EventTypeT],
):
    """Handles data, events, and caller management for server-side runtimes.

    This class extends `RuntimeDataHandlerBase` to provide functionality
    specific to server runtimes. It uses the `IdTracker` (from the base class)
    to manage associations for connecting clients. For time synchronization,
    it typically initializes a `TimeSyncServer` to provide a consistent time
    source for these clients. In testing mode, a `FakeSynchronizedClock` can
    be used instead. The server\'s own clock is considered authoritative.

    Type Args:
        DataTypeT: The generic type of data objects that this handler processes.
        EventTypeT: The generic type of event objects that this handler processes.
    """

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
        min_send_frequency_seconds: Optional[float] = None,
        *,
        is_testing: bool = False,
    ):
        """Initializes the ServerRuntimeDataHandler.

        Args:
            data_reader: The `RemoteDataReader` where incoming data, after being
                processed and annotated, will be sent.
            event_source: The primary `AsyncPoller` that sources events for
                all callers handled by this data handler.
            min_send_frequency_seconds: Optional minimum time interval, in seconds,
                for the per-caller event pollers created by the underlying
                `IdTracker`. Passed to `RuntimeDataHandlerBase`.
            is_testing: If True, a `FakeSynchronizedClock` is used as the time
                source. Otherwise, a `TimeSyncServer` is started to provide
                time synchronization to clients.
        """
        super().__init__(data_reader, event_source, min_send_frequency_seconds)

        self.__clock: SynchronizedClock
        self.__server: Optional[TimeSyncServer] = (
            None  # Store server if created
        )

        if is_testing:
            self.__clock = FakeSynchronizedClock()
        else:
            # In a real server scenario, TimeSyncServer provides the clock.
            self.__server = TimeSyncServer()
            self.__server.start_async()  # Assuming start_async is a non-blocking call that schedules startup
            self.__clock = self.__server.get_synchronized_clock()

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]:
        """Registers a new client caller.

        This implementation adds the client caller (identified by `caller_id`,
        `endpoint`, and `port`) to the internal `IdTracker`. It then uses the
        server\'s own synchronized clock (`self.__clock`) when creating the
        `EndpointDataProcessor` for this caller, as the server is the time
        authority.

        Args:
            caller_id: The `CallerIdentifier` assigned to the new client caller
                (typically assigned by the server or a higher-level manager).
            endpoint: The network endpoint (e.g., IP address) of the client caller.
            port: The port number of the client caller.

        Returns:
            An `EndpointDataProcessor` instance configured for communication
            with the registered client caller, using the server\'s authoritative clock.
        """
        self._id_tracker.add(caller_id, endpoint, port)
        # Server uses its own clock as the source of truth for clients.
        return self._create_data_processor(caller_id, self.__clock)

    async def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Handles unregistration of a client caller.

        In the current server-side implementation, this method is a no-op
        and always returns `False`. The rationale is that server-side IDs might
        be kept for potential re-connection or auditing purposes, and explicit
        removal is not performed by default upon a simple unregister call.
        Subclasses could override this for different behavior.

        Args:
            caller_id: The `CallerIdentifier` of the client caller to unregister.

        Returns:
            False, indicating the caller is not actively removed by this method.
        """
        # Server-side might not remove IDs immediately to allow for re-connections
        # or for other tracking purposes. Current behavior is no-op.
        # To fully remove, one might call:
        # return self._id_tracker.remove(caller_id)
        return False
