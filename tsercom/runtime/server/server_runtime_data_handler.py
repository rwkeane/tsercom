"""Implements the server-side runtime data handling logic.

This module defines `ServerRuntimeDataHandler` which is responsible for
managing data flow, caller registration, and time synchronization aspects
for Tsercom runtimes operating in a server role. It assigns CallerIdentifiers
and uses its local clock as the source of truth for time synchronization.
"""

from typing import Generic, TypeVar
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.time_sync_server import TimeSyncServer


EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class ServerRuntimeDataHandler(
    Generic[DataTypeT, EventTypeT],
    RuntimeDataHandlerBase[DataTypeT, EventTypeT],
):
    """Handles data, events, and caller management for server runtimes.

    It utilizes an `IdTracker` to manage caller ID and endpoint associations.
    For time synchronization, it either uses a `TimeSyncServer` or a
    `FakeSynchronizedClock` (in testing mode) to provide a consistent time
    source for connected clients.
    """

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
        *,
        is_testing: bool = False,
    ):
        """Initializes the ServerRuntimeDataHandler.

        Args:
            data_reader: The reader for incoming data instances.
            event_source: The poller for incoming event instances.
            is_testing: If True, enables testing-specific behaviors, notably
                        using `FakeSynchronizedClock` not `TimeSyncServer`.
        """
        super().__init__(data_reader, event_source)

        self.__id_tracker = IdTracker()
        self.__clock: SynchronizedClock

        if is_testing:
            self.__clock = FakeSynchronizedClock()
            return

        self.__server = TimeSyncServer()
        self.__server.start_async()

        self.__clock = self.__server.get_synchronized_clock()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT]:
        """Registers a new caller, returning a data processor.

        Adds caller to ID tracker. Server's synchronized clock is used
        for the data processor.

        Args:
            caller_id: New caller's `CallerIdentifier` (usually server-assigned).
            endpoint: The network endpoint (e.g., IP address) of the caller.
            port: The port number of the caller.

        Returns:
            An `EndpointDataProcessor` for this caller (uses server clock).
        """
        self.__id_tracker.add(caller_id, endpoint, port)
        return self._create_data_processor(caller_id, self.__clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Handles unregistration of a caller.

        Currently a no-op in server impl; IDs kept for re-connection.
        Returns False as caller is not actively removed.

        Args:
            caller_id: The `CallerIdentifier` of the caller to unregister.
        """
        return False

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Tries to get CallerIdentifier for a given endpoint and port.

        Args:
            endpoint: The network endpoint of the caller.
            port: The port number of the caller.

        Returns:
            The `CallerIdentifier` if found, otherwise `None`.
        """
        return self.__id_tracker.try_get(endpoint, port)
