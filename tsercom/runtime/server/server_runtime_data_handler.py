"""Implements the server-side runtime data handling logic.

This module defines `ServerRuntimeDataHandler` which is responsible for
managing data flow, caller registration, and time synchronization aspects
for Tsercom runtimes operating in a server role. It assigns CallerIdentifiers
and uses its local clock as the source of truth for time synchronization.
"""

from typing import Generic, TypeVar
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData # Import ExposedData
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
from tsercom.timesync.server.time_sync_server import TimeSyncServer


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType", bound=ExposedData) # Constrain TDataType


class ServerRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    """Handles data, events, and caller management for server runtimes.

    It utilizes an `IdTracker` to manage caller ID and endpoint associations.
    For time synchronization, it either uses a `TimeSyncServer` or a
    `FakeSynchronizedClock` (in testing mode) to provide a consistent time
    source for connected clients.
    """

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
        *,
        is_testing: bool = False,
    ):
        """Initializes the ServerRuntimeDataHandler.

        Args:
            data_reader: The reader for incoming data instances.
            event_source: The poller for incoming event instances.
            is_testing: If True, enables testing-specific behaviors, notably
                        using `FakeSynchronizedClock` instead of `TimeSyncServer`.
        """
        super().__init__(data_reader, event_source)

        self.__id_tracker = IdTracker()
        self.__clock: SynchronizedClock # Explicitly annotate __clock

        if is_testing:
            self.__clock = FakeSynchronizedClock()
            return

        self.__server = TimeSyncServer()
        self.__server.start_async()

        self.__clock = self.__server.get_synchronized_clock()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[TDataType]:
        """Registers a new caller and its endpoint, returning a data processor.

        Adds the caller to the ID tracker. The server's synchronized clock
        is used for the data processor.

        Args:
            caller_id: The `CallerIdentifier` of the new caller (usually assigned by the server).
            endpoint: The network endpoint (e.g., IP address) of the caller.
            port: The port number of the caller.

        Returns:
            An `EndpointDataProcessor` configured for this caller using the server's clock.
        """
        self.__id_tracker.add(caller_id, endpoint, port)
        return self._create_data_processor(caller_id, self.__clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool: # Changed return type to bool
        """Handles unregistration of a caller.

        In this server implementation, this method is currently a no-op.
        CallerIDs are kept to allow re-establishment of connections.
        Returning False as the caller is not actively removed.

        Args:
            caller_id: The `CallerIdentifier` of the caller to unregister.
        """
        # Keep all CallerID instances around, so a connection can be
        # re-established with the same id (if reconnection possible).
        return False # Return bool

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Tries to retrieve the CallerIdentifier for a given endpoint and port.

        Args:
            endpoint: The network endpoint of the caller.
            port: The port number of the caller.

        Returns:
            The `CallerIdentifier` if found, otherwise `None`.
        """
        return self.__id_tracker.try_get(endpoint, port)
