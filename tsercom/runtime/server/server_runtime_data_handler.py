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
from tsercom.runtime.caller_processor_registry import (
    CallerProcessorRegistry,
)  # Added
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
from typing import cast # Added for factory return


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType", bound=ExposedData)


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
        # id_tracker: IdTracker, # Removed from parameters
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
        self.__id_tracker = (
            IdTracker()
        )  # Changed to use standard name mangling
        self.__clock: (
            SynchronizedClock  # Changed to use standard name mangling
        )

        if is_testing:
            self.__clock = FakeSynchronizedClock()
        else:
            # TODO(b/265342012): TimeSyncServer probably shouldn't be stored as an attribute
            # if it's only used here for clock creation. Consider refactoring.
            server = TimeSyncServer()
            server.start_async()
            self.__clock = server.get_synchronized_clock()

        # Define processor_factory for CallerProcessorRegistry
        def processor_factory(
            caller_id_for_factory: CallerIdentifier,
        ) -> AsyncPoller[SerializableAnnotatedInstance[TEventType]]:
            # 'self' refers to ServerRuntimeDataHandler instance
            concrete_processor = (
                self._create_data_processor(  # from RuntimeDataHandlerBase
                    caller_id_for_factory, self.__clock  # Use self.__clock
                )
            )
            # The factory must return the AsyncPoller that will be pushed to.
            # Access the mangled name of __internal_poller from __ConcreteDataProcessor
            # using getattr to bypass potential Mypy direct access issues with mangled names.
            internal_poller = getattr(
                concrete_processor,
                "_RuntimeDataHandlerBase__ConcreteDataProcessor__internal_poller",
            )
            return cast(
                AsyncPoller[SerializableAnnotatedInstance[TEventType]],
                internal_poller,
            )

        processor_registry = CallerProcessorRegistry(
            processor_factory=processor_factory
        )

        super().__init__(
            data_reader,
            event_source,
            id_tracker=self.__id_tracker,  # Pass its own IdTracker for address mapping
            processor_registry=processor_registry,
        )

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[TDataType, TEventType]:
        """Registers a new caller and its endpoint.

        Adds the caller's address to its privately managed ID tracker for address mapping.
        The actual processor (AsyncPoller) will be created by the CallerProcessorRegistry
        when data for this caller_id is first routed. This method returns the
        EndpointDataProcessor instance (_RuntimeDataHandlerBase__DataProcessorImpl).

        Args:
            caller_id: The `CallerIdentifier` of the new caller.
            endpoint: The network endpoint (e.g., IP address) of the caller.
            port: The port number of the caller.

        Returns:
            An `EndpointDataProcessor` (_RuntimeDataHandlerBase__DataProcessorImpl)
            configured for this caller.
        """
        self.__id_tracker.add(
            caller_id, endpoint, port
        )  # Use self.__id_tracker

        # Return the __ConcreteDataProcessor instance.
        # The CallerProcessorRegistry's factory is responsible for calling this
        # again (or a similar mechanism) to get the actual poller when needed for routing.
        # For now, we return what _create_data_processor gives us, which is the
        # _RuntimeDataHandlerBase__ConcreteDataProcessor instance (corrected name).
        return self._create_data_processor(
            caller_id, self.__clock
        )  # Use self.__clock

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Handles unregistration of a caller.

        In this server implementation, this method is currently a no-op.
        CallerIDs are kept to allow re-establishment of connections.
        Returning False as the caller is not actively removed.

        Args:
            caller_id: The `CallerIdentifier` of the caller to unregister.
        """
        return False

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
        return self.__id_tracker.try_get(
            endpoint, port
        )  # Use self.__id_tracker
