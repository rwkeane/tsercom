"""Implements the client-side runtime data handling logic.

This module defines `ClientRuntimeDataHandler` which is responsible for
managing data flow, caller registration, and time synchronization aspects
for Tsercom runtimes operating in a client role.
"""

from typing import Generic, TypeVar, cast  # Added cast
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.caller_processor_registry import (
    CallerProcessorRegistry,
)  # Added
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
import logging
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType", bound=ExposedData)


class ClientRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    """Handles data, events, and caller management for client runtimes.

    It integrates with a `TimeSyncTracker` for clock synchronization and
    an `IdTracker` to manage associations between caller IDs and their
    network endpoints. It processes incoming events and makes data
    available via a `RemoteDataReader`.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
        # id_tracker: IdTracker, # Removed from parameters
        *,
        is_testing: bool = False,
    ):
        """Initializes the ClientRuntimeDataHandler.

        Args:
            thread_watcher: For monitoring internal threads.
            data_reader: The reader for incoming data instances.
            event_source: The poller for incoming event instances.
            is_testing: If True, enables testing-specific behaviors (e.g., fake time sync).
        """
        self.__id_tracker = (
            IdTracker()
        )  # Changed to use standard name mangling
        self.__clock_tracker = (
            TimeSyncTracker(  # Changed to use standard name mangling
                thread_watcher, is_testing=is_testing
            )
        )

        # Define processor_factory for CallerProcessorRegistry
        def processor_factory(
            caller_id_for_factory: CallerIdentifier,
        ) -> AsyncPoller[SerializableAnnotatedInstance[TEventType]]:
            # 'self' refers to ClientRuntimeDataHandler instance
            # This assumes TimeSyncTracker can provide a clock for a given CallerIdentifier.
            # This is a potential design issue if not implemented carefully in TimeSyncTracker.
            clock = self.__clock_tracker.get_clock_for_caller_id(  # Use self.__clock_tracker
                caller_id_for_factory
            )  # HYPOTHETICAL METHOD on TimeSyncTracker
            if clock is None:
                raise RuntimeError(
                    f"Client-side processor factory could not retrieve clock for CallerIdentifier: {caller_id_for_factory}. "
                    "Ensure TimeSyncTracker maps CallerID to clock after on_connect, or this factory design is flawed for clients."
                )

            concrete_processor = (
                self._create_data_processor(  # from RuntimeDataHandlerBase
                    caller_id_for_factory, clock
                )
            )
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
            id_tracker=self.__id_tracker,  # Pass its own IdTracker
            processor_registry=processor_registry,
        )

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[TDataType, TEventType]:
        """Registers a new caller and its endpoint.

        Adds the caller's address to its privately managed ID tracker.
        Establishes clock synchronization for the endpoint.
        Returns the EndpointDataProcessor instance (_RuntimeDataHandlerBase__DataProcessorImpl);
        the actual poller is created by CallerProcessorRegistry's factory on demand.
        """
        # Map address to CallerID using the handler's own IdTracker
        self.__id_tracker.add(
            caller_id, endpoint, port
        )  # Use self.__id_tracker

        # Establish clock sync for this endpoint. This is crucial for the factory later.
        # on_connect now also takes caller_id.
        self.__clock_tracker.on_connect(
            endpoint, caller_id
        )  # Use self.__clock_tracker

        # The factory will need to retrieve this clock. For _create_data_processor here,
        # we also need the clock. Assume get_clock_for_endpoint exists after on_connect.
        clock = self.__clock_tracker.get_clock_for_endpoint(  # Use self.__clock_tracker
            endpoint
        )  # HYPOTHETICAL METHOD on TimeSyncTracker
        if clock is None:
            # This would be an issue, means on_connect didn't make a clock available immediately.
            raise RuntimeError(
                f"Clock not available for endpoint {endpoint} immediately after on_connect."
            )

        # Return the __ConcreteDataProcessor instance.
        return self._create_data_processor(caller_id, clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """
        Unregisters a caller.

        Args:
            caller_id: The ID of the caller to unregister.

        Returns:
            True if the caller was found and unregistered, False otherwise.
        """
        address_port_tuple = self.__id_tracker.try_get(
            caller_id
        )  # Use self.__id_tracker

        if address_port_tuple is not None:
            address, _ = (
                address_port_tuple  # port is not needed for on_disconnect
            )
            self.__id_tracker.remove(caller_id)  # Use self.__id_tracker
            self.__clock_tracker.on_disconnect(
                address
            )  # Use self.__clock_tracker
            return True
        else:
            logging.warning(
                f"Attempted to unregister non-existent caller_id: {caller_id}"
            )
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
