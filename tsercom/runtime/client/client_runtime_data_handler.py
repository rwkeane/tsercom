"""Implements the client-side runtime data handling logic.

This module defines `ClientRuntimeDataHandler` which is responsible for
managing data flow, caller registration, and time synchronization aspects
for Tsercom runtimes operating in a client role.
"""

import logging  # Moved to top
from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher


EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class ClientRuntimeDataHandler(
    Generic[DataTypeT, EventTypeT],
    RuntimeDataHandlerBase[DataTypeT, EventTypeT],
):
    """Handles data, events, and caller management for client runtimes.

    Integrates `TimeSyncTracker` for clock sync and `IdTracker` for
    managing caller ID and endpoint associations. It processes incoming
    events and makes data available via a `RemoteDataReader`.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
        min_send_frequency_seconds: float | None = None,
        *,
        is_testing: bool = False,
    ):
        """Initializes the ClientRuntimeDataHandler.

        Args:
            thread_watcher: For monitoring internal threads.
            data_reader: Reader for incoming data instances.
            event_source: Poller for incoming event instances.
            is_testing: If True, enables testing-specific behaviors (e.g.,
                        fake time sync).
        """
        super().__init__(data_reader, event_source, min_send_frequency_seconds)

        self.__clock_tracker = TimeSyncTracker(
            thread_watcher, is_testing=is_testing
        )

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]:
        """Registers new caller, returns data processor.

        Adds caller to ID tracker, inits time sync for the endpoint.

        Args:
            caller_id: The `CallerIdentifier` of the new caller.
            endpoint: Network endpoint (e.g., IP address) of the caller.
            port: Port number of the caller.

        Returns:
            An `EndpointDataProcessor` configured for this caller.
        """
        self._id_tracker.add(caller_id, endpoint, port)
        clock = self.__clock_tracker.on_connect(endpoint)
        return self._create_data_processor(caller_id, clock)

    async def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Unregisters a caller.

        Args:
            caller_id: The ID of the caller to unregister.

        Returns:
            True if caller was found and unregistered, False otherwise.
        """
        address_port_tuple = self._id_tracker.try_get(caller_id)
        if address_port_tuple is None:
            logging.warning(
                "Attempted to unregister non-existent caller_id: %s", caller_id
            )
            return False

        address, _, _ = address_port_tuple
        self._id_tracker.remove(caller_id)
        self.__clock_tracker.on_disconnect(address)
        return True
