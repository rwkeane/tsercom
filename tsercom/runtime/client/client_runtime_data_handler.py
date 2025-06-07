"""Implements the client-side specific `RuntimeDataHandler` logic.

This module defines `ClientRuntimeDataHandler`, a concrete implementation of
`RuntimeDataHandlerBase` tailored for Tsercom runtimes operating in a client role.
It is responsible for managing client-specific aspects of data flow,
caller registration (typically of remote servers or services it connects to),
and time synchronization with those remote entities.
"""

import logging
from typing import Generic, Optional, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.synchronized_clock import (
    SynchronizedClock,
)

EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT")


class ClientRuntimeDataHandler(
    Generic[DataTypeT, EventTypeT],
    RuntimeDataHandlerBase[DataTypeT, EventTypeT],
):
    """Handles data, events, and caller management for client-side runtimes.

    This class extends `RuntimeDataHandlerBase` to provide functionality
    specific to client runtimes. A key responsibility is managing time
    synchronization with each connected remote endpoint (server/service). It uses a
    `TimeSyncTracker` to establish and maintain synchronized clocks for each
    caller (remote endpoint) it registers.

    When registering a caller (representing a connection to a remote server),
    it initiates time synchronization for that server\'s endpoint and uses the
    resulting synchronized clock when creating the `EndpointDataProcessor` for
    that caller. Similarly, it notifies the `TimeSyncTracker` upon unregistering
    a caller.

    Type Args:
        DataTypeT: The generic type of data objects that this handler processes.
        EventTypeT: The generic type of event objects that this handler processes.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
        min_send_frequency_seconds: Optional[float] = None,
        *,
        is_testing: bool = False,
    ):
        """Initializes the ClientRuntimeDataHandler.

        Args:
            thread_watcher: The `ThreadWatcher` instance for monitoring any
                background threads or tasks, particularly those spawned by
                the `TimeSyncTracker`.
            data_reader: The `RemoteDataReader` where incoming data, after being
                processed and annotated, will be sent.
            event_source: The primary `AsyncPoller` that sources events for
                all callers handled by this data handler.
            min_send_frequency_seconds: Optional minimum time interval, in seconds,
                for the per-caller event pollers created by the underlying
                `IdTracker`. Passed to `RuntimeDataHandlerBase`.
            is_testing: If True, configures certain components like
                `TimeSyncTracker` to use test-specific behaviors (e.g.,
                a fake time synchronization mechanism).
        """
        super().__init__(data_reader, event_source, min_send_frequency_seconds)

        self.__clock_tracker: TimeSyncTracker = TimeSyncTracker(
            thread_watcher, is_testing=is_testing
        )

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]:
        """Registers a new remote caller (e.g., a server this client connects to).

        This implementation adds the caller (identified by `caller_id`, `endpoint`,
        and `port`) to the internal `IdTracker` (managed by the base class).
        It then initiates time synchronization for the given `endpoint` using
        the `TimeSyncTracker`, obtaining a `SynchronizedClock` specific to that
        endpoint. Finally, it creates and returns an `EndpointDataProcessor`
        instance configured with this `caller_id` and the obtained
        `SynchronizedClock`.

        Args:
            caller_id: The `CallerIdentifier` assigned to the remote caller.
            endpoint: The network endpoint (e.g., IP address or hostname) of the
                remote caller.
            port: The port number of the remote caller.

        Returns:
            An `EndpointDataProcessor` instance configured for communication
            with the registered remote caller, using a synchronized clock for
            that endpoint.
        """
        # Add to IdTracker (from base class) to associate caller_id with address
        # and to get a dedicated AsyncPoller for its events.
        self._id_tracker.add(caller_id, endpoint, port)

        # Establish time synchronization for this specific endpoint
        clock: SynchronizedClock = self.__clock_tracker.on_connect(endpoint)

        # Create the data processor using the synchronized clock for this endpoint
        return self._create_data_processor(caller_id, clock)

    async def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Unregisters a remote caller and cleans up associated resources.

        This method removes the specified `caller_id` from the `IdTracker`.
        It also notifies the `TimeSyncTracker` that the connection to the
        caller\'s endpoint has been terminated, allowing the tracker to clean up
        any resources associated with time synchronization for that endpoint.

        Args:
            caller_id: The `CallerIdentifier` of the remote caller to unregister.

        Returns:
            True if the caller was found in the `IdTracker` and successfully
            unregistered (including from the clock tracker), False otherwise
            (e.g., if the `caller_id` was not found).
        """
        # Retrieve address details before removing from IdTracker
        # try_get by ID returns (address, port, data_poller)
        id_tracker_entry = self._id_tracker.try_get(caller_id)

        if id_tracker_entry is None:
            logging.warning(
                "Attempted to unregister non-existent caller_id: %s", caller_id
            )
            return False

        address: str = id_tracker_entry[0]  # address is the first element

        # Remove from IdTracker first
        was_removed_from_id_tracker = self._id_tracker.remove(caller_id)

        if not was_removed_from_id_tracker:
            # This case should ideally not be reached if id_tracker_entry was found,
            # but included for robustness.
            logging.warning(
                "Failed to remove caller_id %s from IdTracker, though it was initially found. "
                "Skipping clock_tracker.on_disconnect.",
                caller_id,
            )
            return False

        # If successfully removed from IdTracker, also handle clock tracker cleanup
        self.__clock_tracker.on_disconnect(address)
        return True
