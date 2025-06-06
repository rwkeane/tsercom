"""ShimRuntimeHandle for interacting with a separate process runtime."""

import datetime
from typing import Optional, TypeVar

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.data_reader_source import DataReaderSource
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class ShimRuntimeHandle(
    RuntimeHandle[DataTypeT, EventTypeT],
    RemoteDataReader[AnnotatedInstance[DataTypeT]],
):
    """Handle for a runtime in a separate process.

    Provides mechanisms to start, stop, and send events to a runtime
    in a different process, using multiprocess queues for communication.
    """

    # pylint: disable=R0913,R0917 # Many parameters needed for full handle setup
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSink[EventInstance[EventTypeT]],
        data_queue: MultiprocessQueueSource[AnnotatedInstance[DataTypeT]],
        runtime_command_queue: MultiprocessQueueSink[RuntimeCommand],
        data_aggregator: RemoteDataAggregatorImpl[
            AnnotatedInstance[DataTypeT]
        ],
        block: bool = False,
    ) -> None:
        """Initializes the ShimRuntimeHandle.

        Args:
            thread_watcher: ThreadWatcher to monitor helper threads.
            event_queue: Sink for sending events to the remote runtime.
            data_queue: Source for receiving data from the remote runtime.
            runtime_command_queue: Sink for sending commands (start/stop).
            data_aggregator: Aggregator for data from remote via data_queue.
            block: If True, `on_event` blocks until event is queued.
                   If False, it's non-blocking.
        """
        super().__init__()

        self.__event_queue: MultiprocessQueueSink[
            EventInstance[EventTypeT]
        ] = event_queue
        self.__runtime_command_queue: MultiprocessQueueSink[RuntimeCommand] = (
            runtime_command_queue
        )
        self.__data_aggregator: RemoteDataAggregatorImpl[
            AnnotatedInstance[DataTypeT]
        ] = data_aggregator
        self.__block: bool = block

        self.__data_reader_source: DataReaderSource[
            AnnotatedInstance[DataTypeT]
        ] = DataReaderSource(
            thread_watcher,
            data_queue,
            self.__data_aggregator,
        )

    def start(self) -> None:
        """Starts the remote runtime interaction.

        Starts local data reader source to poll for data, then sends 'start'
        command to remote runtime via command queue.
        """
        self.__data_reader_source.start()
        self.__runtime_command_queue.put_blocking(RuntimeCommand.START)

    def on_event(
        self,
        event: EventTypeT,
        caller_id: Optional[
            CallerIdentifier
        ] = None,  # Added for RuntimeHandle compatibility
        *,
        timestamp: Optional[
            datetime.datetime
        ] = None,  # Added for RuntimeHandle compatibility
    ) -> None:
        """Sends an event to the remote runtime.

        Event is placed on event queue. Behavior (blocking/non-blocking)
        depends on `block` flag from init. `caller_id` and `timestamp` are
        ignored by shim but included for RuntimeHandle interface compatibility.

        Args:
            event: The event to send.
            caller_id: Optional caller ID (ignored).
            timestamp: Optional event timestamp (ignored).
        """
        # `caller_id` and `timestamp` are part of RuntimeHandle interface,
        # but shim doesn't use them when sending to queue.
        _ = caller_id  # Preserved for clarity, not used for queue type
        _ = timestamp  # Preserved for clarity

        effective_timestamp = (
            timestamp
            if timestamp is not None
            else datetime.datetime.now(tz=datetime.timezone.utc)
        )
        event_instance = EventInstance(
            data=event, caller_id=caller_id, timestamp=effective_timestamp
        )
        if self.__block:
            self.__event_queue.put_blocking(event_instance)
        else:
            self.__event_queue.put_nowait(event_instance)

    def stop(self) -> None:
        """Stops the remote runtime interaction.

        Sends 'stop' command to remote runtime via command queue,
        then stops local data reader source.
        """
        self.__runtime_command_queue.put_blocking(RuntimeCommand.STOP)
        self.__data_reader_source.stop()

    def _on_data_ready(self, new_data: AnnotatedInstance[DataTypeT]) -> None:
        """Callback for when new data is ready from the `DataReaderSource`.

        This is part of `RemoteDataReader` interface. `DataReaderSource`
        (polling data queue from remote) calls this. This forwards data
        to the main `data_aggregator`.

        Args:
            new_data: New data item from remote runtime.
        """
        # Data from DataReaderSource (from data_queue).
        # This handle (as RemoteDataReader) gets data from DataReaderSource
        # and forwards to the __data_aggregator from init.
        # pylint: disable=W0212 # Internal callback for client data readiness
        self.__data_aggregator._on_data_ready(new_data)

    def _get_remote_data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[DataTypeT]]:
        """Provides the remote data aggregator for this handle.

        Returns:
            The `RemoteDataAggregator` that processes data from remote.
        """
        return self.__data_aggregator

    @property
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[DataTypeT]]:
        """Provides the remote data aggregator."""
        return self._get_remote_data_aggregator()
