"""Defines ShimRuntimeHandle for interacting with a runtime in a separate process."""

import datetime  # Required for on_event overload, though not used in current simple form
from typing import TypeVar, Optional

from tsercom.caller_id.caller_identifier import (
    CallerIdentifier,
)  # For on_event overload
from tsercom.data.exposed_data import ExposedData
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.api.split_process.data_reader_source import DataReaderSource
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar(
    "TDataType", bound=ExposedData
)  # Type for data handled by the runtime.
TEventType = TypeVar("TEventType")  # Type for events handled by the runtime.


class ShimRuntimeHandle(
    RuntimeHandle[
        TDataType, TEventType
    ],  # Implements the abstract RuntimeHandle
    RemoteDataReader[TDataType],  # Also acts as a RemoteDataReader
):
    """A handle for a runtime operating in a separate process.

    This class provides the mechanisms to start, stop, and send events to a
    runtime that is managed in a different process. It uses multiprocess queues
    for communication (events, commands, data).
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSink[TEventType],
        data_queue: MultiprocessQueueSource[TDataType],
        runtime_command_queue: MultiprocessQueueSink[RuntimeCommand],
        data_aggregator: RemoteDataAggregatorImpl[TDataType],
        block: bool = False,
    ) -> None:
        """Initializes the ShimRuntimeHandle.

        Args:
            thread_watcher: A ThreadWatcher to monitor helper threads.
            event_queue: Queue sink for sending events to the remote runtime.
            data_queue: Queue source for receiving data from the remote runtime.
            runtime_command_queue: Queue sink for sending commands (start/stop)
                                   to the remote runtime.
            data_aggregator: The aggregator that will receive data from the
                             remote runtime via the data_queue.
            block: If True, sending events via `on_event` will block until the
                   event is placed in the queue. If False, it's non-blocking.
        """
        super().__init__()

        self.__event_queue: MultiprocessQueueSink[TEventType] = event_queue
        self.__runtime_command_queue: MultiprocessQueueSink[RuntimeCommand] = (
            runtime_command_queue
        )
        self.__data_aggregator: RemoteDataAggregatorImpl[TDataType] = (
            data_aggregator
        )
        self.__block: bool = block

        # DataReaderSource is initialized with the same data_queue
        self.__data_reader_source: DataReaderSource[TDataType] = (
            DataReaderSource(
                thread_watcher,
                data_queue,
                self.__data_aggregator,  # Pass data_aggregator as the data_reader for the DataReaderSource
            )
        )

    def start(self) -> None:
        """Starts the remote runtime interaction.

        This starts the local data reader source to begin polling for data from
        the remote runtime, and then sends a 'start' command to the remote
        runtime via the command queue.
        """
        self.__data_reader_source.start()
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStart)

    def on_event(
        self,
        event: TEventType,
        caller_id: Optional[
            CallerIdentifier
        ] = None,  # Added for RuntimeHandle compatibility
        *,
        timestamp: Optional[
            datetime.datetime
        ] = None,  # Added for RuntimeHandle compatibility
    ) -> None:
        """Sends an event to the remote runtime.

        The event is placed onto the event queue. Behavior (blocking or
        non-blocking) depends on the `block` flag set during initialization.
        `caller_id` and `timestamp` are ignored by this shim but included for
        compatibility with the `RuntimeHandle` interface.

        Args:
            event: The event to send.
            caller_id: Optional identifier of the caller (ignored by this implementation).
            timestamp: Optional timestamp for the event (ignored by this implementation).
        """
        # `caller_id` and `timestamp` are part of the RuntimeHandle interface,
        # but this shim implementation does not use them when sending to the queue.
        _ = caller_id
        _ = timestamp
        if self.__block:
            self.__event_queue.put_blocking(event)
        else:
            self.__event_queue.put_nowait(event)

    def stop(self) -> None:
        """Stops the remote runtime interaction.

        Sends a 'stop' command to the remote runtime via the command queue,
        and then stops the local data reader source.
        """
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStop)
        self.__data_reader_source.stop()

    def _on_data_ready(self, new_data: TDataType) -> None:
        """Callback for when new data is ready from the `DataReaderSource`.

        This method is part of the `RemoteDataReader` interface. In this setup,
        the `DataReaderSource` (which polls the actual data queue from the remote
        process) calls this method. This implementation then forwards the data
        to the main `data_aggregator`.

        Args:
            new_data: The new data item received from the remote runtime.
        """
        # Data is received from data_queue via DataReaderSource.
        # This ShimRuntimeHandle (as a RemoteDataReader) gets it from DataReaderSource
        # and then forwards it to the __data_aggregator that was provided during init.
        self.__data_aggregator._on_data_ready(new_data)

    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        """Provides the remote data aggregator associated with this handle.

        Returns:
            The `RemoteDataAggregator` instance that processes data received
            from the remote runtime.
        """
        return self.__data_aggregator

    @property
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[TDataType]]:  # type: ignore[override]
        # TODO(developer/bug_id): This property currently returns self._get_remote_data_aggregator(),
        # which is of type RemoteDataAggregator[TDataType]. This may not match the
        # RuntimeHandle base class's expected return type of
        # RemoteDataAggregator[AnnotatedInstance[TDataType]] if TDataType itself
        # is not an AnnotatedInstance. This type mismatch is currently suppressed
        # with type: ignore[override] and needs to be resolved.
        """Provides the remote data aggregator.

        Note: There is a potential type mismatch with the base `RuntimeHandle` class
        if `TDataType` for this handle is not already an `AnnotatedInstance`.
        See TODO comment in source code for details.

        Returns:
            The `RemoteDataAggregator` instance.
        """
        return self._get_remote_data_aggregator()
