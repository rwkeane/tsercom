"""Defines RemoteRuntimeFactory for creating Runtime instances intended for separate processes."""

from typing import Generic, TypeVar

from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.api.split_process.data_reader_sink import DataReaderSink
from tsercom.api.split_process.event_source import EventSource
from tsercom.api.split_process.runtime_command_source import (
    RuntimeCommandSource,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.runtime import Runtime
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RemoteRuntimeFactory(
    Generic[TDataType, TEventType],
    RuntimeFactory[
        TDataType,
        TEventType,
    ],
):
    """Factory for creating Runtimes that operate in a separate process.

    This factory sets up communication channels (queues) for events, data,
    and commands to interact with a runtime that is presumably running in
    a different process. It initializes and uses `DataReaderSink`,
    `EventSource`, and `RuntimeCommandSource` for these interactions.
    """

    def __init__(
        self,
        initializer: RuntimeInitializer[TDataType, TEventType],
        event_source_queue: MultiprocessQueueSource[EventInstance[TEventType]],
        data_reader_queue: MultiprocessQueueSink[AnnotatedInstance[TDataType]],
        command_source_queue: MultiprocessQueueSource[RuntimeCommand],
    ) -> None:
        """Initializes the RemoteRuntimeFactory.

        Args:
            initializer: The RuntimeInitializer for the runtime to be created.
            event_source_queue: Queue source for receiving `EventInstance` objects from the remote runtime.
            data_reader_queue: Queue sink for sending `AnnotatedInstance` data to the remote runtime.
            command_source_queue: Queue source for receiving `RuntimeCommand` objects from the handle.
        """
        super().__init__(other_config=initializer)
        self._initializer_instance = initializer
        self.__event_source_queue = event_source_queue
        self.__data_reader_queue = data_reader_queue
        self.__command_source_queue = command_source_queue

        self.__data_reader_sink: (
            DataReaderSink[AnnotatedInstance[TDataType]] | None
        ) = None
        self.__event_source: EventSource[TEventType] | None = None
        self.__command_source: RuntimeCommandSource | None = None

    @property
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        """Provides a `RemoteDataReader` for accessing annotated data instances."""
        return self._remote_data_reader()

    @property
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        """Provides an `AsyncPoller` for receiving event instances."""
        return self._event_poller()

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        """Provides the data reader sink for the remote runtime.

        Lazily initializes and returns a `DataReaderSink`.

        Returns:
            A `DataReaderSink` instance.
        """
        # Note: The base `RuntimeFactory` expects `RemoteDataReader[AnnotatedInstance[TDataType]]`.
        # DataReaderSink is designed to be compatible with this expectation.
        if self.__data_reader_sink is None:
            self.__data_reader_sink = DataReaderSink(self.__data_reader_queue)
        return self.__data_reader_sink

    def _event_poller(self) -> AsyncPoller[EventInstance[TEventType]]:
        """Provides the event poller for events from the remote runtime.

        Lazily initializes and returns an `EventSource`.

        Returns:
            An `EventSource` instance.
        """
        if self.__event_source is None:
            self.__event_source = EventSource(self.__event_source_queue)
        return self.__event_source

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory | None,
    ) -> Runtime:
        """Creates the remote Runtime instance and sets up command handling.

        This method initializes the core runtime using the provided initializer.
        It also starts the event source (if previously initialized by a call to `_event_poller`)
        and sets up the command source to relay commands to the newly created runtime.

        Args:
            thread_watcher: A ThreadWatcher to monitor threads created by components.
            data_handler: The data handler for the runtime.
            grpc_channel_factory: Factory for gRPC channels if needed by the runtime.

        Returns:
            The created Runtime instance, configured for remote operation.
        """
        runtime = self._initializer_instance.create(
            thread_watcher=thread_watcher,
            data_handler=data_handler,
            grpc_channel_factory=grpc_channel_factory,
        )

        if self.__event_source:  # self.__event_source is current_event_poller
            self.__event_source.start(thread_watcher)
        else:
            # WARNING: Event source was not initialized prior to create call.
            # This might indicate an issue if events are expected to be processed
            # immediately after runtime creation without a prior call to _event_poller.
            # Consider if _event_poller should always be called, e.g., in __init__ or at the start of create.
            pass  # Or log a warning, though current design relies on _event_poller being called if needed.

        self.__command_source = RuntimeCommandSource(
            self.__command_source_queue
        )
        self.__command_source.start_async(thread_watcher, runtime)
        return runtime

    def stop_command_source(self) -> None:
        # Helper print statements for debugging during the test run
        print(f"RemoteRuntimeFactory: Attempting to stop command source for initializer type {type(self._initializer_instance).__name__}...", flush=True)
        if self.__command_source is not None:
            # Assuming self.__command_source has a stop_async method
            # as per previous modifications to RuntimeCommandSource
            self.__command_source.stop_async()
            print(f"RemoteRuntimeFactory: Command source stop_async() called for {type(self._initializer_instance).__name__}.", flush=True)
        else:
            print(f"RemoteRuntimeFactory: No command source to stop for {type(self._initializer_instance).__name__}.", flush=True)
