"""RemoteRuntimeFactory for creating Runtimes for separate processes."""

from typing import Generic, TypeVar

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.api.split_process.data_reader_sink import DataReaderSink
from tsercom.api.split_process.event_source import EventSource
from tsercom.api.split_process.runtime_command_source import (
    RuntimeCommandSource,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance

# SerializableAnnotatedInstance will be removed
# from tsercom.data.serializable_annotated_instance import SerializableAnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class RemoteRuntimeFactory(
    Generic[DataTypeT, EventTypeT],
    RuntimeFactory[
        DataTypeT,
        EventTypeT,
    ],
):
    """Factory for Runtimes that operate in a separate process.

    Sets up communication channels (queues) for events, data, and commands
    to interact with a runtime in a different process. Uses DataReaderSink,
    EventSource, and RuntimeCommandSource.
    """

    def __init__(
        self,
        initializer: RuntimeInitializer[DataTypeT, EventTypeT],
        event_source_queue: MultiprocessQueueSource[EventInstance[EventTypeT]],
        data_reader_queue: MultiprocessQueueSink[AnnotatedInstance[DataTypeT]],
        command_source_queue: MultiprocessQueueSource[RuntimeCommand],
    ) -> None:
        """Initializes the RemoteRuntimeFactory.

        Args:
            initializer: Initializer for the runtime.
            event_source_queue: Queue source for receiving EventInstances.
            data_reader_queue: Queue sink for sending AnnotatedInstances.
            command_source_queue: Queue source for receiving RuntimeCommands.
        """
        super().__init__(other_config=initializer)
        self._initializer_instance = initializer
        self.__event_source_queue = event_source_queue  # Type updated
        self.__data_reader_queue = data_reader_queue
        self.__command_source_queue = command_source_queue

        self.__data_reader_sink: (
            DataReaderSink[AnnotatedInstance[DataTypeT]] | None
        ) = None
        self.__event_source: EventSource[EventTypeT] | None = None
        self.__command_source: RuntimeCommandSource | None = None

    @property
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[DataTypeT]]:
        """Gets the `DataReaderSink` for sending data to remote runtime."""
        return self._remote_data_reader()

    @property
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[EventTypeT]]:
        """Gets the `EventSource` for polling events from remote runtime."""
        return self._event_poller()

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[DataTypeT]]:
        """Provides the data reader sink for the remote runtime.

        Lazily initializes and returns a `DataReaderSink`.

        Returns:
            A `DataReaderSink` instance.
        """
        # Note: Base `RuntimeFactory` expects RemoteDataReader[AnnotatedInstance[DataTypeT]].
        # DataReaderSink is compatible.
        if self.__data_reader_sink is None:
            self.__data_reader_sink = DataReaderSink(self.__data_reader_queue)
        return self.__data_reader_sink

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[EventTypeT]]:
        """Provides the event poller for events from remote runtime.

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
        data_handler: RuntimeDataHandler[DataTypeT, EventTypeT],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        """Creates remote Runtime instance and sets up command handling.

        Initializes core runtime. Starts event source (if initialized via
        `_event_poller`). Sets up command source to relay to new runtime.

        Args:
            thread_watcher: ThreadWatcher to monitor component threads.
            data_handler: Data handler for the runtime.
            grpc_channel_factory: gRPC channel factory (required).

        Returns:
            Created Runtime instance, configured for remote operation.
        """
        runtime = self._initializer_instance.create(
            thread_watcher=thread_watcher,
            data_handler=data_handler,
            grpc_channel_factory=grpc_channel_factory,
        )
        current_event_poller = (
            self._event_poller()
        )  # This will initialize EventSource if None
        if isinstance(current_event_poller, EventSource):
            if not current_event_poller.is_running:
                current_event_poller.start(thread_watcher)
        else:
            # This case should not happen if _event_poller always returns EventSource
            # or raises. Consider logging if None/wrong type.
            pass  # Or logger.error("Event poller not EventSource post-creation")

        self.__command_source = RuntimeCommandSource(
            self.__command_source_queue
        )
        self.__command_source.start_async(thread_watcher, runtime)
        return runtime

    def _stop(self) -> None:
        """Stops command source and event source."""
        if self.__command_source is not None:
            self.__command_source.stop_async()
        if (
            self.__event_source is not None
            and hasattr(self.__event_source, "is_running")
            and self.__event_source.is_running
        ):
            self.__event_source.stop()
