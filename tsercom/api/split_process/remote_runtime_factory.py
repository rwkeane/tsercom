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
        """Gets the `DataReaderSink` used for sending data to the remote runtime."""
        return self._remote_data_reader()

    @property
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        """Gets the `EventSource` used for polling events from the remote runtime."""
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

        current_event_poller = (
            self._event_poller()
        )  # This will initialize EventSource if None
        if isinstance(current_event_poller, EventSource):
            if not current_event_poller.is_running:
                current_event_poller.start(thread_watcher)
        else:
            # This case should ideally not happen if _event_poller always returns EventSource
            # or raises an error. Consider logging an error if it's None or wrong type.
            pass  # Or logger.error("Event poller is not an EventSource instance after creation")

        self.__command_source = RuntimeCommandSource(
            self.__command_source_queue
        )
        self.__command_source.start_async(thread_watcher, runtime)
        return runtime

    def _stop(self) -> None:
        """Stops associated components like the command source and event source."""
        if self.__command_source is not None:
            self.__command_source.stop_async()
        if (
            self.__event_source is not None
            and hasattr(self.__event_source, "is_running")
            and self.__event_source.is_running
        ):
            self.__event_source.stop()
