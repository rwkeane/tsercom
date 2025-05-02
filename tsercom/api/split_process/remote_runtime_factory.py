from typing import Generic, TypeVar

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.api.split_process.data_reader_sink import DataReaderSink
from tsercom.api.split_process.event_source import EventSource
from tsercom.api.split_process.runtime_command_source import (
    RuntimeCommandSource,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher

TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class RemoteRuntimeFactory(
    Generic[TDataType, TEventType], RuntimeFactory[TDataType, TEventType]
):
    def __init__(
        self,
        initializer: RuntimeInitializer,
        event_source: MultiprocessQueueSource[TEventType],
        data_reader_sink: MultiprocessQueueSink[TDataType],
        command_source: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__initializer = initializer
        self.__event_queue_source = event_source
        self.__data_reader_queue_sink = data_reader_sink
        self.__command_queue_source = command_source

        self.__event_source: EventSource | None = None
        self.__data_reader: DataReaderSink | None = None
        self.__command_source: RuntimeCommandSource | None = None

        super().__init__()

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime[TEventType]:
        runtime = self.__initializer.create(
            thread_watcher, data_handler, grpc_channel_factory
        )

        self.__event_source = EventSource(self.__event_queue_source)

        self.__data_reader = DataReaderSink(self.__data_reader_queue_sink)
        self.__event_source.start(thread_watcher)

        self.__command_source = RuntimeCommandSource(
            self.__command_queue_source
        )
        self.__command_source.start_async(thread_watcher, runtime)

        return runtime

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        return self.__data_reader

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        return self.__event_source
