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
        event_source_queue: MultiprocessQueueSource[TEventType],
        data_reader_queue: MultiprocessQueueSink[TDataType],
        command_source_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        super().__init__(other_config=initializer)
        self.__initializer = initializer
        self.__event_queue_source_obj = event_source_queue
        self.__data_reader_queue_sink_obj = data_reader_queue
        self.__command_queue_source_obj = command_source_queue

        self.__data_reader: DataReaderSink[TDataType] | None = None
        self.__event_source: EventSource[TEventType] | None = None
        self.__command_source: RuntimeCommandSource | None = None

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        if self.__data_reader is None:
            self.__data_reader = DataReaderSink(self.__data_reader_queue_sink_obj)
        return self.__data_reader

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        if self.__event_source is None:
            self.__event_source = EventSource(self.__event_queue_source_obj)
        return self.__event_source

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        if self.__event_source:
            self.__event_source.start(thread_watcher)
        else:
            # This case should ideally not be reached if the flow through
            # RuntimeFactory.create_runtime_components is correct,
            # as _create_data_handler (which calls _event_poller)
            # should have been called prior to this create method.
            # Consider raising an error or ensuring _event_poller is called.
            pass


        runtime = self.__initializer.create(
            thread_watcher, data_handler, grpc_channel_factory
        )

        self.__command_source = RuntimeCommandSource(
            self.__command_queue_source_obj
        )
        self.__command_source.start_async(thread_watcher, runtime)

        return runtime
