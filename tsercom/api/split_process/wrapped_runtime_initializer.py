from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.api.split_process.data_reader_sink import DataReaderSink
from tsercom.api.split_process.runtime_data_source import (
    RuntimeDataSource,
)
from tsercom.runtime.runtime import Runtime
from tsercom.api.split_process.runtime_command import RuntimeCommand
from tsercom.runtime.server.server_runtime_initializer import (
    ServerRuntimeInitializer,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class WrappedRuntimeInitializer(Generic[TDataType, TEventType]):
    def __init__(
        self,
        initializer: ServerRuntimeInitializer[TDataType, TEventType],
        event_queue: MultiprocessQueueSource[TEventType],
        data_queue: MultiprocessQueueSink[TDataType],
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__initializer = initializer
        self.__data_queue = data_queue
        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue

        self.__runtime_data_source: RuntimeDataSource | None = None

    def create_runtime(
        self,
        thread_watcher: ThreadWatcher,
        grpc_channel_factory: GrpcChannelFactory,
        *args,
        **kwargs,
    ) -> Runtime[TEventType]:
        data_reader = DataReaderSink(self.__data_queue)
        runtime = self.__initializer.create(
            data_reader, grpc_channel_factory, *args, **kwargs
        )
        self.__runtime_data_source = RuntimeDataSource(
            thread_watcher, self.__event_queue, self.__runtime_command_queue
        )
        self.__runtime_data_source.start_async(runtime)
        return runtime
