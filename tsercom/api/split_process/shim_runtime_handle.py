from typing import TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.api.split_process.data_reader_source import DataReaderSource
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.runtime_command import RuntimeCommand
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
TInitializerType = TypeVar("TInitializerType", bound=RuntimeInitializer)


class ShimRuntimeHandle(
    RuntimeHandle[TDataType, TEventType, TInitializerType],
    RemoteDataReader[TDataType],
):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSink[TEventType],
        data_queue: MultiprocessQueueSource[TDataType],
        runtime_command_queue: MultiprocessQueueSink[RuntimeCommand],
        data_aggregator: RemoteDataAggregatorImpl[TDataType],
        initializer: TInitializerType,
        block: bool = False,
    ):
        super().__init__()

        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue
        self.__data_reader_source = DataReaderSource(
            thread_watcher, data_queue, self.__data_aggregtor
        )
        self.__data_aggregtor = data_aggregator
        self.__initializer = initializer
        self.__block = block

    def start_async(self):
        self.__data_reader_source.start()
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStart)

    def on_event(self, event: TEventType):
        if self.__block:
            self.__event_queue.put_blocking(event)
        else:
            self.__event_queue.put_nowait(event)

    def stop(self):
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStop)
        self.__data_reader_source.stop()

    def _on_data_ready(self, new_data: TDataType) -> None:
        self.__data_aggregtor._on_data_ready(new_data)

    def _get_remote_data_aggregator(self) -> RemoteDataAggregator:
        return self.__data_aggregtor

    def _get_initializer(self) -> TInitializerType:
        return self.__initializer
