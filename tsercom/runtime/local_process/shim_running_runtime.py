from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.local_process.data_reader_source import DataReaderSource
from tsercom.runtime.running_runtime import RunningRuntime
from tsercom.runtime.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import MultiprocessQueueSource
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class ShimRunningRuntime(
    RunningRuntime[TEventType, TDataType], RemoteDataReader[TDataType]
):
    def __init__(
        self,
        thread_watcher : ThreadWatcher,
        aggregator_thread_pool : ThreadPoolExecutor,
        event_queue: MultiprocessQueueSink[TEventType],
        data_queue: MultiprocessQueueSource[TDataType],
        runtime_command_queue: MultiprocessQueueSink[RuntimeCommand],
    ):
        super().__init__()

        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue
        self.__data_reader_source = DataReaderSource(thread_watcher, data_queue, self.__data_aggregtor)

        self.__data_aggregtor: RemoteDataAggregatorImpl = (
            RemoteDataAggregatorImpl(aggregator_thread_pool)
        )

    def start_async(self):
        self.__data_reader_source.start()
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStart)

    def on_event(self, event: TEventType, block: bool = False):
        if block:
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
