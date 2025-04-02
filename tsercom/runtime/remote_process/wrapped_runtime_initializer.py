from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.runtime.remote_process.data_reader_sink import DataReaderSink
from tsercom.runtime.remote_process.runtime_data_source import RuntimeDataSource
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_command import RuntimeCommand
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_sink import MultiprocessQueueSink
from tsercom.threading.multiprocess.multiprocess_queue_source import MultiprocessQueueSource
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")

class WrappedRuntimeInitializer(
        Generic[TDataType, TEventType]):
    def __init__(
        self,
        initializer: RuntimeInitializer[TDataType, TEventType],
        event_queue: MultiprocessQueueSource[TEventType],
        data_queue: MultiprocessQueueSink[TDataType],
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__initializer = initializer
        self.__data_queue = data_queue
        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue

        self.__runtime_data_source : RuntimeDataSource | None = None

    def create_runtime(self, clock : SynchronizedClock, thread_watcher : ThreadWatcher) -> Runtime[TEventType]:
        data_reader = DataReaderSink(self.__data_queue)
        runtime = self.__initializer.create(clock, data_reader)
        self.__runtime_data_source = RuntimeDataSource(thread_watcher, self.__event_queue, self.__runtime_command_queue)
        self.__runtime_data_source.start_async(runtime)
        return runtime
