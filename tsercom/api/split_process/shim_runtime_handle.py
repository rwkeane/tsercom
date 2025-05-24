from typing import TypeVar

from tsercom.data.exposed_data import ExposedData
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


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class ShimRuntimeHandle(
    RuntimeHandle[TDataType, TEventType],
    RemoteDataReader[TDataType],
):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSink[TEventType],
        data_queue: MultiprocessQueueSource[TDataType], # This is the data_source queue
        runtime_command_queue: MultiprocessQueueSink[RuntimeCommand],
        data_aggregator: RemoteDataAggregatorImpl[TDataType], # This is the main process aggregator
        block: bool = False,
    ):
        super().__init__()
        print(f"DEBUG: [ShimRuntimeHandle.__init__] Initializing. data_queue (source for this handle): {data_queue}, data_aggregator: {data_aggregator}")

        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue
        self.__data_aggregtor = data_aggregator # Corrected variable name from previous context if needed, using provided spelling
        self.__block = block

        print(f"DEBUG: [ShimRuntimeHandle.__init__] Creating DataReaderSource with data_queue: {data_queue} and aggregator: {self.__data_aggregtor}")
        self.__data_reader_source = DataReaderSource(
            thread_watcher, data_queue, self.__data_aggregtor # Pass the aggregator here
        )
        print(f"DEBUG: [ShimRuntimeHandle.__init__] DataReaderSource created: {self.__data_reader_source}")


    def start(self):
        print(f"DEBUG: [ShimRuntimeHandle.start] Calling self.__data_reader_source.start()")
        self.__data_reader_source.start()
        print(f"DEBUG: [ShimRuntimeHandle.start] Calling self.__runtime_command_queue.put_blocking(RuntimeCommand.kStart)")
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStart)
        print(f"DEBUG: [ShimRuntimeHandle.start] RuntimeCommand.kStart put on queue.")

    def on_event(self, event: TEventType):
        if self.__block:
            self.__event_queue.put_blocking(event)
        else:
            self.__event_queue.put_nowait(event)

    def stop(self):
        self.__runtime_command_queue.put_blocking(RuntimeCommand.kStop)
        self.__data_reader_source.stop()

    def _on_data_ready(self, new_data: TDataType) -> None:
        # This method is part of RemoteDataReader interface, usually called when this class itself is used as a reader.
        # In this ShimRuntimeHandle, data is typically *received* from data_queue via DataReaderSource
        # and then fed into its __data_aggregtor.
        # If ShimRuntimeHandle itself were to receive data directly (not via DataReaderSource), this would be used.
        data_value = getattr(getattr(new_data, 'data', new_data), 'value', str(new_data))
        print(f"DEBUG: [ShimRuntimeHandle._on_data_ready] new_data: {data_value}. Forwarding to self.__data_aggregtor._on_data_ready.")
        self.__data_aggregtor._on_data_ready(new_data)


    def _get_remote_data_aggregator(self) -> RemoteDataAggregator:
        return self.__data_aggregtor
