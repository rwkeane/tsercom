import threading
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TDataType = TypeVar("TDataType", bound=ExposedData)


class DataReaderSource(Generic[TDataType]):
    def __init__(
        self,
        watcher: ThreadWatcher,
        queue: MultiprocessQueueSource[TDataType], # This is the data_source from SplitRuntimeFactoryFactory
        data_reader: RemoteDataReader[TDataType], # This is the RemoteDataAggregatorImpl from main process
    ):
        self.__queue = queue
        self.__data_reader = data_reader # This is the aggregator to call _on_data_ready on
        self.__watcher = watcher

        self.__thread: threading.Thread | None = None
        self.__is_running = IsRunningTracker()
        print(f"DEBUG: [DataReaderSource.__init__] Initialized. queue: {self.__queue}, data_reader (aggregator): {self.__data_reader}")

    @property
    def is_running(self):
        return self.__is_running.get()

    def start(self):
        print(f"DEBUG: [DataReaderSource.start] Starting __poll_for_data thread.")
        self.__is_running.start() # Start running before thread creation
        self.__thread = self.__watcher.create_tracked_thread(
            self.__poll_for_data
        )
        self.__thread.start()
        print(f"DEBUG: [DataReaderSource.start] __poll_for_data thread started.")


    def stop(self):
        print(f"DEBUG: [DataReaderSource.stop] Stopping __poll_for_data thread.")
        self.__is_running.stop()
        if self.__thread:
            self.__thread.join(timeout=5) # Wait for thread to finish
            print(f"DEBUG: [DataReaderSource.stop] __poll_for_data thread stopped and joined.")


    def __poll_for_data(self):
        print(f"DEBUG: [DataReaderSource.__poll_for_data] Thread started. Polling for data from queue: {self.__queue}")
        while self.__is_running.get():
            print(f"DEBUG: [DataReaderSource.__poll_for_data] Calling self.__queue.get_blocking(timeout=1)")
            data = self.__queue.get_blocking(timeout=1)
            if data is not None:
                data_value = getattr(getattr(data, 'data', data), 'value', str(data))
                caller_id_value = getattr(data, 'caller_id', 'UnknownCallerId')
                print(f"DEBUG: [DataReaderSource.__poll_for_data] Received data from queue. Data: {data_value}, Caller ID: {caller_id_value}")
                print(f"DEBUG: [DataReaderSource.__poll_for_data] Calling self.__data_reader._on_data_ready (aggregator._on_data_ready). Data: {data_value}, Caller ID: {caller_id_value}")
                try:
                    self.__data_reader._on_data_ready(data) # data_reader here is the RemoteDataAggregatorImpl
                    print(f"DEBUG: [DataReaderSource.__poll_for_data] Successfully called self.__data_reader._on_data_ready. Data: {data_value}, Caller ID: {caller_id_value}")
                except Exception as e:
                    print(f"DEBUG: [DataReaderSource.__poll_for_data] Error calling self.__data_reader._on_data_ready: {e}")

            else:
                print(f"DEBUG: [DataReaderSource.__poll_for_data] No data received from queue (timeout).")
        print(f"DEBUG: [DataReaderSource.__poll_for_data] Thread finished polling as is_running is false.")
