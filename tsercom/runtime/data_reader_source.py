import threading
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.multiprocess.multiprocess_output_queue import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TDataType = TypeVar("TDataType", bound=ExposedData)


class DataReaderSource(Generic[TDataType]):
    def __init__(
        self,
        watcher: ThreadWatcher,
        queue: MultiprocessQueueSource[TDataType],
        data_reader: RemoteDataReader[TDataType],
    ):
        self.__queue = queue
        self.__data_reader = data_reader
        self.__watcher = watcher

        self.__thread: threading.Thread | None = None
        self.__is_running = IsRunningTracker(self)

    @property
    def is_running(self):
        return self.__is_running.get()

    def start(self):
        self.__thread = self.__watcher.create_tracked_thread(
            self.__poll_for_data
        )
        self.__thread.start()
        self.__is_running.start()

    def stop(self):
        self.__is_running.stop()

    def __poll_for_data(self):
        while self.__is_running.get():
            data = self.__queue.get_blocking(timeout=1)
            if data is not None:
                self.__data_reader._on_data_ready(data)
