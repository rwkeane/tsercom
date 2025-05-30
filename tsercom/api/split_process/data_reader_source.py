"""Defines DataReaderSource for reading data from a multiprocess queue and forwarding it."""

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
    """Reads data from a MultiprocessQueueSource and forwards it to a RemoteDataReader.

    This class runs a dedicated thread to continuously poll a multiprocess
    queue for incoming data. When data is retrieved, it's passed to the
    `_on_data_ready` method of the configured `RemoteDataReader` instance.
    It manages the lifecycle (start/stop) of the polling thread.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        queue: MultiprocessQueueSource[TDataType],
        data_reader: RemoteDataReader[TDataType],
    ) -> None:
        """Initializes the DataReaderSource.

        Args:
            watcher: A ThreadWatcher instance to monitor the polling thread.
            queue: The multiprocess queue source from which data will be read.
            data_reader: The RemoteDataReader instance to which data will be forwarded.
        """
        self.__queue = queue
        self.__data_reader = data_reader
        self.__watcher = watcher

        self.__thread: threading.Thread | None = None
        self.__is_running = IsRunningTracker()

    @property
    def is_running(self) -> bool:
        """Checks if the data reader source is currently running.

        Returns:
            True if running, False otherwise.
        """
        return self.__is_running.get()

    def start(self) -> None:
        """Starts the data polling thread.

        If already started, this method might have no effect or could raise
        an error depending on IsRunningTracker's behavior (currently seems to allow restart).
        A new thread is created and started to poll for data from the queue.
        """
        self.__is_running.start()
        self.__thread = self.__watcher.create_tracked_thread(
            target=self.__poll_for_data
        )
        self.__thread.start()

    def stop(self) -> None:
        """Stops the data polling thread.

        Signals the polling thread to terminate and then waits for it to join.
        A timeout is used for the join operation.
        """
        self.__is_running.stop()
        if self.__thread:
            self.__thread.join(timeout=5)
            if self.__thread.is_alive():
                raise RuntimeError(
                    f"ERROR: DataReaderSource thread for queue {self.__queue} did not terminate within 5 seconds."
                )

    def __poll_for_data(self) -> None:
        """Continuously polls the queue for data and forwards it.

        This method runs in a dedicated thread. It blocks on fetching data
        from the queue with a timeout. If data is received, it's passed to
        the data_reader. Exceptions during data forwarding are caught to
        keep the polling loop alive.
        """
        while self.__is_running.get():
            data = self.__queue.get_blocking(timeout=1)
            if data is not None:
                try:
                    self.__data_reader._on_data_ready(data)
                except Exception as e:
                    # It's important to catch exceptions here to prevent the
                    # polling thread from dying silently if _on_data_ready
                    # (e.g., in aggregator) raises one.
                    self.__watcher.on_exception_seen(e)
                    raise e
