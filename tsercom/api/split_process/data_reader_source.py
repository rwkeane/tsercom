"""DataReaderSource: reads from multiprocess queue, forwards data."""

import threading
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class DataReaderSource(Generic[DataTypeT]):
    """Reads from MultiprocessQueueSource, forwards to RemoteDataReader.

    Runs a thread to poll a queue for data. Received data is passed
    to `_on_data_ready` of the configured `RemoteDataReader`.
    Manages polling thread lifecycle.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        queue: MultiprocessQueueSource[DataTypeT],
        data_reader: RemoteDataReader[DataTypeT],
    ) -> None:
        """Initializes the DataReaderSource.

        Args:
            watcher: ThreadWatcher to monitor the polling thread.
            queue: Multiprocess queue source for reading data.
            data_reader: RemoteDataReader to forward data to.
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

        A new thread is created and started to poll for data from the queue.

        Raises:
            RuntimeError: If the source is already running.
        """
        self.__is_running.start()
        self.__thread = self.__watcher.create_tracked_thread(
            target=self.__poll_for_data
        )
        self.__thread.start()

    def stop(self) -> None:
        """Stops the data polling thread.

        Signals polling thread to terminate and waits for it to join (with timeout).
        """
        self.__is_running.stop()
        if self.__thread:
            self.__thread.join(timeout=5)
            if self.__thread.is_alive():
                # Long error message
                raise RuntimeError(
                    f"ERROR: DataReaderSource thread for queue {self.__queue} did not terminate within 5 seconds."
                )

    def __poll_for_data(self) -> None:
        """Continuously polls queue for data and forwards it.

        Runs in a dedicated thread. Blocks on fetching data from queue (with
        timeout). Received data is passed to data_reader. Exceptions during
        forwarding are caught to keep polling loop alive.
        """
        while self.__is_running.get():
            data = self.__queue.get_blocking(timeout=1)
            if data is not None:
                try:
                    # pylint: disable=W0212 # Internal callback for client data readiness
                    self.__data_reader._on_data_ready(data)
                except Exception as e:
                    # It's important to catch exceptions here to prevent the
                    # polling thread from dying silently if _on_data_ready
                    # (e.g., in aggregator) raises one.
                    self.__watcher.on_exception_seen(e)
                    raise e
