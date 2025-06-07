"""Error watcher source that reads exceptions from a multiprocess queue."""

import logging
import threading

from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker

logger = logging.getLogger(__name__)


class SplitProcessErrorWatcherSource:
    """Monitors a queue for exceptions from another process; reports locally.

    Runs a thread to poll a `MultiprocessQueueSource` for `Exception` objects.
    Received exceptions (likely from another process via a sink) are passed
    to a local `ThreadWatcher`.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        exception_queue: MultiprocessQueueSource[Exception],
    ) -> None:
        """Initializes the SplitProcessErrorWatcherSource.

        Args:
            thread_watcher: Local ThreadWatcher to report exceptions from queue to.
            exception_queue: Multiprocess queue source to read exceptions from.
        """
        self.__thread_watcher: ThreadWatcher = thread_watcher
        self.__queue: MultiprocessQueueSource[Exception] = exception_queue
        self.__is_running: IsRunningTracker = IsRunningTracker()
        self.__thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the thread that polls the exception queue.

        New thread continuously monitors the exception queue. If an exception
        is retrieved, it's reported to the local `ThreadWatcher`.

        Raises:
            RuntimeError: If `start` is called when already running.
        """
        self.__is_running.start()  # Will raise RuntimeError if already running.

        def loop_until_exception() -> None:
            """Polls exception queue, reports exceptions until stopped."""
            while self.__is_running.get():
                # Poll queue with timeout to allow checking is_running.
                remote_exception = self.__queue.get_blocking(timeout=1)
                if remote_exception is not None:
                    try:
                        self.__thread_watcher.on_exception_seen(
                            remote_exception
                        )
                    # pylint: disable=W0718 # Catch any error from queue processing to keep watcher alive
                    except Exception as e_seen:
                        # Long but readable error log
                        logger.error(
                            "Exception occurred within ThreadWatcher.on_exception_seen() "
                            "while handling %s: %s",
                            type(remote_exception).__name__,
                            e_seen,
                            exc_info=True,
                        )
                        # Loop should continue to report further exceptions.

        self.__thread = self.__thread_watcher.create_tracked_thread(
            target=loop_until_exception  # Pass target for clarity
        )
        self.__thread.start()

    def stop(self) -> None:
        """Stops the exception polling thread.

        Signals polling thread to terminate. Thread exits after current poll
        (with timeout) and observing `is_running` state change.

        Raises:
            RuntimeError: If `stop` called when source not currently running.
        """
        self.__is_running.stop()  # Will raise RuntimeError if not running.
        if self.__thread is not None:  # Check if thread exists
            self.__thread.join(timeout=2.0)  # Join with a timeout
            if self.__thread.is_alive():
                # Long warning message
                logger.warning(
                    "SplitProcessErrorWatcherSource: Polling thread %s did not join in 2.0s.",
                    self.__thread.name,
                )

    @property
    def is_running(self) -> bool:
        """Checks if error watcher source is currently polling for exceptions.

        Returns:
            True if polling thread is active, False otherwise.
        """
        return self.__is_running.get()
