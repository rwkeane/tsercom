"""Defines an error watcher source that reads exceptions from a multiprocess queue."""

import logging  # Add logging import
import threading
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker

logger = logging.getLogger(__name__)


class SplitProcessErrorWatcherSource:
    """Monitors a queue for exceptions from another process and reports them locally.

    This class runs a dedicated thread to poll a `MultiprocessQueueSource`
    for `Exception` objects. When an exception is received (presumably from a
    different process that put it onto the queue via a corresponding sink),
    it's passed to a local `ThreadWatcher` instance.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        exception_queue: MultiprocessQueueSource[Exception],
    ) -> None:
        """Initializes the SplitProcessErrorWatcherSource.

        Args:
            thread_watcher: The local ThreadWatcher to which exceptions
                            received from the queue will be reported.
            exception_queue: The multiprocess queue source from which
                             exceptions are read.
        """
        self.__thread_watcher: ThreadWatcher = thread_watcher
        self.__queue: MultiprocessQueueSource[Exception] = exception_queue
        self.__is_running: IsRunningTracker = IsRunningTracker()
        self.__thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the thread that polls the exception queue.

        A new thread is created that continuously monitors the exception queue.
        If an exception is retrieved, it's reported to the local `ThreadWatcher`.

        Raises:
            RuntimeError: If `start` is called when the source is already running.
        """
        self.__is_running.start()  # Will raise RuntimeError if already running.

        def loop_until_exception() -> None:
            """Polls the exception queue and reports exceptions until stopped."""
            while self.__is_running.get():
                # Poll the queue with a timeout to allow checking is_running periodically.
                remote_exception = self.__queue.get_blocking(timeout=1)
                if remote_exception is not None:
                    try:
                        self.__thread_watcher.on_exception_seen(
                            remote_exception
                        )
                    except Exception as e_seen:
                        logger.error(
                            f"Exception occurred within ThreadWatcher.on_exception_seen() "
                            f"while handling {type(remote_exception).__name__}: {e_seen}",
                            exc_info=True,
                        )
                        # Loop should continue to report further exceptions.

        self.__thread = self.__thread_watcher.create_tracked_thread(
            target=loop_until_exception  # Pass target for clarity
        )
        self.__thread.start()

    def stop(self) -> None:
        """Stops the exception polling thread.

        Signals the polling thread to terminate. The thread will exit after its
        current polling attempt (with timeout) completes and it observes the
        changed `is_running` state.

        Raises:
            RuntimeError: If `stop` is called when the source is not currently running.
        """
        self.__is_running.stop()  # Will raise RuntimeError if not running.
        if self.__thread is not None:  # Check if thread exists
            self.__thread.join(timeout=2.0)  # Join with a timeout
            if self.__thread.is_alive():
                logger.warning(
                    f"SplitProcessErrorWatcherSource: Polling thread {self.__thread.name} "
                    f"did not join in 2.0s."
                )

    @property
    def is_running(self) -> bool:
        """Checks if the error watcher source is currently polling for exceptions.

        Returns:
            True if the polling thread is active, False otherwise.
        """
        return self.__is_running.get()
