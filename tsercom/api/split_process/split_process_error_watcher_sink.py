"""Defines an ErrorWatcher that sinks exceptions to a multiprocess queue."""

from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher


class SplitProcessErrorWatcherSink(ErrorWatcher):
    """An ErrorWatcher that forwards caught exceptions to a multiprocess queue.

    This class is typically used in a child process. It monitors threads using
    a `ThreadWatcher`. If `run_until_exception` catches an exception from
    the `ThreadWatcher`, this sink puts the exception onto a
    `MultiprocessQueueSink` before re-raising it. This allows a parent process
    to observe exceptions occurring in the child process.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        exception_queue: MultiprocessQueueSink[Exception],
    ) -> None:
        """Initializes the SplitProcessErrorWatcherSink.

        Args:
            thread_watcher: The ThreadWatcher instance that monitors threads
                            for exceptions.
            exception_queue: The multiprocess queue sink to which caught
                             exceptions will be sent.
        """
        self.__thread_watcher: ThreadWatcher = thread_watcher
        self.__queue: MultiprocessQueueSink[Exception] = exception_queue

    def run_until_exception(self) -> None:
        """Runs until an exception is caught by the ThreadWatcher, then forwards it.

        This method blocks until the underlying `ThreadWatcher` detects an
        exception in one of its monitored threads. The caught exception is
        then put onto the `exception_queue` for another process to consume,
        and finally, the exception is re-raised in the current thread.

        Raises:
            Exception: Re-raises any exception caught by the `ThreadWatcher`.
        """
        try:
            self.__thread_watcher.run_until_exception()
        except Exception as e:
            try:
                self.__queue.put_nowait(e)
                # Attempt to ensure the item is flushed before the process potentially terminates.
                # This is important because put_nowait() might return before the item is actually
                # transmitted by the queue's feeder thread.
                self.__queue.close()
                self.__queue.join_thread()
            except Exception:
                # Log or handle queueing error if necessary. For now, prioritize raising original error.
                # In a real scenario, you might log `q_err` here.
                # import logging
                # logging.getLogger(__name__).error(f"Failed to queue exception to parent: {q_err}", exc_info=True)
                pass  # Original error 'e' will be raised regardless.
            raise e
