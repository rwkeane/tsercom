"""Defines an ErrorWatcher that sinks exceptions to a multiprocess queue."""

import logging

from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher


# pylint: disable=R0903 # Abstract interface/protocol class
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
        """Runs until ThreadWatcher catches an exception, then forwards it.

        Blocks until `ThreadWatcher` detects an exception in a monitored
        thread. The caught exception is put onto `exception_queue`
        for another process, then re-raised in the current thread.

        Raises:
            Exception: Re-raises any exception caught by the `ThreadWatcher`.
        """
        try:
            self.__thread_watcher.run_until_exception()
        except Exception as e:
            # Attempt to put the original exception onto the queue.
            # If this fails, log it, but prioritize re-raising the original exception.
            try:
                self.__queue.put_nowait(e)
            except (
                OSError,
                ValueError,
            ) as queue_e:  # More specific exceptions for queue operations
                # ValueError can be raised by pickle if 'e' is unpicklable
                # OSError can be raised for broken pipes etc.
                # queue.Full might also be relevant if using standard 'queue'
                # but MultiprocessQueueSink might have different behavior.
                # Assuming OSError and Value/PickleError are most likely for IPC.
                logger = logging.getLogger(__name__)
                logger.error(
                    "Failed to put exception onto error_queue: %s. Original error: %s",
                    queue_e,
                    e,
                )
            raise e
