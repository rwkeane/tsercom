"""Defines SplitProcessErrorWatcherSink for sending exceptions to a multiprocess queue."""

from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher


class SplitProcessErrorWatcherSink(ErrorWatcher):
    """An ErrorWatcher that forwards exceptions to a MultiprocessQueueSink.

    This class serves as a client to a `ThreadWatcher`. When the `ThreadWatcher`
    detects an exception in one of its monitored threads, it notifies this
    sink (via `on_exception_in_tracked_thread`). This sink then places the
    received exception onto a multiprocess queue, allowing another process
    to be made aware of the error.
    """
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        exception_queue: MultiprocessQueueSink[Exception],
    ) -> None:
        """Initializes the SplitProcessErrorWatcherSink.

        Args:
            thread_watcher: The ThreadWatcher instance whose exceptions will be
                            captured and forwarded by this sink. This sink will
                            be registered as a client to this watcher.
            exception_queue: The multiprocess queue sink to which caught
                             exceptions will be sent.
        """
        super().__init__() # Initialize base ErrorWatcher
        # Store the ThreadWatcher and the exception queue.
        self.__thread_watcher: ThreadWatcher = thread_watcher
        self.__queue: MultiprocessQueueSink[Exception] = exception_queue

        # Register this sink as a client of the thread_watcher.
        # This is crucial for on_exception_in_tracked_thread to be called.
        self.__thread_watcher.add_client(self)

    def on_exception_in_tracked_thread(self, exc: Exception) -> None:
        """Handles an exception received from a ThreadWatcher client callback.

        This method is called by the `ThreadWatcher` (to which this sink is
        registered as a client) when one of its tracked threads raises an
        exception. The received exception is then put onto the configured
        multiprocess queue.

        Args:
            exc: The exception caught by the ThreadWatcher.
        """
        # Attempt to send the exception to the multiprocess queue.
        # Using put_nowait to avoid blocking if the queue is full.
        # If the queue is full, the error might not be propagated.
        success = self.__queue.put_nowait(exc)
        if not success:
            # TODO(b/your-bug-tracker): Log or handle if exception cannot be queued.
            # This indicates the error reporting channel is saturated or broken.
            # Silently failing here could mask critical system issues.
            print(f"ERROR: SplitProcessErrorWatcherSink failed to queue exception: {exc}")


    def run_until_exception(self) -> None:
        """Runs the associated ThreadWatcher until an exception occurs.

        This method blocks until the `ThreadWatcher` (passed during initialization)
        detects an exception from one of its monitored threads.
        The `ThreadWatcher` itself will notify this sink via
        `on_exception_in_tracked_thread` when an exception occurs, which then
        queues the exception. This method then re-raises that exception,
        propagating it to the caller.

        Raises:
            Exception: Re-raises any exception caught by the `ThreadWatcher`.
        """
        # The ThreadWatcher will call on_exception_in_tracked_thread on this instance
        # when it catches an exception. That method is responsible for queuing it.
        # This call will block until an exception is caught by ThreadWatcher
        # and then ThreadWatcher will re-raise it.
        self.__thread_watcher.run_until_exception()

    def check_for_exception(self) -> None:
        """Checks if the associated ThreadWatcher has caught an exception and re-raises it.

        If the `ThreadWatcher` has an exception pending, this method will
        cause that exception to be re-raised in the current thread's context.
        The exception would have been previously queued by the
        `on_exception_in_tracked_thread` callback.
        
        Raises:
            Exception: Re-raises any exception caught by the `ThreadWatcher`.
        """
        # This call will re-raise an exception if ThreadWatcher has one stored.
        # The exception would have been queued by on_exception_in_tracked_thread
        # when ThreadWatcher first caught it.
        self.__thread_watcher.check_for_exception()
