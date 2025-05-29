"""
Defines the `ThreadWatcher` class.

This module provides the `ThreadWatcher` class, which is an implementation of
the `ErrorWatcher` interface. It is used for creating and managing threads
(via `ThrowingThread`) and thread pools (via `ThrowingThreadPoolExecutor`),
and for catching and surfacing exceptions that occur within them. This allows
for a centralized way to monitor errors from various background tasks.
"""

import threading
from typing import List, Any
from collections.abc import Callable

from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.throwing_thread import ThrowingThread
from tsercom.threading.throwing_thread_pool_executor import (
    ThrowingThreadPoolExecutor,
)


# Manages threads and surfaces exceptions from them.
class ThreadWatcher(ErrorWatcher):
    """
    This class provides a simple interface for managing a threaded environment,
    as well as tracking and surfacing exceptions that appear from any such
    thread. It extends ErrorWatcher to provide a mechanism for blocking
    until an error occurs.
    """

    def __init__(self) -> None:
        """
        Initializes the ThreadWatcher.

        Sets up synchronization primitives for tracking exceptions from threads.
        """
        self.__barrier = threading.Event()
        # Lock to protect access to the list of exceptions.
        self.__exceptions_lock = threading.Lock()
        self.__exceptions: List[Exception] = []

    def create_tracked_thread(
        self, target: Callable[[], None], is_daemon: bool = True
    ) -> threading.Thread:
        """
        Creates a `ThrowingThread` instance that is tracked by this watcher. # Changed in docstring

        Exceptions occurring on this thread will be caught and reported via
        `on_exception_seen`.

        Args:
            target (Callable[[], None]): The callable object to be invoked when
                                         the thread starts.

        Returns:
            ThrowingThread: The created thread object.
        """
        return ThrowingThread(
            target=target, on_error_cb=self.on_exception_seen, daemon=is_daemon
        )

    def create_tracked_thread_pool_executor(
        self, *args: Any, **kwargs: Any
    ) -> ThrowingThreadPoolExecutor:
        """
        Creates a `ThrowingThreadPoolExecutor` instance.

        Threads within this pool will have their exceptions caught and reported
        via `on_exception_seen`. Accepts the same arguments as
        `concurrent.futures.ThreadPoolExecutor`.

        Args:
            *args (Any): Positional arguments to pass to the ThreadPoolExecutor constructor.
            **kwargs (Any): Keyword arguments to pass to the ThreadPoolExecutor constructor.

        Returns:
            ThrowingThreadPoolExecutor: The created thread pool executor.
        """
        return ThrowingThreadPoolExecutor(
            error_cb=self.on_exception_seen, *args, **kwargs
        )

    def run_until_exception(self) -> None:
        """
        Blocks execution until an exception is caught from one of the tracked threads.

        This method is thread-safe. Once an exception is caught and stored,
        this method will raise the first caught exception.

        Raises:
            Exception: The first exception caught by the watcher.
                       (Future versions might use ExceptionGroup for Python 3.11+).
        """
        while True:
            self.__barrier.wait()
            with self.__exceptions_lock:
                if not self.__exceptions:
                    # Spurious wakeup or exception handled and cleared, reset barrier
                    self.__barrier.clear()
                    continue

                # TODO: Change to ExceptionGroup.
                # Requires Python 3.11+. For now, raising the first exception.

                raise self.__exceptions[0]

    def check_for_exception(self) -> None:
        """
        Checks if any exceptions have been caught and raises the first one if so.

        This method is thread-safe and non-blocking. If no exceptions have
        been caught, it does nothing.

        Raises:
            Exception: The first exception caught by the watcher if any exist.
                       (Future versions might use ExceptionGroup for Python 3.11+).
        """
        # Quick check without lock if barrier is not set
        if not self.__barrier.is_set():
            return

        with self.__exceptions_lock:
            if not self.__exceptions:
                return

            # TODO: Change to ExceptionGroup.
            # Requires Python 3.11+. For now, raising the first exception.

            raise self.__exceptions[0]

    def on_exception_seen(self, e: Exception) -> None:
        """
        Callback method for when an exception is caught by a tracked thread or pool.

        Stores the exception and signals any waiting threads (e.g., in
        `run_until_exception`) that an exception has occurred.

        Args:
            e (Exception): The exception that was caught.
        """
        with self.__exceptions_lock:
            self.__exceptions.append(e)
            self.__barrier.set()  # Signal that an exception is available
