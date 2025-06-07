"""
Defines the `ThreadWatcher` class.

This module provides `ThreadWatcher`, an `ErrorWatcher` implementation.
It creates/manages threads (via `ThrowingThread`) and thread pools
(via `ThrowingThreadPoolExecutor`), catching and surfacing exceptions
from them for centralized error monitoring.
"""

import threading
from collections.abc import Callable
from typing import Any, List

from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.throwing_thread import ThrowingThread
from tsercom.threading.throwing_thread_pool_executor import (
    ThrowingThreadPoolExecutor,
)


# Manages threads and surfaces exceptions from them.
class ThreadWatcher(ErrorWatcher):
    """
    Manages a threaded environment, tracks and surfaces exceptions.

    Extends ErrorWatcher to block until an error occurs.
    """

    def __init__(self) -> None:
        """
        Initializes ThreadWatcher.

        Sets up sync primitives for tracking thread exceptions.
        """
        self.__barrier = threading.Event()
        self.__exceptions_lock = threading.Lock()  # Protects __exceptions
        self.__exceptions: List[Exception] = []

    def create_tracked_thread(
        self, target: Callable[[], None], is_daemon: bool = True
    ) -> threading.Thread:
        """
        Creates a `ThrowingThread` tracked by this watcher.

        Exceptions on this thread are caught and reported via
        `on_exception_seen`.

        Args:
            target: Callable invoked when the thread starts.
            is_daemon: Whether the thread is a daemon thread.

        Returns:
            The created `ThrowingThread` object.
        """
        return ThrowingThread(
            target=target, on_error_cb=self.on_exception_seen, daemon=is_daemon
        )

    def create_tracked_thread_pool_executor(
        self, *args: Any, **kwargs: Any
    ) -> ThrowingThreadPoolExecutor:
        """
        Creates a `ThrowingThreadPoolExecutor` instance.

        Threads in this pool report exceptions via `on_exception_seen`.
        Accepts `concurrent.futures.ThreadPoolExecutor` arguments.

        Args:
            *args: Positional args for `ThreadPoolExecutor` constructor.
            **kwargs: Keyword args for `ThreadPoolExecutor` constructor.

        Returns:
            The created `ThrowingThreadPoolExecutor`.
        """
        if "error_cb" in kwargs:
            # User-provided 'error_cb' is overridden.
            del kwargs["error_cb"]

        return ThrowingThreadPoolExecutor(
            self.on_exception_seen, *args, **kwargs
        )

    def run_until_exception(self) -> None:
        """
        Blocks until an exception is caught from a tracked thread.

        Thread-safe. Raises the first caught exception.

        Raises:
            Exception: First exception caught by the watcher.
                       (Future: Python 3.11+ `ExceptionGroup`).
        """
        while True:
            self.__barrier.wait()
            with self.__exceptions_lock:
                if not self.__exceptions:
                    # Spurious wakeup or cleared exception, reset barrier
                    self.__barrier.clear()
                    continue

                # TODO(Python3.11+): Consider ExceptionGroup for multiple errors.
                # Requires Python 3.11+. Raising first exception for now.

                raise self.__exceptions[0]

    def check_for_exception(self) -> None:
        """
        Checks for caught exceptions, raises first one if any.

        Thread-safe and non-blocking. Does nothing if no exceptions.

        Raises:
            Exception: First caught exception if any exist.
                       (Future: Python 3.11+ ExceptionGroup).
        """
        if not self.__barrier.is_set():  # Quick check without lock
            return

        with self.__exceptions_lock:
            if not self.__exceptions:
                return

            # TODO(Python3.11+): Consider ExceptionGroup for multiple errors.
            # Requires Python 3.11+. Raising first exception for now.

            raise self.__exceptions[0]

    def on_exception_seen(self, e: Exception) -> None:
        """
        Callback for exceptions caught by a tracked thread/pool.

        Stores exception, signals waiting threads (e.g., in
        `run_until_exception`) that an exception occurred.

        Args:
            e: The exception that was caught.
        """
        with self.__exceptions_lock:
            self.__exceptions.append(e)
            self.__barrier.set()  # Signal exception availability
