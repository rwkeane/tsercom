from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List
from collections.abc import Callable

from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.throwing_thread import ThrowingThread
from tsercom.threading.throwing_thread_pool_executor import (
    ThrowingThreadPoolExecutor,
)


class ThreadWatcher(ErrorWatcher):
    """
    This class provides a simple interface for managing a threaded environment,
    as well as tracking and surfacing exceptions that appear from any such
    thread.
    """

    def __init__(self) -> None:
        # Waits for an exception, and passes it to waiting thread when an
        # exception is thrown.
        self.__barrier = threading.Event()
        self.__exceptions_lock = threading.Lock()
        self.__exceptions: List[Exception] = []

    def create_tracked_thread(
        self, target: Callable[[], None]
    ) -> threading.Thread:
        """
        Creates a threading.Thread instance such that exceptions on that thread
        are exposed as they would be for exceptions occuring on the TaskRunner.
        """
        return ThrowingThread(
            target=target, on_error_cb=self.on_exception_seen
        )

    def create_tracked_thread_pool_executor(  # type: ignore
        self, *args, **kwargs
    ) -> ThreadPoolExecutor:
        """
        Creates a ThreadPool such that each thread of the ThreadPool has error
        handling as would be done for an exception on this TaskRunner instance.
        """
        return ThrowingThreadPoolExecutor(  # type: ignore
            error_cb=self.on_exception_seen, *args, **kwargs
        )

    def run_until_exception(self) -> None:
        """
        Runs until an exception is seen, at which point it will be thrown. This
        method is thread safe and can be called from any thread.
        """
        while True:
            self.__barrier.wait()
            with self.__exceptions_lock:
                if len(self.__exceptions) == 0:
                    continue

                # TODO: Change to ExceptionGroup.
                # Requires Python 3.11+. For now, raising the first exception.
                # raise ExceptionGroup(
                #     "Errors hit in async thread(s)!", self.__exceptions
                # )

                raise self.__exceptions[0]

    def check_for_exception(self) -> None:
        """
        If an exception has been seen, throw it. Else, do nothing. This method
        is thread safe and can be called from any thread.
        """
        if not self.__barrier.is_set():
            return

        with self.__exceptions_lock:
            if len(self.__exceptions) == 0:
                return

            # TODO: Change to ExceptionGroup.
            # Requires Python 3.11+. For now, raising the first exception.
            # raise ExceptionGroup(
            #     "Errors hit in async thread(s)!", self.__exceptions
            # )

            raise self.__exceptions[0]

    def on_exception_seen(self, e: Exception) -> None:
        """
        To be called when an exception that the watcher should surface is
        called.
        """
        with self.__exceptions_lock:
            self.__exceptions.append(e)
            self.__barrier.set()
