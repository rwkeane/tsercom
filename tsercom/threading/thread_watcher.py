from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, List
from collections.abc import Callable

from tsercom.threading.throwing_thread import ThrowingThread
from tsercom.threading.throwing_thread_pool_executor import ThrowingThreadPoolExecutor

class ThreadWatcher:
    """
    This class provides a simple interface for managing a threaded environment,
    as well as tracking and surfacing exceptions that appear from any such
    thread.
    """
    def __init__(self):
        # Waits for an exception, and passes it to waiting thread when an
        # exception is thrown.
        self.__barrier = threading.Event()
        self.__exceptions_lock = threading.Lock()
        self.__exceptions : List[Exception] = []

    def create_tracked_thread(
            self, target : Callable[[], None]) -> threading.Thread:
        """
        Creates a threading.Thread instance such that exceptions on that thread
        are exposed as they would be for exceptions occuring on the TaskRunner.
        """
        return ThrowingThread(target = target,
                              on_error_cb = self.on_exception_seen)
    
    def create_tracked_thread_pool_executor(
            self, *args, **kwargs) -> ThreadPoolExecutor:
        """
        Creates a ThreadPool such that each thread of the ThreadPool has error
        handling as would be done for an exception on this TaskRunner instance.
        """
        return ThrowingThreadPoolExecutor(
                error_cb = self.on_exception_seen, *args, **kwargs)

    def run_until_exception(self):
        """
        Runs until an exception is seen, at which point it will be thrown.
        """
        while True:
            self.__barrier.wait()
            with self.__exceptions_lock:
                if len(self.__exceptions) == 0:
                    continue

                raise ExceptionGroup("Errors hit in async thread(s)!",
                                     self.__exceptions)

    def on_exception_seen(self, e : Exception):
        with self.__exceptions_lock:
            self.__exceptions.append(e)
            self.__barrier.set()