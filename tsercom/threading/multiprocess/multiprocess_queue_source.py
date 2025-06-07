"""
Defines the `MultiprocessQueueSource[QueueTypeT]` class.

This module provides `MultiprocessQueueSource`, a generic, read-only
(source) wrapper around `multiprocessing.Queue`. It offers a clear,
type-safe interface for receiving items from a shared queue between
processes, focusing on "get" operations.
"""

from multiprocessing import Queue as MpQueue
from queue import (
    Empty,
)  # Exception for non-blocking get on empty queue.
from typing import Generic, TypeVar

# Type variable for the generic type of items in the queue.
QueueTypeT = TypeVar("QueueTypeT")


# Provides a source (read-only) interface for a multiprocessing queue.
class MultiprocessQueueSource(Generic[QueueTypeT]):
    """
    Wrapper around `multiprocessing.Queue` for a source-only interface.

    Handles getting items; generic for queues of any specific type.
    """

    def __init__(self, queue: "MpQueue[QueueTypeT]") -> None:
        """
        Initializes with a given multiprocessing queue.

        Args:
            queue: The multiprocessing queue to be used as source.
        """
        self.__queue: "MpQueue[QueueTypeT]" = queue

    def get_blocking(self, timeout: float | None = None) -> QueueTypeT | None:
        """
        Retrieves item from queue, blocking if needed until item available.

        Args:
            timeout: Max time (secs) to wait for item if queue empty.
                     None means block indefinitely. Defaults to None.

        Returns:
            Optional[QueueTypeT]: Item from queue, or None if timeout.
                                  Note: `multiprocessing.Queue.get()` can
                                  raise `queue.Empty` on timeout (caught),
                                  or `EOFError`/`OSError` if queue broken.
        """
        try:
            return self.__queue.get(block=True, timeout=timeout)
        except Empty:  # Catches timeout for multiprocessing.Queue.get().
            return None
        # Other exceptions (EOFError, OSError) indicate severe queue issues.

    def get_or_none(self) -> QueueTypeT | None:
        """
        Retrieves an item from the queue without blocking.

        If the queue is empty, this method returns immediately.

        Returns:
            Optional[QueueTypeT]: Item from queue, or None if empty.
                                  Can also raise `EOFError`/`OSError`
                                  if queue is broken.
        """
        try:
            return self.__queue.get_nowait()
        except Empty:  # Queue is empty.
            return None
        # Other exceptions (EOFError, OSError) are not caught here.
