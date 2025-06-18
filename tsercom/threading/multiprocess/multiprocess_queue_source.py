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
from typing import Generic, TypeVar, cast

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
        self._closed: bool = False

    def close(self) -> None:
        """
        Closes the queue source.

        Marks the source as closed and calls close() on the underlying
        multiprocessing.Queue. This is important to allow resources to be freed
        and for processes waiting on the queue to potentially unblock.
        """
        if not self._closed:
            self._closed = True
            if hasattr(self.__queue, "close"):
                try:
                    self.__queue.close()
                except Exception:  # pylint: disable=broad-except
                    # Log or handle potential errors on queue close
                    pass

    def is_closed(self) -> bool:
        """
        Checks if the queue source is closed.

        Returns:
            True if the source is closed, False otherwise.
        """
        return self._closed

    def get(self, timeout: float | None = None) -> QueueTypeT:
        """
        Retrieves an item from the queue, blocking if necessary.

        Args:
            timeout: Max time (secs) to wait for an item. If None, blocks indefinitely.
                     If 0, it's effectively non-blocking (raises Empty if no item).

        Returns:
            The item from the queue.

        Raises:
            queue.Empty: If the queue is empty and timeout is reached (or if non-blocking
                         and queue is empty).
            EOFError/OSError: If the queue is broken.
        """
        if self._closed:
            raise RuntimeError("Cannot get from a closed queue.")
        # Underlying queue.get() raises queue.Empty on timeout.
        return self.__queue.get(block=True, timeout=timeout)

    def get_blocking(self, timeout: float | None = None) -> QueueTypeT | None:
        """
        Retrieves item from queue, blocking if needed until item available.
        This version returns None on timeout instead of raising queue.Empty.

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
            if self._closed:  # Check again, in case closed during blocking
                raise RuntimeError("Queue was closed during blocking get.")
            return self.__queue.get(block=True, timeout=timeout)
        except Empty:  # Catches timeout for multiprocessing.Queue.get().
            return None
        # Other exceptions (EOFError, OSError) indicate severe queue issues.

    def get_nowait(self) -> QueueTypeT:
        """
        Retrieves an item from the queue without blocking.

        Returns:
            Item from queue.

        Raises:
            queue.Empty: If the queue is empty.
            EOFError/OSError: If the queue is broken.
        """
        if self._closed:
            raise RuntimeError("Cannot get_nowait from a closed queue.")
        # This will raise queue.Empty if no item is available.
        return self.__queue.get_nowait()
