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
from typing import Generic, TypeVar, Any, cast

# Type variable for the generic type of items in the queue.
QueueTypeT = TypeVar("QueueTypeT")


# Provides a source (read-only) interface for a multiprocessing queue.
class MultiprocessQueueSource(Generic[QueueTypeT]):
    """
    Wrapper around `multiprocessing.Queue` for a source-only interface.

    Handles getting items; generic for queues of any specific type.
    """

    def __init__(self, queue: Any) -> None:
        """
        Initializes with a given multiprocessing queue.

        Args:
            queue: The multiprocessing queue (or a compatible proxy)
                   to be used as source.
        """
        self.__queue: Any = queue

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
            actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
            return actual_q.get(block=True, timeout=timeout)
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
            actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
            return actual_q.get_nowait()
        except Empty:  # Queue is empty.
            return None
        # Other exceptions (EOFError, OSError) are not caught here.

    def close(self) -> None:
        """
        Closes the queue and joins the background feeder thread.

        For a multiprocessing.Queue, `close()` indicates that no more data
        will be put on this queue by this process. The queue will be flushed.
        `join_thread()` waits for the background feeder thread to terminate,
        which ensures that all data in the buffer has been flushed to the pipe.
        """
        actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
        # Manager queue proxies might use _close() instead of close()
        if hasattr(actual_q, "_close") and callable(actual_q._close):
            actual_q._close()
        elif hasattr(actual_q, "close") and callable(actual_q.close):
            actual_q.close()
        # If neither, this might be an issue or a different type of queue.

        # It's good practice to join the thread for source queues
        # to ensure all buffered items are processed or flushed.
        if hasattr(
            actual_q, "join_thread"
        ):  # join_thread is specific to mp.Queue
            actual_q.join_thread()

    @property
    def closed(self) -> bool:
        """
        Indicates if the queue is closed (conceptually).

        Note: `multiprocessing.Queue` does not have a public 'closed'
        property. This property currently returns False as a placeholder.
        """
        return False  # Underlying mp.Queue has no simple 'closed' property

    @property
    def empty(self) -> bool:
        """
        Returns True if the queue is empty, False otherwise.
        """
        actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
        return actual_q.empty()

    def __len__(self) -> int:
        """
        Returns the approximate number of items in the queue.
        """
        actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
        return actual_q.qsize()

    @property
    def underlying_queue(self) -> "MpQueue[QueueTypeT]":
        """
        Provides access to the underlying multiprocessing.Queue instance.
        """
        return cast("MpQueue[QueueTypeT]", self.__queue)
