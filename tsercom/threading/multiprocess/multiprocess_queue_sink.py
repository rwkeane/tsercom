"""
Defines the `MultiprocessQueueSink[QueueTypeT]` class.

This module provides `MultiprocessQueueSink`, a generic, write-only (sink)
wrapper around `multiprocessing.Queue`. It offers a clear, type-safe
interface for sending items to a shared queue between processes,
focusing on "put" operations.
"""

from multiprocessing import Queue as MpQueue
from queue import (
    Full,
)  # Exception for non-blocking put on full queue.
from typing import Generic, TypeVar

# Type variable for the generic type of items in the queue.
QueueTypeT = TypeVar("QueueTypeT")


# Provides a sink (write-only) interface for a multiprocessing queue.
class MultiprocessQueueSink(Generic[QueueTypeT]):
    """
    Wrapper around `multiprocessing.Queue` for a sink-only interface.

    Handles putting items; generic for queues of any specific type.
    """

    def __init__(self, queue: "MpQueue[QueueTypeT]") -> None:
        """
        Initializes with a given multiprocessing queue.

        Args:
            queue: The multiprocessing queue to be used as the sink.
        """
        self.__queue: "MpQueue[QueueTypeT]" = queue

    def put_blocking(
        self, obj: QueueTypeT, timeout: float | None = None
    ) -> bool:
        """
        Puts item into queue, blocking if needed until space available.

        Args:
            obj: The item to put into the queue.
            timeout: Max time (secs) to wait for space if queue full.
                     None means block indefinitely. Defaults to None.

        Returns:
            True if item put successfully, False if timeout occurred
            (queue remained full).
        """
        try:
            self.__queue.put(obj, block=True, timeout=timeout)
            return True
        except Full:  # Timeout occurred and queue is still full.
            return False

    def put_nowait(self, obj: QueueTypeT) -> bool:
        """
        Puts an item into the queue without blocking.

        If the queue is full, this method returns immediately.

        Args:
            obj: The item to put into the queue.

        Returns:
            True if item put successfully, False if queue currently full.
        """
        try:
            self.__queue.put_nowait(obj)
            return True
        except Full:  # Queue is full.
            return False

    def close(self) -> None:
        """
        Closes the queue.

        For a multiprocessing.Queue, this indicates that no more data will be
        put on this queue by this process. The queue will be flushed and
        the background thread will exit when all data is consumed.
        """
        self.__queue.close()

    @property
    def closed(self) -> bool:
        """
        Indicates if the queue is closed (conceptually).

        Note: `multiprocessing.Queue` does not have a public 'closed'
        property that reflects whether `close()` has been called or if items
        can still be retrieved. This property currently returns False as a
        placeholder, as the primary 'closed' state management might be
        handled by delegating wrappers or application logic.
        """
        return False  # Underlying mp.Queue has no simple 'closed' property

    @property
    def empty(self) -> bool:
        """
        Returns True if the queue is empty, False otherwise.
        """
        return self.__queue.empty()

    def __len__(self) -> int:
        """
        Returns the approximate number of items in the queue.
        """
        return self.__queue.qsize()

    @property
    def underlying_queue(self) -> "MpQueue[QueueTypeT]":
        """
        Provides access to the underlying multiprocessing.Queue instance.
        """
        return self.__queue
