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
from typing import Generic, TypeVar, Any, cast

# Type variable for the generic type of items in the queue.
QueueTypeT = TypeVar("QueueTypeT")


# Provides a sink (write-only) interface for a multiprocessing queue.
class MultiprocessQueueSink(Generic[QueueTypeT]):
    """
    Wrapper around `multiprocessing.Queue` for a sink-only interface.

    Handles putting items; generic for queues of any specific type.
    """

    def __init__(self, queue: Any) -> None:
        """
        Initializes with a given multiprocessing queue.

        Args:
            queue: The multiprocessing queue (or a compatible proxy)
                   to be used as the sink.
        """
        self.__queue: Any = queue

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
            actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
            actual_q.put(obj, block=True, timeout=timeout)
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
            actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
            actual_q.put_nowait(obj)
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
        actual_q = cast("MpQueue[QueueTypeT]", self.__queue)
        # Manager queue proxies might use _close() instead of close()
        if hasattr(actual_q, "_close") and callable(actual_q._close):
            actual_q._close()
        elif hasattr(actual_q, "close") and callable(actual_q.close):
            actual_q.close()
        # If neither, this might be an issue or a different type of queue.

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
