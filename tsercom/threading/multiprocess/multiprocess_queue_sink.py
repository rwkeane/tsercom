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
        self._closed: bool = False

    def close(self) -> None:
        """
        Closes the queue sink.

        Marks the sink as closed and calls close() on the underlying
        multiprocessing.Queue, indicating no more data will be put by this process.
        """
        if not self._closed:
            self._closed = True
            # Standard multiprocessing.Queue has a close() method.
            # It's good practice to call it to signal that this end
            # will no longer put items.
            if hasattr(self.__queue, "close"):
                try:
                    self.__queue.close()
                except Exception:  # pylint: disable=broad-except
                    # Broad exception capture if queue close fails for some reason
                    # (e.g., already closed, platform issues). Logging could be added here.
                    pass

    def is_closed(self) -> bool:
        """
        Checks if the queue sink is closed.

        Returns:
            True if the sink is closed, False otherwise.
        """
        return self._closed

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
