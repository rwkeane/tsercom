"""Defines the `MultiprocessQueueSink[QueueTypeT]` class.

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
    """Wrapper around `multiprocessing.Queue` for a sink-only interface.

    Handles putting items; generic for queues of any specific type.
    """

    def __init__(self, queue: "MpQueue[QueueTypeT]", is_blocking: bool = True) -> None:
        """Initialize with a given multiprocessing queue.

        Args:
            queue: The multiprocessing queue to be used as the sink.
            is_blocking: If True, `put_blocking` will block if the queue is full.
                         If False, `put_blocking` will behave like `put_nowait`
                         (i.e., non-blocking and potentially lossy if full).
                         Defaults to True.

        """
        self.__queue: MpQueue[QueueTypeT] = queue
        self.__is_blocking: bool = is_blocking

    def put_blocking(self, obj: QueueTypeT, timeout: float | None = None) -> bool:
        """Put item into queue. Behavior depends on `self.__is_blocking`.

        If `self.__is_blocking` is True (default), this method blocks if necessary
        until space is available in the queue or the timeout expires.
        If `self.__is_blocking` is False, this method attempts to put the item
        without blocking (similar to `put_nowait`) and returns immediately.
        In this non-blocking mode, the `timeout` parameter is ignored.

        Args:
            obj: The item to put into the queue.
            timeout: Max time (secs) to wait for space if queue full and
                     `self.__is_blocking` is True. None means block indefinitely.
                     This parameter is ignored if `self.__is_blocking` is False.
                     Defaults to None.

        Returns:
            True if item put successfully.
            If blocking: False if timeout occurred (queue remained full).
            If non-blocking: False if queue was full at the time of call.

        """
        if not self.__is_blocking:
            # Non-blocking behavior: attempt to put, return status.
            try:
                self.__queue.put(obj, block=False)
                return True
            except Full:
                return False
        else:
            # Blocking behavior
            try:
                self.__queue.put(obj, block=True, timeout=timeout)
                return True
            except Full:
                return False

    def put_nowait(self, obj: QueueTypeT) -> bool:
        """Put an item into the queue without blocking.

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
