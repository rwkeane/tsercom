"""
Defines the `MultiprocessQueueSource[TQueueType]` class.

This module provides the `MultiprocessQueueSource` class, which serves as a
generic, read-only (source) wrapper around a standard `multiprocessing.Queue`.
It is designed to provide a clear and type-safe interface for receiving items
from a queue that is shared between processes, abstracting the underlying queue
and focusing solely on the "get" operations.
"""

import multiprocessing
from queue import (
    Empty,
)  # Exception raised when a non-blocking get is called on an empty queue.
from typing import Generic, TypeVar, cast


# Type variable for the generic type of items in the queue.
TQueueType = TypeVar("TQueueType")


# Provides a source (read-only) interface for a multiprocessing queue.
class MultiprocessQueueSource(Generic[TQueueType]):
    """
    A wrapper around `multiprocessing.Queue` that provides a source-only
    interface for getting items from the queue. This class is generic
    and can handle queues of any specific type.
    """

    def __init__(self, queue: multiprocessing.Queue[TQueueType]) -> None:
        """
        Initializes the MultiprocessQueueSource with a given multiprocessing queue.

        Args:
            queue (multiprocessing.Queue[TQueueType]): The multiprocessing queue
                to be used as the source.
        """
        self.__queue: multiprocessing.Queue[TQueueType] = queue

    def get_blocking(self, timeout: float | None = None) -> TQueueType | None:
        """
        Retrieves an item from the queue, blocking if necessary until an item is available.

        Args:
            timeout (Optional[float]): The maximum time (in seconds) to wait for
                an item if the queue is empty. If None, blocks indefinitely.
                Defaults to None.

        Returns:
            Optional[TQueueType]: The item retrieved from the queue, or None if
                                  a timeout occurred and the queue remained empty.
                                  Note: `multiprocessing.Queue.get()` can raise `queue.Empty`
                                  on timeout, which this method catches and converts to None.
                                  It can also raise `EOFError` or `OSError` if the queue is broken.
        """
        try:
            # Cast is no longer needed if __queue is correctly typed.
            return self.__queue.get(block=True, timeout=timeout)
        except (
            Empty
        ):  # multiprocessing.Queue.get() raises queue.Empty on timeout.
            return None
        # Other exceptions like EOFError or OSError are not caught here,
        # as they indicate a more severe problem with the queue itself.

    def get_or_none(self) -> TQueueType | None:
        """
        Retrieves an item from the queue without blocking.

        If the queue is empty, this method returns immediately.

        Returns:
            Optional[TQueueType]: The item retrieved from the queue, or None if
                                  the queue is currently empty.
                                  It can also raise `EOFError` or `OSError` if the queue is broken.
        """
        try:
            # Cast is no longer needed if __queue is correctly typed.
            return self.__queue.get_nowait()
        except Empty:
            # This exception is raised if the queue is empty.
            return None
        # Other exceptions like EOFError or OSError are not caught here.
