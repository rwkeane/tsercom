"""
Defines the `MultiprocessQueueSink[TQueueType]` class.

This module provides the `MultiprocessQueueSink` class, which serves as a
generic, write-only (sink) wrapper around a standard `multiprocessing.Queue`.
It is designed to provide a clear and type-safe interface for sending items
to a queue that is shared between processes, abstracting the underlying queue
and focusing solely on the "put" operations.
"""
import multiprocessing
from queue import Full # Exception raised when a non-blocking put is called on a full queue.
from typing import Generic, TypeVar


# Type variable for the generic type of items in the queue.
TQueueType = TypeVar("TQueueType")


# Provides a sink (write-only) interface for a multiprocessing queue.
class MultiprocessQueueSink(Generic[TQueueType]):
    """
    A wrapper around `multiprocessing.Queue` that provides a sink-only
    interface for putting items into the queue. This class is generic
    and can handle queues of any specific type.
    """

    def __init__(self, queue: multiprocessing.Queue) -> None:
        """
        Initializes the MultiprocessQueueSink with a given multiprocessing queue.

        Args:
            queue (multiprocessing.Queue[TQueueType]): The multiprocessing queue
                to be used as the sink.
        """
        self.__queue = queue

    def put_blocking(
        self, obj: TQueueType, timeout: float | None = None
    ) -> bool:
        """
        Puts an item into the queue, blocking if necessary until space is available.

        Args:
            obj (TQueueType): The item to put into the queue.
            timeout (Optional[float]): The maximum time (in seconds) to wait for
                space if the queue is full. If None, blocks indefinitely.
                Defaults to None.

        Returns:
            bool: True if the item was successfully put into the queue,
                  False if a timeout occurred (and the queue remained full).
        """
        try:
            # Attempt to put the item with blocking and optional timeout.
            self.__queue.put(obj, block=True, timeout=timeout)
            return True
        except Full:
            # This exception is raised if the timeout occurs and the queue is still full.
            return False

    def put_nowait(self, obj: TQueueType) -> bool:
        """
        Puts an item into the queue without blocking.

        If the queue is full, this method returns immediately.

        Args:
            obj (TQueueType): The item to put into the queue.

        Returns:
            bool: True if the item was successfully put into the queue,
                  False if the queue is currently full.
        """
        try:
            # Attempt to put the item without blocking.
            self.__queue.put_nowait(obj)
            return True
        except Full:
            # This exception is raised if the queue is full.
            return False
