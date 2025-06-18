"""
Defines the  class.

This module provides , a generic, read-only
(source) wrapper around a . It offers a clear,
type-safe interface for receiving items from a shared queue between
processes, focusing on "get" operations and unwrapping items from Envelopes.
"""

from queue import Empty  # Exception for non-blocking get on empty queue.
from typing import Generic, TypeVar, Optional

from tsercom.common.messages import Envelope
from tsercom.threading.multiprocess.base_multiprocess_queue import BaseMultiprocessQueue

# Type variable for the generic type of items originally placed in the envelope.
QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessQueueSource(Generic[QueueTypeT]):
    """
    Wrapper around  for a source-only interface.

    Handles getting items and unwrapping them from  objects.
    Generic for queues that convey items of type  within Envelopes.
    """

    def __init__(self, queue: BaseMultiprocessQueue[Envelope[QueueTypeT]]) -> None:
        """
        Initializes with a given BaseMultiprocessQueue that transports Envelopes.

        Args:
            queue: The BaseMultiprocessQueue instance to be used as source.
                   This queue is expected to yield  items.
        """
        self.__queue: BaseMultiprocessQueue[Envelope[QueueTypeT]] = queue

    def get_blocking(self, timeout: Optional[float] = None) -> Optional[QueueTypeT]:
        """
        Retrieves an item from the queue, blocking if needed, and unwraps it.

        Args:
            timeout: Max time (secs) to wait for an item if the queue is empty.
                      means block indefinitely. Defaults to .

        Returns:
            The unwrapped item of type  from the queue,
            or  if a timeout occurs.
        """
        try:
            # The underlying queue is expected to return Envelope[QueueTypeT]
            envelope = self.__queue.get(block=True, timeout=timeout)
            if envelope is not None:
                return envelope.data
            return None  # Should not happen if get(block=True) only returns None on timeout with specific underlying queues
        except Empty:  # Catches timeout if underlying queue.get() raises it.
            return None
        # Other exceptions (e.g., EOFError, OSError from broken underlying pipes)
        # are allowed to propagate as they indicate severe queue issues.

    def get_or_none(self) -> Optional[QueueTypeT]:
        """
        Retrieves an item from the queue without blocking and unwraps it.

        If the queue is empty, this method returns  immediately.

        Returns:
            The unwrapped item of type  from the queue,
            or  if the queue is empty.
        """
        try:
            # The underlying queue is expected to return Envelope[QueueTypeT]
            # BaseMultiprocessQueue defines get(block=False) for non-blocking.
            envelope = self.__queue.get(block=False)
            if envelope is not None:
                return envelope.data
            return None # Should generally not be reached if get(block=False) raises Empty
        except Empty:  # Queue is empty.
            return None
        # Other exceptions are allowed to propagate.
