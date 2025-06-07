"""Defines `ThreadSafeQueue[T]`, a generic wrapper for `queue.Queue`."""

import queue
import threading
from typing import Generic, TypeVar

# Type variable for the generic type of items in the queue.
T = TypeVar("T")


# A simple thread-safe wrapper around the standard library's queue.Queue.
class ThreadSafeQueue(Generic[T]):
    """A generic thread-safe queue implementation.

    This class wraps the standard `queue.Queue`. While Python's `queue.Queue`
    is already thread-safe for basic operations (put, get), this wrapper
    adds an explicit lock (`threading.Lock`) around these. Users may prefer
    explicit locking patterns or need compound atomic operations via a
    single lock. It also offers a simplified interface with explicit type
    hinting for items stored in the queue.
    """

    def __init__(self) -> None:
        """Initializes a new thread-safe queue."""
        self._queue: queue.Queue[T] = queue.Queue()
        self._lock = (
            threading.Lock()
        )  # Lock for thread safety in queue operations

    def push(
        self, item: T, block: bool = True, timeout: float | None = None
    ) -> None:
        """Puts an item into the queue.

        Args:
            item: The item to be added to the queue.
            block: Whether to block if queue is full (if max size).
                   Defaults to True.
            timeout: Max time in seconds to wait if blocking.
                     None means block indefinitely. Defaults to None.

        Raises:
            queue.Full: If queue full and `block` is False or timeout.
        """
        with self._lock:
            self._queue.put(item, block, timeout)

    def pop(self, block: bool = True, timeout: float | None = None) -> T:
        """Removes and returns an item from the queue.

        Args:
            block: Whether to block if the queue is empty. Defaults to True.
            timeout: Max time in secs to wait for item if blocking.
                     None means block indefinitely. Defaults to None.

        Returns:
            The item removed from the queue.

        Raises:
            queue.Empty: If queue empty and `block` is False or timeout.
        """
        with self._lock:
            return self._queue.get(block, timeout)

    def size(self) -> int:
        """Returns the approximate size of the queue.

        Note: Size may not be exact in multithreaded environments,
        as items can be added/removed between `qsize()` call
        and its result return.

        Returns:
            Approximate number of items in queue.
        """
        with self._lock:
            return self._queue.qsize()

    def empty(self) -> bool:
        """Checks if the queue is empty.

        Note: Similar to `size()`, empty status can change immediately
        after this method returns in multithreaded context.

        Returns:
            True if the queue is empty, False otherwise.
        """
        with self._lock:
            return self._queue.empty()
