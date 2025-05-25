"""
Defines `ThreadSafeQueue[T]`, a generic wrapper around the standard `queue.Queue`.

This module provides the `ThreadSafeQueue` class, which is a generic wrapper
that encapsulates Python's standard `queue.Queue`. It aims to offer a clear,
type-hinted interface for queue operations.
"""
import queue # Standard library queue module
import threading # Standard library threading module
from typing import Any, TypeVar, Generic

# Type variable for the generic type of items in the queue.
T = TypeVar("T")


# A simple thread-safe wrapper around the standard library's queue.Queue.
class ThreadSafeQueue(Generic[T]):
    """
    A generic thread-safe queue implementation.

    This class wraps the standard `queue.Queue`. While Python's `queue.Queue`
    is already documented as thread-safe for its basic operations (put, get),
    this wrapper provides an explicit additional layer of locking (`threading.Lock`)
    around these operations. This can be preferred by users who desire an explicit
    locking pattern or for potential future extensions that might require compound
    atomic operations involving multiple queue interactions under a single lock.
    It also offers a simplified interface with explicit type hinting for the
    items stored in the queue.
    """

    def __init__(self) -> None:
        """
        Initializes a new thread-safe queue.
        """
        self._queue: queue.Queue[T] = queue.Queue()
        self._lock = threading.Lock() # Lock to ensure thread safety for queue operations

    def push(
        self, item: T, block: bool = True, timeout: float | None = None
    ) -> None:
        """
        Puts an item into the queue.

        Args:
            item (T): The item to be added to the queue.
            block (bool): Whether to block if the queue is full (if it has a max size).
                          Defaults to True.
            timeout (Optional[float]): Maximum time in seconds to wait for space if blocking.
                                     If None, blocks indefinitely. Defaults to None.

        Raises:
            queue.Full: If the queue is full and `block` is False or `timeout` is reached.
        """
        with self._lock:
            self._queue.put(item, block, timeout)

    def pop(self, block: bool = True, timeout: float | None = None) -> T:
        """
        Removes and returns an item from the queue.

        Args:
            block (bool): Whether to block if the queue is empty. Defaults to True.
            timeout (Optional[float]): Maximum time in seconds to wait for an item if blocking.
                                     If None, blocks indefinitely. Defaults to None.

        Returns:
            T: The item removed from the queue.

        Raises:
            queue.Empty: If the queue is empty and `block` is False or `timeout` is reached.
        """
        with self._lock:
            return self._queue.get(block, timeout)

    def size(self) -> int:
        """
        Returns the approximate size of the queue.

        Note: The returned size may not be exact in a multithreaded environment
        as items can be added or removed between the time `qsize()` is called
        and the time its result is returned.

        Returns:
            int: The approximate number of items in the queue.
        """
        with self._lock:
            return self._queue.qsize()

    def empty(self) -> bool:
        """
        Checks if the queue is empty.

        Note: Similar to `size()`, the empty status can change immediately after
        this method returns in a multithreaded context.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        with self._lock:
            return self._queue.empty()
