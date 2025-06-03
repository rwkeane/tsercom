"""
Provides generic `Atomic[AtomicTypeT]` for thread-safe value access.

This module defines the `Atomic` class, which wraps a value of a generic type
`AtomicTypeT` and ensures that read (`get`) and write (`set`) operations on this
value are performed atomically via `threading.Lock`. Suitable for
sharing simple values between multiple threads without explicit locking
at the call site.
"""

import threading
from typing import Generic, TypeVar

# Type variable for the generic type stored in Atomic.
AtomicTypeT = TypeVar("AtomicTypeT")


# Provides thread-safe, atomic access to a value.
class Atomic(Generic[AtomicTypeT]):
    """
    This class provides atomic access (via locks) to an underlying type.
    It ensures that operations like getting and setting the value are
    thread-safe.
    """

    def __init__(self, value: AtomicTypeT) -> None:
        """
        Initializes the Atomic wrapper with an initial value.

        Args:
            value (AtomicTypeT): The initial value to be stored atomically.
        """
        self.__value: AtomicTypeT = value
        self.__lock = threading.Lock()  # Lock to ensure atomic operations

    def set(self, value: AtomicTypeT) -> None:
        """
        Atomically sets the stored value.

        This operation acquires a lock to ensure that the value is updated
        atomically and is thread-safe.

        Args:
            value (AtomicTypeT): The new value to store.
        """
        with self.__lock:
            self.__value = value

    def get(self) -> AtomicTypeT:
        """
        Atomically retrieves the stored value.

        This operation acquires a lock to ensure that the value is read
        atomically and is thread-safe.

        Returns:
            AtomicTypeT: The current value stored atomically.
        """
        with self.__lock:
            return self.__value
