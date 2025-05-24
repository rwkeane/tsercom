import threading
from typing import Generic, TypeVar

TType = TypeVar("TType")


class Atomic(Generic[TType]):
    """
    This class provides atomic access (via locks) to an underlying type, both
    synchronously and asynchronously.
    """

    def __init__(self, value: TType) -> None:
        """Initializes the Atomic wrapper with an initial value.

        Args:
            value: The initial value to be stored atomically.
        """
        self.__value: TType = value
        self.__lock = threading.Lock()

    def set(self, value: TType) -> None:
        """Atomically sets the stored value.

        This operation acquires a lock to ensure that the value is updated
        atomically and is thread-safe.

        Args:
            value: The new value to store.
        """
        with self.__lock:
            self.__value = value

    def get(self) -> TType:
        """Atomically retrieves the stored value.

        This operation acquires a lock to ensure that the value is read
        atomically and is thread-safe.

        Returns:
            The current value stored atomically.
        """
        with self.__lock:
            return self.__value
