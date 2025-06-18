from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any
from tsercom.common.messages import Envelope # Assuming Envelope is used by these queues

T = TypeVar("T")

class BaseMultiprocessQueue(ABC, Generic[T]):
    """
    Abstract base class for all multiprocess queue implementations.

    Defines the common interface for putting and getting items, checking status,
    and managing the queue lifecycle.
    Items are typically wrapped in an Envelope.
    """

    @abstractmethod
    def put(self, item: Envelope[T], block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Puts an item onto the queue.

        Args:
            item: The item (wrapped in an Envelope) to put onto the queue.
            block: Whether to block if the queue is full.
            timeout: Maximum time to block.

        Raises:
            queue.Full: If the queue is full and blocking is disabled or timeout occurs.
        """
        ...

    @abstractmethod
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Envelope[T]:
        """
        Gets an item from the queue.

        Args:
            block: Whether to block if the queue is empty.
            timeout: Maximum time to block.

        Returns:
            The item (wrapped in an Envelope) from the queue.

        Raises:
            queue.Empty: If the queue is empty and blocking is disabled or timeout occurs.
        """
        ...

    @abstractmethod
    def empty(self) -> bool:
        """
        Checks if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise.
        """
        ...

    @abstractmethod
    def full(self) -> bool:
        """
        Checks if the queue is full.

        Returns:
            True if the queue is full, False otherwise.
        """
        ...

    @abstractmethod
    def qsize(self) -> int:
        """
        Returns the approximate size of the queue.

        Returns:
            The approximate number of items in the queue.
        """
        ...

    @abstractmethod
    def join_thread(self) -> None:
        """
        Joins any background threads associated with the queue.
        This is relevant for queues that use helper threads (e.g., some stdlib.Queue versions).
        If not applicable, this method can be a no-op.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Closes the queue, releasing any underlying resources.
        Further operations on the queue may raise an error after closing.
        """
        ...

    # Optional: Methods for raw tensor handling if this base class should enforce them.
    # However, AggregatingMultiprocessQueue uses a Protocol for this, which is more flexible.
    # So, not adding put_tensor/get_tensor here to keep the base generic.
    # Specific implementations or intermediate bases can add them if needed.

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} qsize={self.qsize()}>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={hex(id(self))})>"
