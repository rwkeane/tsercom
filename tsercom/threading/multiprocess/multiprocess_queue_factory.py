"""Defines the abstract base class for multiprocess queue factories.

This module provides the `MultiprocessQueueFactory` ABC, which defines the interface
for queue factories.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessQueueFactory(ABC, Generic[QueueTypeT]):
    """Abstract base class for multiprocess queue factories.

    This class defines the interface that all multiprocess queue factories
    should implement.
    """

    @abstractmethod
    def create_queues(
        self,
        max_ipc_queue_size: int | None = None,
        is_ipc_blocking: bool = True,
    ) -> tuple[MultiprocessQueueSink[QueueTypeT], MultiprocessQueueSource[QueueTypeT]]:
        """Create a pair of queues for inter-process communication.

        Args:
            max_ipc_queue_size: The maximum size for the created IPC queues.
                                None or non-positive means unbounded.
            is_ipc_blocking: Determines if `put` operations on the created IPC
                             queues should block.

        Returns:
            A tuple containing two queue instances. The exact type of these
            queues will depend on the specific implementation.

        """
        ...
