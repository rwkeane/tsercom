"""Defines the abstract base class for multiprocess queue factories and a deprecated factory function.

This module provides the `MultiprocessQueueFactory` ABC, which defines the interface
for queue factories. It also contains the `create_multiprocess_queues` function,
which is considered for deprecation in favor of concrete factory implementations.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Tuple, Generic

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessQueueFactory(ABC, Generic[QueueTypeT]):
    """
    Abstract base class for multiprocess queue factories.

    This class defines the interface that all multiprocess queue factories
    should implement.
    """

    @abstractmethod
    def create_queues(
        self,
    ) -> Tuple[
        MultiprocessQueueSink[QueueTypeT], MultiprocessQueueSource[QueueTypeT]
    ]:
        """
        Creates a pair of queues for inter-process communication.

        Returns:
            A tuple containing two queue instances. The exact type of these
            queues will depend on the specific implementation.
        """
        ...
