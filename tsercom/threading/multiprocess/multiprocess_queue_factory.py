"""
Defines the base interface for multiprocess queue factories and provides a
factory function for creating standard multiprocess queue pairs.
"""

from abc import ABC, abstractmethod
from multiprocessing import Queue as MpQueue
from typing import TypeVar, Tuple, Any

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessQueueFactory(ABC):
    """
    Abstract base class for multiprocess queue factories.

    This class defines the interface that all multiprocess queue factories
    should implement.
    """

    @abstractmethod
    def create_queues(self) -> Tuple[Any, Any]:
        """
        Creates a pair of queues for inter-process communication.

        Returns:
            A tuple containing two queue instances. The exact type of these
            queues will depend on the specific implementation.
        """
        ...

    @abstractmethod
    def create_queue(self) -> Any:
        """
        Creates a single queue for inter-process communication.

        Returns:
            A queue instance. The exact type of this queue will depend
            on the specific implementation.
        """
        ...


# This function can be considered for deprecation in favor of DefaultMultiprocessQueueFactory.
def create_multiprocess_queues() -> tuple[
    MultiprocessQueueSink[QueueTypeT],
    MultiprocessQueueSource[QueueTypeT],
]:
    """
    Creates a connected pair of MultiprocessQueueSink and MultiprocessQueueSource
    using standard multiprocessing.Queue.

    These queues are based on `multiprocessing.Queue` and allow for sending
    and receiving data between processes.

    Returns:
        tuple[
            MultiprocessQueueSink[QueueTypeT],
            MultiprocessQueueSource[QueueTypeT],
        ]: A tuple with the sink (for putting) and source (for getting)
           for the created multiprocess queue.
    """
    queue: "MpQueue[QueueTypeT]" = MpQueue()

    sink = MultiprocessQueueSink[QueueTypeT](queue)
    source = MultiprocessQueueSource[QueueTypeT](queue)

    return sink, source
