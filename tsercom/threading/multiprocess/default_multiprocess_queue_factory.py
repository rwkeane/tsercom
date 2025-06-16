"""Defines the DefaultMultiprocessQueueFactory."""

from multiprocessing import Queue as MpQueue
from typing import Tuple, TypeVar, Generic

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

T = TypeVar("T")


class DefaultMultiprocessQueueFactory(MultiprocessQueueFactory[T], Generic[T]):
    """
    A concrete factory for creating standard multiprocessing queues.

    This factory uses the standard `multiprocessing.Queue`.
    The `create_queues` method returns queues wrapped in
    `MultiprocessQueueSink` and `MultiprocessQueueSource`.
    The `create_queue` method returns a raw `multiprocessing.Queue`.
    """

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """
        Creates a pair of standard multiprocessing queues wrapped in Sink/Source.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a standard `multiprocessing.Queue` internally.
        """
        std_queue: MpQueue[T] = MpQueue()
        sink = MultiprocessQueueSink[T](std_queue)
        source = MultiprocessQueueSource[T](std_queue)
        return sink, source
