"""Defines the DefaultMultiprocessQueueFactory."""

from multiprocessing import Queue as MpQueue
from typing import Tuple, Any

from tsercom.threading.multiprocess.multiprocess_queue_factory import MultiprocessQueueFactory
from tsercom.threading.multiprocess.multiprocess_queue_sink import MultiprocessQueueSink
from tsercom.threading.multiprocess.multiprocess_queue_source import MultiprocessQueueSource


class DefaultMultiprocessQueueFactory(MultiprocessQueueFactory):
    """
    A concrete factory for creating standard multiprocessing queues.

    This factory uses the standard `multiprocessing.Queue`.
    The `create_queues` method returns queues wrapped in
    `MultiprocessQueueSink` and `MultiprocessQueueSource`.
    The `create_queue` method returns a raw `multiprocessing.Queue`.
    """

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[Any], MultiprocessQueueSource[Any]]:
        """
        Creates a pair of standard multiprocessing queues wrapped in Sink/Source.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a standard `multiprocessing.Queue` internally.
        """
        std_queue: MpQueue[Any] = MpQueue()
        sink = MultiprocessQueueSink[Any](std_queue)
        source = MultiprocessQueueSource[Any](std_queue)
        return sink, source

    def create_queue(self) -> MpQueue:
        """
        Creates a single, standard `multiprocessing.Queue`.

        Returns:
            A `multiprocessing.Queue` instance capable of holding any type.
        """
        return MpQueue()
