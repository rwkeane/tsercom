"""Defines the DefaultMultiprocessQueueFactory."""

from multiprocessing import Queue as MpQueue
from multiprocessing.managers import SyncManager
from typing import Tuple, TypeVar, Generic, Optional, cast

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

    def __init__(self, manager: Optional[SyncManager] = None) -> None:
        super().__init__()
        self.__manager = manager

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """
        Creates a pair of standard multiprocessing queues wrapped in Sink/Source.

        If a manager was provided during factory initialization, its Queue()
        method is used; otherwise, a standard `multiprocessing.Queue` is created.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances.
        """
        actual_queue: MpQueue[T]
        if self.__manager:
            # Changed cast(MpQueue[T], ...) to cast(MpQueue, ...)
            actual_queue = cast(MpQueue, self.__manager.Queue())
        else:
            actual_queue = MpQueue()

        sink = MultiprocessQueueSink[T](actual_queue)
        source = MultiprocessQueueSource[T](actual_queue)
        return sink, source
