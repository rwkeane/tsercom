"""Defines the DefaultMultiprocessQueueFactory."""

import multiprocessing  # Ensure multiprocessing is imported for .queues
from multiprocessing import Queue as MpQueue
from multiprocessing.managers import SyncManager
from typing import Tuple, TypeVar, Generic, Optional, cast  # Added cast

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

    def __init__(
        self, manager: Optional[SyncManager] = None
    ) -> None:  # Changed type hint to use Optional
        super().__init__()
        self._manager = manager

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
        actual_queue: multiprocessing.queues.Queue[T]  # More specific type
        if self._manager:
            # The manager's Queue() returns a multiprocessing.queues.Queue.
            actual_queue = cast(
                multiprocessing.queues.Queue[T], self._manager.Queue()
            )
        else:
            # This path should ideally not be taken when used by DelegatingMultiprocessQueueFactory
            actual_queue = MpQueue()

        sink = MultiprocessQueueSink[T](actual_queue)
        source = MultiprocessQueueSource[T](actual_queue)
        return sink, source
