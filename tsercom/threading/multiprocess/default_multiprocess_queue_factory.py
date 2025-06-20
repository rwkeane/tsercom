"""Defines the DefaultMultiprocessQueueFactory."""

import multiprocessing as std_mp  # Added for context and explicit queue type
from typing import Generic, TypeVar

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
    """A concrete factory for creating standard multiprocessing queues.

    This factory uses the standard `multiprocessing.Queue`.
    The `create_queues` method returns queues wrapped in
    `MultiprocessQueueSink` and `MultiprocessQueueSource`.
    The `create_queue` method returns a raw `multiprocessing.Queue`.
    """

    def __init__(
        self,
        ctx_method: str = "spawn",  # Defaulting to 'spawn'
        context: std_mp.context.BaseContext | None = None,
    ):
        """Initializes the DefaultMultiprocessQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use if a specific
                        `context` is not provided. Defaults to 'spawn'.
                        Common options include 'fork', 'spawn', 'forkserver'.
            context: An optional existing multiprocessing context (e.g., from
                     `multiprocessing.get_context()`). If None, a new context
                     is created using the specified `ctx_method`.

        """
        if context is not None:
            self._mp_context: std_mp.context.BaseContext = context
        else:
            # Ensure std_mp is used here, not torch.multiprocessing
            self._mp_context = std_mp.get_context(ctx_method)

    def create_queues(
        self,
    ) -> tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """Creates a pair of standard multiprocessing queues wrapped in Sink/Source,
        using the configured multiprocessing context.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a context-aware `multiprocessing.Queue` internally.

        """
        # The type of queue created by self._mp_context.Queue() is typically
        # multiprocessing.queues.Queue, not the alias MpQueue if it was from
        # `from multiprocessing import Queue`.
        std_queue: std_mp.queues.Queue[T] = self._mp_context.Queue()
        sink = MultiprocessQueueSink[T](std_queue)
        source = MultiprocessQueueSource[T](std_queue)
        return sink, source
