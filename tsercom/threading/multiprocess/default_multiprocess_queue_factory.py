"""Defines the DefaultMultiprocessQueueFactory."""

import multiprocessing as std_mp  # Added for context and explicit queue type
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

    def __init__(
        self,
        ctx_method: str = "spawn",  # Defaulting to 'spawn'
        context: std_mp.context.BaseContext | None = None,
        max_ipc_queue_size: int = -1,
        is_ipc_blocking: bool = True,
    ):
        """Initializes the DefaultMultiprocessQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use if a specific
                        `context` is not provided. Defaults to 'spawn'.
                        Common options include 'fork', 'spawn', 'forkserver'.
            context: An optional existing multiprocessing context (e.g., from
                     `multiprocessing.get_context()`). If None, a new context
                     is created using the specified `ctx_method`.
            max_ipc_queue_size: The maximum size for the created IPC queues.
                                A value of -1 or 0 typically means unbounded
                                or platform-dependent large size. Defaults to -1.
            is_ipc_blocking: Determines if `put` operations on the created IPC
                             queues should block when full. Defaults to True.
                             This parameter is stored but its application depends
                             on the queue usage logic (e.g., in MultiprocessQueueSink).
        """
        if context is not None:
            self._mp_context: std_mp.context.BaseContext = context
        else:
            # Ensure std_mp is used here, not torch.multiprocessing
            self._mp_context = std_mp.get_context(ctx_method)
        self._max_ipc_queue_size: int = max_ipc_queue_size
        self._is_ipc_blocking: bool = is_ipc_blocking

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """
        Creates a pair of standard multiprocessing queues wrapped in Sink/Source,
        using the configured multiprocessing context.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a context-aware `multiprocessing.Queue` internally.
        """
        # The type of queue created by self._mp_context.Queue() is typically
        # multiprocessing.queues.Queue, not the alias MpQueue if it was from
        # `from multiprocessing import Queue`.
        # Use self._max_ipc_queue_size for queue creation.
        # A maxsize of <= 0 means platform-dependent default on many systems (effectively "unbounded").
        effective_maxsize = (
            self._max_ipc_queue_size if self._max_ipc_queue_size > 0 else 0
        )
        std_queue: std_mp.queues.Queue[T] = self._mp_context.Queue(
            maxsize=effective_maxsize
        )
        sink = MultiprocessQueueSink[T](std_queue, is_blocking=self._is_ipc_blocking)
        source = MultiprocessQueueSource[T](std_queue)
        return sink, source
