"""Defines the DefaultMultiprocessQueueFactory."""

import multiprocessing as std_mp  # Added for context and explicit queue type
from typing import Tuple, TypeVar, Generic, Optional

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
        max_ipc_queue_size: Optional[int] = None,
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
                                `None` or a non-positive value means unbounded
                                (platform-dependent large size). Defaults to `None`.
            is_ipc_blocking: Determines if `put` operations on the created IPC
                             queues should block when full. Defaults to True.
                             This parameter is stored but its application depends
                             on the queue usage logic (e.g., in MultiprocessQueueSink).
        """
        if context is not None:
            self.__mp_context: std_mp.context.BaseContext = context
        else:
            # Ensure std_mp is used here, not torch.multiprocessing
            self.__mp_context = std_mp.get_context(ctx_method)
        self.__max_ipc_queue_size: Optional[int] = max_ipc_queue_size
        self.__is_ipc_blocking: bool = is_ipc_blocking

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
        # A maxsize of <= 0 for multiprocessing.Queue means platform-dependent default (effectively "unbounded").
        effective_maxsize = 0
        if self.__max_ipc_queue_size is not None and self.__max_ipc_queue_size > 0:
            effective_maxsize = self.__max_ipc_queue_size

        std_queue: std_mp.queues.Queue[T] = self.__mp_context.Queue(
            maxsize=effective_maxsize
        )
        sink = MultiprocessQueueSink[T](std_queue, is_blocking=self.__is_ipc_blocking)
        source = MultiprocessQueueSource[T](std_queue)
        return sink, source
