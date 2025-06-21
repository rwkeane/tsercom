"""Defines a factory for creating torch.multiprocessing queues."""

import multiprocessing as std_mp
from typing import Tuple, TypeVar, Generic
import torch.multiprocessing as mp

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


class TorchMultiprocessQueueFactory(MultiprocessQueueFactory[T], Generic[T]):
    """
    Provides an implementation of `MultiprocessQueueFactory` specialized for
    `torch.Tensor` objects.

    It utilizes `torch.multiprocessing.Queue` instances, which are chosen
    for their ability to leverage shared memory, thereby optimizing the
    inter-process transfer of tensor data by potentially avoiding costly
    serialization and deserialization. The `create_queues` method returns
    these torch queues wrapped in the standard `MultiprocessQueueSink` and
    `MultiprocessQueueSource` for interface consistency.
    """

    def __init__(
        self,
        ctx_method: str = "spawn",
        context: std_mp.context.BaseContext | None = None,
    ):
        """Initializes the TorchMultiprocessQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use if context
                        is not provided. Defaults to 'spawn'. Other options
                        include 'fork' and 'forkserver'.
            context: An optional existing multiprocessing context to use.
                     If None, a new context is created using ctx_method.
        """
        if context is not None:
            self.__mp_context = context
        else:
            self.__mp_context = mp.get_context(ctx_method)

    def create_queues(
        self,
        max_ipc_queue_size: Optional[int] = None,
        is_ipc_blocking: bool = True,
    ) -> tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """Creates a pair of torch.multiprocessing queues wrapped in Sink/Source.

        These queues are suitable for inter-process communication, especially
        when transferring torch.Tensor objects, as they can utilize shared
        memory to avoid data copying. The underlying queue is a
        torch.multiprocessing.Queue.

        Args:
            max_ipc_queue_size: The maximum size for the created IPC queues.
                                `None` or a non-positive value means unbounded
                                (platform-dependent large size). Defaults to `None`.
            is_ipc_blocking: Determines if `put` operations on the created IPC
                             queues should block when full. Defaults to True.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a torch.multiprocessing.Queue internally.
        """
        # For torch.multiprocessing.Queue, maxsize=0 means platform default (usually large).
        effective_maxsize = 0
        if max_ipc_queue_size is not None and max_ipc_queue_size > 0:
            effective_maxsize = max_ipc_queue_size

        torch_queue: mp.Queue[T] = self.__mp_context.Queue(maxsize=effective_maxsize)
        # MultiprocessQueueSink and MultiprocessQueueSource are generic and compatible
        # with torch.multiprocessing.Queue, allowing consistent queue interaction.
        sink = MultiprocessQueueSink[T](torch_queue, is_blocking=is_ipc_blocking)
        source = MultiprocessQueueSource[T](torch_queue)
        return sink, source
