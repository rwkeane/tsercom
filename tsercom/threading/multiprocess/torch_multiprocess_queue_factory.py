"""Defines a factory for creating torch.multiprocessing queues."""

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

    def __init__(self, ctx_method: str = "spawn"):
        """Initializes the TorchMultiprocessQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use.
                        Defaults to 'spawn'. Other options include 'fork'
                        and 'forkserver'.
        """
        self._mp_context = mp.get_context(ctx_method)

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """Creates a pair of torch.multiprocessing queues wrapped in Sink/Source.

        These queues are suitable for inter-process communication, especially
        when transferring torch.Tensor objects, as they can utilize shared
        memory to avoid data copying. The underlying queue is a
        torch.multiprocessing.Queue.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a torch.multiprocessing.Queue internally.
        """
        torch_queue: mp.Queue[T] = self._mp_context.Queue()
        # MultiprocessQueueSink and MultiprocessQueueSource are generic and compatible
        # with torch.multiprocessing.Queue, allowing consistent queue interaction.
        sink = MultiprocessQueueSink[T](torch_queue)
        source = MultiprocessQueueSource[T](torch_queue)
        return sink, source
