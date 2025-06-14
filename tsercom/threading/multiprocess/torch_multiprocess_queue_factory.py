"""Defines a factory for creating torch.multiprocessing queues."""

from typing import Tuple, Any
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


class TorchMultiprocessQueueFactory(MultiprocessQueueFactory):
    """
    Provides an implementation of `MultiprocessQueueFactory` specialized for
    `torch.Tensor` objects.

    It utilizes `torch.multiprocessing.Queue` instances, which are chosen
    for their ability to leverage shared memory, thereby optimizing the
    inter-process transfer of tensor data by potentially avoiding costly
    serialization and deserialization. The `create_queues` method returns
    these torch queues wrapped in the standard `MultiprocessQueueSink` and
    `MultiprocessQueueSource` for interface consistency, while `create_queue`
    provides direct access to a raw `torch.multiprocessing.Queue`.
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
    ) -> Tuple[MultiprocessQueueSink[Any], MultiprocessQueueSource[Any]]:
        """Creates a pair of torch.multiprocessing queues wrapped in Sink/Source.

        These queues are suitable for inter-process communication, especially
        when transferring torch.Tensor objects, as they can utilize shared
        memory to avoid data copying. The underlying queue is a
        torch.multiprocessing.Queue.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a torch.multiprocessing.Queue internally.
        """
        torch_queue = self._mp_context.Queue()
        # MultiprocessQueueSink and MultiprocessQueueSource are generic and compatible
        # with torch.multiprocessing.Queue, allowing consistent queue interaction.
        sink = MultiprocessQueueSink[Any](torch_queue)
        source = MultiprocessQueueSource[Any](torch_queue)
        return sink, source

    def create_queue(self) -> mp.Queue:
        """Creates a single torch.multiprocessing queue.

        This queue is suitable for inter-process communication, especially
        when transferring torch.Tensor objects.

        Returns:
            A torch.multiprocessing.Queue instance.
        """
        return self._mp_context.Queue()
