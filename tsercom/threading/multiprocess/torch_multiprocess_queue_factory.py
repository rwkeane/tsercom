"""Defines a factory for creating torch.multiprocessing queues."""

from multiprocessing import Queue as MpQueue
from typing import Tuple, TypeVar, Generic, Optional, cast
from multiprocessing.managers import SyncManager
import torch.multiprocessing as tmp

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
        manager: Optional[SyncManager] = None,
        ctx_method: str = "spawn",
    ) -> None:
        """Initializes the TorchMultiprocessQueueFactory.

        Args:
            manager: An optional multiprocessing.Manager instance. If provided,
                     its Queue() method will be used.
            ctx_method: The multiprocessing context method to use if no manager
                        is provided. Defaults to 'spawn'.
        """
        super().__init__()

        self.__manager = manager
        if manager is None:
            self.__mp_context = tmp.get_context(ctx_method)
        else:
            self.__mp_context = None  # type: ignore

        assert (not self.__mp_context) != (not self.__manager)

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[T], MultiprocessQueueSource[T]]:
        """Creates a pair of torch.multiprocessing queues wrapped in Sink/Source.

        If a manager was provided during factory initialization, its Queue()
        method is used; otherwise, a torch.multiprocessing.Queue is created
        using the factory's context.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances.
        """
        actual_queue: MpQueue[T]
        if self.__manager:
            actual_queue = cast(MpQueue, self.__manager.Queue())
        elif self.__mp_context:
            actual_queue = self.__mp_context.Queue()
        else:
            # This case should not be reached if constructor logic is correct
            raise RuntimeError(
                "TorchMultiprocessQueueFactory not properly initialized with a manager or context."
            )

        sink = MultiprocessQueueSink[T](actual_queue)
        source = MultiprocessQueueSource[T](actual_queue)
        return sink, source
