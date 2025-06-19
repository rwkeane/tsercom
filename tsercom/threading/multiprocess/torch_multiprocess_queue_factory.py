"""Defines a factory for creating torch.multiprocessing queues."""

# Standard library imports
from multiprocessing import Queue as MpQueue
from multiprocessing.managers import SyncManager
from typing import Tuple, TypeVar, Generic, Optional, cast

# Third-party imports
import torch.multiprocessing as tmp

# Project imports
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
        # ctx_method: str = "spawn", # Removed to allow respecting global context
    ) -> None:
        """Initializes the TorchMultiprocessQueueFactory.

        Args:
            manager: An optional multiprocessing.Manager instance. If provided,
                     its Queue() method will be used. If None,
                     `torch.multiprocessing.get_context()` will be used,
                     respecting the globally set start method.
        """
        super().__init__()

        self.__manager = manager
        if manager is None:
            # Use the current torch multiprocessing context, which should be
            # set by set_torch_mp_start_method_if_needed before factory creation.
            self.__mp_context = tmp.get_context()
        else:
            self.__mp_context = None # Manager will provide its own context for queues

        # Ensure either manager or context is available, but not both if manager implies context
        # If manager is provided, it's responsible for queue creation context.
        # If manager is None, self.__mp_context must have been set.
        assert bool(self.__manager) != bool(self.__mp_context), \
            "Either a manager or an mp_context (derived from no manager) must be exclusively active."

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
            actual_queue = cast(MpQueue[T], self.__manager.Queue())
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

    def shutdown(self) -> None:
        """
        Shuts down the factory and any associated resources.
        For this torch factory, if a manager was provided, it attempts to
        shut down the manager. If a torch multiprocessing context was created,
        it does not require explicit shutdown here as standard Python Queue
        objects managed by a context don't have explicit individual shutdown methods.
        The manager, if it owns the queue, is responsible for its lifecycle.
        """
        if self.__manager and hasattr(self.__manager, "shutdown"):
            # Similar checks and error handling as in DefaultMultiprocessQueueFactory
            try:
                if (
                    hasattr(self.__manager, "_process")
                    and self.__manager._process is not None
                    and self.__manager._process.is_alive()
                ):
                    self.__manager.shutdown()
            except Exception:  # pylint: disable=broad-except
                # Log or handle shutdown error
                pass
        # No specific shutdown needed for self.__mp_context.Queue() instances typically.
        # The context itself doesn't have a public shutdown method in the same way a manager does.
