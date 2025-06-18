"""Factory for creating delegating multiprocess queues."""

import multiprocessing
from multiprocessing.synchronize import Lock as LockType # Use for manager locks too
from typing import Tuple, TypeVar, Generic, Any

from tsercom.common.system.torch_utils import TORCH_IS_AVAILABLE
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.delegating_queue_sink import (
    DelegatingQueueSink,
)
from tsercom.threading.multiprocess.delegating_queue_source import (
    DelegatingQueueSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Conditionally import TorchMultiprocessQueueFactory for type hinting _torch_factory
# The actual import for instantiation is done locally in __init__ if TORCH_IS_AVAILABLE.
if TORCH_IS_AVAILABLE:
    from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
        TorchMultiprocessQueueFactory as TorchFactoryTypeAlias, # Renamed for Pylint C0103
    )
else:
    TorchFactoryTypeAlias = None # type: ignore # Renamed for Pylint C0103


QueueItemT = TypeVar("QueueItemT", bound=Any)  # Type for items in the queue


# pylint: disable=R0903 # Too few public methods (1/2) - Acceptable for a factory
class DelegatingMultiprocessQueueFactory(
    MultiprocessQueueFactory[QueueItemT], Generic[QueueItemT]
):
    """
    A factory that creates DelegatingQueueSink and DelegatingQueueSource pairs.

    These queues use a shared lock and dictionary (from a multiprocessing.Manager)
    to coordinate the lazy selection of an underlying transport queue
    (either Torch or Default), which are pre-created and passed to them.
    The selection is based on the first data object sent.
    """

    _default_factory: DefaultMultiprocessQueueFactory[QueueItemT]
    _torch_factory: "TorchFactoryTypeAlias[QueueItemT] | None" # Updated type hint name

    def __init__(self) -> None:
        """
        Initializes the DelegatingMultiprocessQueueFactory.
        The multiprocessing manager is created here to ensure its lifetime
        is tied to the factory, and thus to the queues it creates.
        """
        self._manager = multiprocessing.Manager()
        self._default_factory = DefaultMultiprocessQueueFactory[QueueItemT]()

        if TORCH_IS_AVAILABLE:
            # pylint: disable=C0415 # Import outside toplevel - Intentional for conditional dependency
            from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
                TorchMultiprocessQueueFactory as ConcreteTorchFactoryInstance,
            )

            self._torch_factory = ConcreteTorchFactoryInstance[QueueItemT]()
        else:
            self._torch_factory = None

    def create_queues(
        self,
    ) -> Tuple[
        DelegatingQueueSink[QueueItemT], DelegatingQueueSource[QueueItemT]
    ]:
        """
        Creates a pair of delegating queues for inter-process communication.

        A multiprocessing.Manager is used to create a shared lock and dictionary
        for coordination. Both default and (if available) Torch underlying
        queues are created upfront and passed to the delegating wrappers.

        Returns:
            A tuple containing (DelegatingQueueSink, DelegatingQueueSource).
        """
        shared_lock: AcquirerProxy = (
            self._manager.Lock()
        )  # Explicitly type here
        shared_dict = self._manager.dict()

        shared_dict["initialized"] = False
        shared_dict["queue_type"] = None  # 'torch' or 'default'

        # Create both types of queue pairs
        default_sink: MultiprocessQueueSink[QueueItemT]
        default_source: MultiprocessQueueSource[QueueItemT]
        default_sink, default_source = self._default_factory.create_queues()

        torch_sink: MultiprocessQueueSink[QueueItemT] | None = None
        torch_source: MultiprocessQueueSource[QueueItemT] | None = None
        if TORCH_IS_AVAILABLE and self._torch_factory:
            torch_sink, torch_source = self._torch_factory.create_queues()

        # Create the delegating wrappers, passing all necessary components
        sink = DelegatingQueueSink[QueueItemT](
            shared_lock=shared_lock,
            shared_dict=shared_dict,
            default_queue_sink=default_sink,
            default_queue_source=default_source,  # Passed to sink for consistency if needed, though sink primarily uses sinks
            torch_queue_sink=torch_sink,
            torch_queue_source=torch_source,  # Passed to sink for consistency
        )

        source = DelegatingQueueSource[QueueItemT](
            shared_lock=shared_lock,
            shared_dict=shared_dict,
            default_queue_sink=default_sink,  # Passed to source for consistency
            default_queue_source=default_source,
            torch_queue_sink=torch_sink,  # Passed to source for consistency
            torch_queue_source=torch_source,
        )

        return sink, source

    # Optional: If explicit cleanup of the manager is ever needed.
    # def shutdown_manager(self):
    #     if hasattr(self, '_manager') and self._manager is not None:
    #         # Consider implications: queues might still be active.
    #         # self._manager.shutdown()
    #         pass
