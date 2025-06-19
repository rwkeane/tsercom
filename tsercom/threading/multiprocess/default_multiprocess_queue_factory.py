"""Defines the DefaultMultiprocessQueueFactory."""

from multiprocessing import Queue as MpQueue
from multiprocessing.managers import SyncManager
from typing import Tuple, TypeVar, Generic, Optional, cast

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

    def __init__(self, manager: Optional[SyncManager] = None) -> None:
        super().__init__()
        self.__manager = manager

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
        actual_queue: MpQueue[T]
        if self.__manager:
            actual_queue = cast(MpQueue[T], self.__manager.Queue())
        else:
            actual_queue = MpQueue()

        sink = MultiprocessQueueSink[T](actual_queue)
        source = MultiprocessQueueSource[T](actual_queue)
        return sink, source

    def shutdown(self) -> None:
        """
        Shuts down the factory and any associated resources.
        For this default factory, if a manager was provided, it attempts to
        shut down the manager.
        """
        if self.__manager and hasattr(self.__manager, "shutdown"):
            # Check if the manager is still running/valid before shutdown
            # This check can be OS and manager-state dependent.
            # A simple hasattr check is done, but more robust checks might be needed
            # if manager could be in an unexpected state.
            try:
                # Example: Check if server address is available if applicable
                # or if manager._state.value == State.STARTED (internal, use with caution)
                # For now, we directly attempt shutdown if manager exists and has method.
                if (
                    hasattr(self.__manager, "_process")
                    and self.__manager._process is not None
                    and self.__manager._process.is_alive()
                ):
                    self.__manager.shutdown()
            except Exception:  # pylint: disable=broad-except
                # Log or handle shutdown error, e.g., if manager already shut down
                # print(f"Error shutting down manager: {e}")
                pass
