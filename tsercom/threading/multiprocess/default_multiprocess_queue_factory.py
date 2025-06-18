"""Defines the DefaultMultiprocessQueueFactory and its queue type."""

from multiprocessing import Queue as MpQueue
import queue # For standard queue exceptions
from typing import Tuple, TypeVar, Generic, Optional

from tsercom.threading.multiprocess.base_multiprocess_queue import BaseMultiprocessQueue
from tsercom.threading.multiprocess.base_multiprocess_queue_factory import BaseMultiprocessQueueFactory
# MultiprocessQueueFactory ABC might be different from BaseMultiprocessQueueFactory
# DefaultMultiprocessQueueFactory should implement BaseMultiprocessQueueFactory if it produces BaseMultiprocessQueues.
# The original code had it implementing MultiprocessQueueFactory which returned Sinks/Sources.
# This needs to be reconciled. For AggregatingQueue to work, it needs BaseMultiprocessQueues.

from tsercom.common.messages import Envelope # For type hinting in DefaultStdQueue

T = TypeVar("T")

# Max size for internal queues, can be configured if needed
DEFAULT_MAX_QUEUE_SIZE = 1000

class DefaultStdQueue(BaseMultiprocessQueue[T]):
    """
    A wrapper for the standard `multiprocessing.Queue` that conforms to
    the `BaseMultiprocessQueue` interface.
    """
    def __init__(self, max_size: int = DEFAULT_MAX_QUEUE_SIZE):
        self._max_size = max_size
        self._queue: MpQueue[Envelope[T]] = MpQueue(maxsize=self._max_size)

    def put(self, item: Envelope[T], block: bool = True, timeout: Optional[float] = None) -> None:
        try:
            self._queue.put(item, block=block, timeout=timeout)
        except queue.Full as e: # MpQueue can raise queue.Full
            raise queue.Full from e

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Envelope[T]:
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty as e: # MpQueue can raise queue.Empty
            raise queue.Empty from e

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        # multiprocessing.Queue doesn't have a reliable full() method that works like queue.Queue.
        # It will block on put if full. We can estimate based on qsize if max_size is known and finite.
        # However, qsize is also approximate.
        # For simplicity, returning False, as it will block if truly full.
        # Or, if max_size > 0, we can try: return self._queue.qsize() >= self._max_size
        if self._max_size > 0:
            try:
                return self._queue.qsize() >= self._max_size
            except NotImplementedError: # qsize not implemented on all platforms (e.g. macOS for mp.Queue)
                return False
        return False # If max_size is 0 (infinite), it's never full in that sense.

    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except NotImplementedError:
             return 0 # qsize not implemented on all platforms

    def join_thread(self) -> None:
        # multiprocessing.Queue does not have a join_thread method for task tracking.
        if hasattr(self._queue, "join_thread"):
            self._queue.join_thread() # type: ignore
        pass

    def close(self) -> None:
        if hasattr(self._queue, "close"):
            self._queue.close() # type: ignore
        # For mp.Queue, might also want to call cancel_join_thread if it was used.
        if hasattr(self._queue, "cancel_join_thread"):
            self._queue.cancel_join_thread() # type: ignore


class DefaultMultiprocessQueueFactory(BaseMultiprocessQueueFactory[T], Generic[T]):
    """
    A concrete factory for creating standard multiprocessing queues
    that conform to the BaseMultiprocessQueue interface.
    """
    def __init__(self, max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE):
        self._max_queue_size = max_queue_size

    def create_queue(self) -> BaseMultiprocessQueue[T]:
        """
        Creates a single standard multiprocessing queue instance, wrapped by DefaultStdQueue.
        """
        return DefaultStdQueue[T](max_size=self._max_queue_size)

    # This method is what DelegatingMultiprocessQueueFactory currently calls.
    # It needs to return two BaseMultiprocessQueue[T] instances.
    def create_queues(self) -> Tuple[BaseMultiprocessQueue[T], BaseMultiprocessQueue[T]]:
        """
        Creates a pair of standard multiprocessing queues, each wrapped by DefaultStdQueue.
        This is to satisfy the current usage in DelegatingMultiprocessQueueFactory,
        which expects two queues (even if it only uses one for the default path).
        A cleaner approach would be for Delegating factory to call create_queue().
        """
        # TODO: DelegatingMultiprocessQueueFactory should ideally call create_queue for the default path.
        # For now, providing two, as it expects a tuple and takes the first element.
        return (DefaultStdQueue[T](max_size=self._max_queue_size),
                DefaultStdQueue[T](max_size=self._max_queue_size))
