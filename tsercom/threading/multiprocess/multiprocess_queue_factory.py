"""
Provides a factory function for creating multiprocess queue pairs.

This module contains the `create_multiprocess_queues` factory function,
which is responsible for instantiating and returning a connected pair of
`MultiprocessQueueSink` and `MultiprocessQueueSource` objects. These objects
serve as wrappers around a shared `multiprocessing.Queue`, facilitating
inter-process communication by providing distinct interfaces for sending (sink)
and receiving (source) data.
"""

from multiprocessing import Queue as MpQueue
from typing import TypeVar

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Type variable for the generic type of the queue.
QueueTypeT = TypeVar("QueueTypeT")


# Factory function to create a pair of connected multiprocess queue sink and source.
def create_multiprocess_queues() -> tuple[
    MultiprocessQueueSink[QueueTypeT],
    MultiprocessQueueSource[QueueTypeT],
]:
    """
    Creates a connected pair of MultiprocessQueueSink and MultiprocessQueueSource.

    These queues are based on `multiprocessing.Queue` and allow for sending
    and receiving data between processes.

    Returns:
        tuple[
            MultiprocessQueueSink[QueueTypeT],
            MultiprocessQueueSource[QueueTypeT],
        ]: A tuple with the sink (for putting) and source (for getting)
           for the created multiprocess queue.
    """
    queue: "MpQueue[QueueTypeT]" = MpQueue()

    sink = MultiprocessQueueSink[QueueTypeT](queue)
    source = MultiprocessQueueSource[QueueTypeT](queue)

    return sink, source
