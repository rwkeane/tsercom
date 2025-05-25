"""
Provides a factory function for creating multiprocess queue pairs.

This module contains the `create_multiprocess_queues` factory function,
which is responsible for instantiating and returning a connected pair of
`MultiprocessQueueSink` and `MultiprocessQueueSource` objects. These objects
serve as wrappers around a shared `multiprocessing.Queue`, facilitating
inter-process communication by providing distinct interfaces for sending (sink)
and receiving (source) data.
"""
import multiprocessing
from typing import TypeVar

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Type variable for the generic type of the queue.
TQueueType = TypeVar("TQueueType")


# Factory function to create a pair of connected multiprocess queue sink and source.
def create_multiprocess_queues() -> (
    tuple[
        MultiprocessQueueSink[TQueueType], MultiprocessQueueSource[TQueueType]
    ]
):
    """
    Creates a connected pair of MultiprocessQueueSink and MultiprocessQueueSource.

    These queues are based on `multiprocessing.Queue` and allow for sending
    and receiving data between processes.

    Returns:
        tuple[MultiprocessQueueSink[TQueueType], MultiprocessQueueSource[TQueueType]]:
            A tuple containing the sink (for putting items) and the source
            (for getting items) for the created multiprocess queue.
    """
    # Create a standard multiprocessing queue.
    queue: multiprocessing.Queue[TQueueType] = multiprocessing.Queue()

    # Wrap the queue with sink and source objects.
    sink = MultiprocessQueueSink[TQueueType](queue)
    source = MultiprocessQueueSource[TQueueType](queue)

    return sink, source
