"""Multiprocessing utilities, primarily for inter-process queues."""

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

__all__ = [
    "MultiprocessQueueSink",
    "MultiprocessQueueSource",
]
