"""Multiprocessing utilities, primarily for inter-process queues."""

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

__all__ = [
    "create_multiprocess_queues",
    "MultiprocessQueueSink",
    "MultiprocessQueueSource",
]
