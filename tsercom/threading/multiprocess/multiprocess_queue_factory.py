import multiprocessing
from typing import TypeVar

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


TQueueType = TypeVar("TQueueType")


def create_multiprocess_queues() -> (
    tuple[
        MultiprocessQueueSink[TQueueType], MultiprocessQueueSource[TQueueType]
    ]
):
    queue = multiprocessing.Queue()

    sink = MultiprocessQueueSink(queue)
    source = MultiprocessQueueSource(queue)

    return sink, source
