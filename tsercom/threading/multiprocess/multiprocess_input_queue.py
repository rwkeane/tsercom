import multiprocessing
from queue import Full
from typing import Generic, TypeVar


TQueueType = TypeVar("TQueueType")


class MultiprocessQueueSink(Generic[TQueueType]):

    def __init__(self, queue: multiprocessing.Queue):
        self.__queue = queue

    def put_blocking(self,
                     obj: TQueueType,
                     timeout: float | None = None) -> bool:
        try:
            self.__queue.put(obj, block=True, timeout=timeout)
            return True
        except Full:
            return False

    def put_nowait(self, obj: TQueueType) -> bool:
        try:
            self.__queue.put_nowait(obj)
            return True
        except Full:
            return False
