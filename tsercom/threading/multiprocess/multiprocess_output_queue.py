import multiprocessing
from queue import Empty
from typing import Generic, TypeVar


TQueueType = TypeVar("TQueueType")


class MultiprocessQueueSource(Generic[TQueueType]):
    def __init__(self, queue: multiprocessing.Queue):
        self.__queue = queue

    def get_blocking(self, timeout: float | None = None) -> TQueueType | None:
        try:
            return self.__queue.get(blobk=True, timeout=timeout)
        except Empty:
            return None

    def get_or_none(self) -> TQueueType | None:
        try:
            return self.__queue.get_nowait()
        except Empty:
            return None
