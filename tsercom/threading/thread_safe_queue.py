import queue
import threading
from typing import Any


class ThreadSafeQueue:
    def __init__(self) -> None:
        self._queue = queue.Queue()  # type: ignore
        self._lock = threading.Lock()

    def push(
        self, item: Any, block: bool = True, timeout: float | None = None
    ) -> None:
        with self._lock:
            self._queue.put(item, block, timeout)

    def pop(self, block: bool = True, timeout: float | None = None) -> Any:
        with self._lock:
            return self._queue.get(block, timeout)

    def size(self) -> int:
        with self._lock:
            return self._queue.qsize()

    def empty(self) -> bool:
        with self._lock:
            return self._queue.empty()
