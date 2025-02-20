import queue
import threading


class ThreadSafeQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._lock = threading.Lock()

    def push(self, item, block=True, timeout=None):
        with self._lock:
            self._queue.put(item, block, timeout)

    def pop(self, block=True, timeout=None):
        with self._lock:
            return self._queue.get(block, timeout)

    def size(self):
        with self._lock:
            return self._queue.qsize()

    def empty(self):
        with self._lock:
            return self._queue.empty()