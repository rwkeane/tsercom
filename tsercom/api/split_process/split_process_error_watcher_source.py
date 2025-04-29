from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


class SplitProcessErrorWatcherSource:
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        exception_queue: MultiprocessQueueSource[Exception],
    ):
        self.__thread_watcher = thread_watcher
        self.__queue = exception_queue
        self.__is_running = IsRunningTracker()

    def start(self):
        self.__is_running.start()

        def loop_until_exception():
            while self.__is_running.get():
                remote_exception = self.__queue.get_blocking(timeout=1)
                if remote_exception is not None:
                    self.__thread_watcher.on_exception_seen(remote_exception)

        thread = self.__thread_watcher.create_tracked_thread(
            loop_until_exception
        )
        thread.start()

    def stop(self):
        self.__is_running.stop()

    def is_running(self) -> bool:
        return self.__is_running.is_running()
