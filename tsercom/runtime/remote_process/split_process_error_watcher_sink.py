from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_sink import MultiprocessQueueSink
from tsercom.threading.thread_watcher import ThreadWatcher


class SplitProcessErrorWatcherSink(ErrorWatcher):
    def __init__(self,
                 thread_watcher : ThreadWatcher,
                 exception_queue: MultiprocessQueueSink[Exception]):
        self.__thread_watcher = thread_watcher
        self.__queue = exception_queue

    def run_until_exception(self) -> None:
        try:
            self.__thread_watcher.run_until_exception()
        except Exception as e:
            self.__queue.put_nowait(e)
            raise e