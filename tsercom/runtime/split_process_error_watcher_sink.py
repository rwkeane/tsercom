from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_output_queue import (
    MultiprocessQueueSource,
)


class SplitThreadErrorWatcherSink(ErrorWatcher):
    def __init__(self, exception_queue: MultiprocessQueueSource[Exception]):
        self.__queue = exception_queue

    def run_until_exception(self) -> None:
        """
        Runs until an exception is seen, at which point it will be thrown.
        """
        remote_exception = self.__queue.get_blocking()
        raise remote_exception
