from multiprocessing import Process
from typing import Callable, Tuple, Any, Optional
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)

# You might need to import the specific queue type from 'create_multiprocess_queues' if available,
# otherwise, use 'Any' or a generic 'multiprocessing.Queue'.
# from tsercom.threading.multiprocess.multiprocess_queue_factory import SomeQueueType
from multiprocessing import (
    Queue as MultiprocessQueue,
)  # Using standard multiprocessing.Queue as placeholder


class ProcessCreator:
    def create_process(
        self, target: Callable, args: Tuple[Any, ...], daemon: bool
    ) -> Optional[Process]:
        try:
            return Process(target=target, args=args, daemon=daemon)
        except Exception:
            # Consider logging here if a logger is available
            return None


class SplitErrorWatcherSourceFactory:
    def create(
        self,
        thread_watcher: ThreadWatcher,
        error_source_queue: MultiprocessQueue,
    ) -> SplitProcessErrorWatcherSource:
        return SplitProcessErrorWatcherSource(
            thread_watcher, error_source_queue
        )
