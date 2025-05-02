from typing import Generic, TypeVar

from tsercom.data.event_instance import EventInstance
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TEventType = TypeVar("TEventType")


class EventSource(Generic[TEventType], AsyncPoller[EventInstance[TEventType]]):
    def __init__(self, event_source: MultiprocessQueueSource[TEventType]):
        self.__event_source = event_source
        self.__is_running = IsRunningTracker()

        super().__init__()

    def start(self, thread_watcher: ThreadWatcher):
        self.__is_running.start()

        def loop_until_exception():
            while self.__is_running.get():
                remote_instance = self.__event_source.get_blocking(timeout=1)
                if remote_instance is not None:
                    self.on_available(remote_instance)

        thread = thread_watcher.create_tracked_thread(loop_until_exception)
        thread.start()

    def stop(self):
        self.__is_running.stop()
